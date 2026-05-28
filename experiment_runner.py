import argparse
import copy
import csv
import functools
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, ttest_rel
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

from iqa_models import BackboneSpec, IDFIQA, WeightedPatchIDFIQA, get_backbone_extractors


BASE_IQADATASETS = ["LIVE", "CSIQ", "TID2013", "KADID-10k", "PIPAL"]
PHASE1_DATASETS = ["LIVE", "CSIQ", "TID2013", "KADID-10k", "PIPAL", "JPEG AIC-4"]


def logistic_func(x, beta1, beta2, beta3, beta4, beta5):
    """Five-parameter logistic mapping used for PLCC calibration.

    beta1 controls amplitude, beta2 slope, beta3 midpoint, beta4 linear term,
    and beta5 bias term.
    """
    logistic_part = beta2 * (x - beta3)
    clipped = np.clip(logistic_part, -100, 100)
    return beta1 * (0.5 - 1 / (1 + np.exp(clipped))) + beta4 * x + beta5


def compute_srcc_plcc(preds: np.ndarray, gts: np.ndarray) -> Tuple[float, float, np.ndarray]:
    srcc, _ = spearmanr(preds, gts)
    try:
        initial = [np.max(gts), 10, np.mean(preds), 0.1, 0.1]
        popt, _ = curve_fit(logistic_func, preds, gts, p0=initial, maxfev=10000)
        mapped = logistic_func(preds, *popt)
    except Exception:
        mapped = preds
    plcc, _ = pearsonr(mapped, gts)
    return float(srcc), float(plcc), mapped


def paired_bootstrap_pvalue(
    preds_a: np.ndarray, preds_b: np.ndarray, gts: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(gts)
    diffs_srcc = []
    diffs_plcc = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        srcc_a, plcc_a, _ = compute_srcc_plcc(preds_a[idx], gts[idx])
        srcc_b, plcc_b, _ = compute_srcc_plcc(preds_b[idx], gts[idx])
        diffs_srcc.append(srcc_b - srcc_a)
        diffs_plcc.append(plcc_b - plcc_a)
    p_srcc = 2 * min(np.mean(np.array(diffs_srcc) <= 0), np.mean(np.array(diffs_srcc) >= 0))
    p_plcc = 2 * min(np.mean(np.array(diffs_plcc) <= 0), np.mean(np.array(diffs_plcc) >= 0))
    return float(p_srcc), float(p_plcc)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


@dataclass
class Sample:
    ref_tensor: torch.Tensor
    dist_tensor: torch.Tensor
    score: float
    sample_id: str
    ref_name: str
    dist_name: str


class InMemoryIqaDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "ref_img": s.ref_tensor,
            "dist_img": s.dist_tensor,
            "score": torch.tensor(s.score, dtype=torch.float32),
            "sample_id": s.sample_id,
            "ref_name": s.ref_name,
            "dist_name": s.dist_name,
        }


def load_iqadataset(name: str) -> Dataset:
    from iqadataset import load_dataset_pytorch

    return load_dataset_pytorch(name)


def load_jpeg_aic_dataset(root_dir: str, manifest_path: str) -> InMemoryIqaDataset:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"JPEG AIC manifest not found: {manifest_path}")
    tfm = transforms.ToTensor()
    samples: List[Sample] = []
    with open(manifest_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ref_img", "dist_img", "score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"JPEG AIC manifest missing columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            ref_path = os.path.join(root_dir, row["ref_img"])
            dist_path = os.path.join(root_dir, row["dist_img"])
            ref_img = Image.open(ref_path).convert("RGB")
            dist_img = Image.open(dist_path).convert("RGB")
            samples.append(
                Sample(
                    ref_tensor=tfm(ref_img),
                    dist_tensor=tfm(dist_img),
                    score=float(row["score"]),
                    sample_id=str(i),
                    ref_name=row["ref_img"],
                    dist_name=row["dist_img"],
                )
            )
    return InMemoryIqaDataset(samples)


def infer_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    output_csv: str,
    num_workers: int,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if max_samples is not None and len(dataset) > max_samples:
        idx = random.sample(range(len(dataset)), max_samples)
        dataset = Subset(dataset, idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    rows = []
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Inference {os.path.basename(output_csv)}"):
            ref = batch["ref_img"].to(device)
            dist = batch["dist_img"].to(device)
            scores = model(ref, dist).detach().cpu().numpy()
            gt = batch["score"].cpu().numpy()
            preds.append(scores)
            gts.append(gt)
            for i in range(len(scores)):
                rows.append(
                    {
                        "sample_id": batch["sample_id"][i],
                        "ref_name": batch["ref_name"][i],
                        "dist_name": batch["dist_name"][i],
                        "pred": float(scores[i]),
                        "gt": float(gt[i]),
                    }
                )

    safe_mkdir(os.path.dirname(output_csv))
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "ref_name", "dist_name", "pred", "gt"])
        writer.writeheader()
        writer.writerows(rows)

    pred_arr = np.concatenate(preds)
    gt_arr = np.concatenate(gts)
    return {"pred": pred_arr, "gt": gt_arr}


def evaluate_ssim_like(ref: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ref_gray = ref.mean(dim=1, keepdim=True)
    dist_gray = dist.mean(dim=1, keepdim=True)
    mu_x = ref_gray.mean(dim=(2, 3), keepdim=True)
    mu_y = dist_gray.mean(dim=(2, 3), keepdim=True)
    sigma_x = ((ref_gray - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_y = ((dist_gray - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
    sigma_xy = ((ref_gray - mu_x) * (dist_gray - mu_y)).mean(dim=(2, 3), keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim.view(-1)


def geometric_augment(dist: torch.Tensor) -> torch.Tensor:
    angle = random.uniform(-2.0, 2.0)
    tx = random.randint(-5, 5)
    ty = random.randint(-5, 5)
    scale = random.uniform(0.95, 1.05)
    return TF.affine(dist, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0])


def load_state(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_state(path: str, state: Dict[str, dict]):
    safe_mkdir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def build_models(
    device: torch.device,
    percent: float,
    window: int,
    patch_size: int,
    backbone: str,
    feature_layer: str,
    weight_layer: str,
    aggregation: str = "max",
    softmax_temperature: float = 1.0,
):
    spec = BackboneSpec(backbone_name=backbone, feature_layer=feature_layer, weight_layer=weight_layer)
    feat_extractor, weight_extractor, normalize, node_key = get_backbone_extractors(spec)
    baseline = IDFIQA(
        feature_extractor=copy.deepcopy(feat_extractor),
        normalize=normalize,
        feature_node_key=node_key,
        percent_features_to_keep=percent,
        window_size=window,
        device=device,
    )

    weighted = WeightedPatchIDFIQA(
        feature_extractor=feat_extractor,
        weight_extractor=weight_extractor,
        normalize=normalize,
        feature_node_key=node_key,
        weight_node_key="weights",
        percent_features_to_keep=percent,
        window_size=window,
        patch_size=patch_size,
        aggregation=aggregation,
        softmax_temperature=softmax_temperature,
        device=device,
    )
    return baseline, weighted


def run_phase1_dataset(
    dataset_name: str,
    dataset: Dataset,
    out_dir: str,
    device: torch.device,
    num_workers: int,
    batch_size: int,
):
    baseline, weighted = build_models(
        device=device,
        percent=0.6,
        window=4,
        patch_size=8,
        backbone="vgg16",
        feature_layer="features.24",
        weight_layer="features.17",
        aggregation="max",
    )

    baseline_out = os.path.join(out_dir, "phase1", dataset_name, "baseline_predictions.csv")
    weighted_out = os.path.join(out_dir, "phase1", dataset_name, "patch_weighted_predictions.csv")

    b = infer_dataset(baseline, dataset, device, baseline_out, num_workers, batch_size=batch_size)
    w = infer_dataset(weighted, dataset, device, weighted_out, num_workers, batch_size=batch_size)

    b_srcc, b_plcc, _ = compute_srcc_plcc(b["pred"], b["gt"])
    w_srcc, w_plcc, _ = compute_srcc_plcc(w["pred"], w["gt"])
    p_boot_srcc, p_boot_plcc = paired_bootstrap_pvalue(b["pred"], w["pred"], b["gt"], n_boot=400)
    _, p_ttest = ttest_rel(w["pred"], b["pred"])

    result = {
        "dataset": dataset_name,
        "baseline": {"srcc": b_srcc, "plcc": b_plcc, "predictions": baseline_out},
        "patch_weighted": {"srcc": w_srcc, "plcc": w_plcc, "predictions": weighted_out},
        "delta": {"srcc": w_srcc - b_srcc, "plcc": w_plcc - b_plcc},
        "significance": {
            "paired_bootstrap_p_srcc": p_boot_srcc,
            "paired_bootstrap_p_plcc": p_boot_plcc,
            "paired_ttest_p_preds": float(p_ttest),
        },
    }
    safe_mkdir(os.path.join(out_dir, "phase1"))
    with open(os.path.join(out_dir, "phase1", f"{dataset_name}_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_weight_map_ablation(dataset: Dataset, out_dir: str, device: torch.device, num_workers: int, batch_size: int):
    variants = [
        ("conv2_2", "features.7"),
        ("conv3_1", "features.10"),
        ("conv4_1", "features.17"),
        ("same_as_quality", "features.24"),
    ]
    aggregations = [("max", None), ("average", None), ("uniform", None), ("softmax", 0.5), ("softmax", 1.0), ("softmax", 2.0)]
    results = []
    for label, weight_layer in variants:
        for agg, temp in aggregations:
            _, weighted = build_models(
                device=device,
                percent=0.6,
                window=4,
                patch_size=8,
                backbone="vgg16",
                feature_layer="features.24",
                weight_layer=weight_layer,
                aggregation=agg,
                softmax_temperature=(temp if temp is not None else 1.0),
            )
            out_csv = os.path.join(out_dir, "phase2", "weight_map_ablation", f"{label}_{agg}_{temp}.csv")
            pred = infer_dataset(weighted, dataset, device, out_csv, num_workers, batch_size=batch_size)
            srcc, plcc, _ = compute_srcc_plcc(pred["pred"], pred["gt"])
            results.append(
                {
                    "weight_source": label,
                    "aggregation": agg,
                    "temperature": temp,
                    "srcc": srcc,
                    "plcc": plcc,
                    "predictions": out_csv,
                }
            )
    safe_mkdir(os.path.join(out_dir, "phase2"))
    with open(os.path.join(out_dir, "phase2", "weight_map_ablation_tid2013.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_patch_window_sensitivity(dataset: Dataset, out_dir: str, device: torch.device, num_workers: int, batch_size: int):
    patch_sizes = [4, 8, 16, 32]
    window_sizes = [2, 4, 6, 8]
    rows = []
    for patch in patch_sizes:
        for window in window_sizes:
            _, weighted = build_models(
                device=device,
                percent=0.6,
                window=window,
                patch_size=patch,
                backbone="vgg16",
                feature_layer="features.24",
                weight_layer="features.17",
                aggregation="max",
            )
            out_csv = os.path.join(out_dir, "phase2", "patch_window_sensitivity", f"patch{patch}_window{window}.csv")
            start = time.perf_counter()
            # Use a capped subset for this dense grid so one full sweep remains practical.
            pred = infer_dataset(weighted, dataset, device, out_csv, num_workers, batch_size=batch_size, max_samples=500)
            elapsed = time.perf_counter() - start
            srcc, plcc, _ = compute_srcc_plcc(pred["pred"], pred["gt"])
            rows.append(
                {
                    "patch_size": patch,
                    "window_size": window,
                    "srcc": srcc,
                    "plcc": plcc,
                    "seconds": elapsed,
                    "predictions": out_csv,
                }
            )
    safe_mkdir(os.path.join(out_dir, "phase2"))
    with open(os.path.join(out_dir, "phase2", "patch_window_sensitivity_tid2013.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_threshold_ablation(dataset: Dataset, out_dir: str, device: torch.device, num_workers: int, batch_size: int):
    percents = [0.4, 0.5, 0.6, 0.7, 0.8]
    rows = []
    for p in percents:
        _, weighted = build_models(
            device=device,
            percent=p,
            window=4,
            patch_size=8,
            backbone="vgg16",
            feature_layer="features.24",
            weight_layer="features.17",
            aggregation="max",
        )
        out_csv = os.path.join(out_dir, "phase2", "threshold_ablation", f"percent_{p:.1f}.csv")
        pred = infer_dataset(weighted, dataset, device, out_csv, num_workers, batch_size=batch_size)
        srcc, plcc, _ = compute_srcc_plcc(pred["pred"], pred["gt"])
        rows.append({"percent_features_to_keep": p, "srcc": srcc, "plcc": plcc, "predictions": out_csv})

    safe_mkdir(os.path.join(out_dir, "phase2"))
    with open(os.path.join(out_dir, "phase2", "threshold_ablation_tid2013.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_backbone_comparison(dataset: Dataset, out_dir: str, device: torch.device, num_workers: int, batch_size: int):
    _, weighted = build_models(
        device=device,
        percent=0.6,
        window=4,
        patch_size=8,
        backbone="efficientnet_b4",
        feature_layer="features.5.1.block.1",
        weight_layer="features.3.0.block.1",
        aggregation="max",
    )
    out_csv = os.path.join(out_dir, "phase2", "backbone_comparison", "efficientnet_b4.csv")
    pred = infer_dataset(weighted, dataset, device, out_csv, num_workers, batch_size=batch_size, max_samples=1000)
    srcc, plcc, _ = compute_srcc_plcc(pred["pred"], pred["gt"])
    result = {"backbone": "efficientnet_b4", "srcc": srcc, "plcc": plcc, "predictions": out_csv}
    safe_mkdir(os.path.join(out_dir, "phase2"))
    with open(os.path.join(out_dir, "phase2", "backbone_comparison_tid2013.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_geometric_robustness(dataset: Dataset, out_dir: str, device: torch.device, num_workers: int, batch_size: int):
    baseline, weighted = build_models(
        device=device,
        percent=0.6,
        window=4,
        patch_size=8,
        backbone="vgg16",
        feature_layer="features.24",
        weight_layer="features.17",
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Geometric robustness"):
            ref = batch["ref_img"].to(device)
            dist = batch["dist_img"].to(device)
            dist_aug = geometric_augment(dist)
            gt = batch["score"].cpu().numpy()
            pred_b = baseline(ref, dist_aug).cpu().numpy()
            pred_w = weighted(ref, dist_aug).cpu().numpy()
            pred_s = evaluate_ssim_like(ref, dist_aug).cpu().numpy()
            for i in range(len(gt)):
                rows.append({"gt": float(gt[i]), "baseline": float(pred_b[i]), "patch_weighted": float(pred_w[i]), "ssim": float(pred_s[i])})
    safe_mkdir(os.path.join(out_dir, "phase3"))
    csv_path = os.path.join(out_dir, "phase3", "geometric_robustness_samples.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gt", "baseline", "patch_weighted", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
    gt = np.array([x["gt"] for x in rows])
    base = np.array([x["baseline"] for x in rows])
    patch = np.array([x["patch_weighted"] for x in rows])
    ssim = np.array([x["ssim"] for x in rows])
    result = {
        "baseline": {"srcc": float(spearmanr(base, gt)[0])},
        "patch_weighted": {"srcc": float(spearmanr(patch, gt)[0])},
        "ssim": {"srcc": float(spearmanr(ssim, gt)[0])},
        "samples": csv_path,
    }
    with open(os.path.join(out_dir, "phase3", "geometric_robustness_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result


def run_complexity_analysis(dataset: Dataset, out_dir: str, num_workers: int):
    tfm_dataset = Subset(dataset, list(range(min(100, len(dataset)))))
    results = []
    for device_name in ["cpu", "cuda"]:
        if device_name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(device_name)
        baseline, weighted = build_models(
            device=device,
            percent=0.6,
            window=4,
            patch_size=8,
            backbone="vgg16",
            feature_layer="features.24",
            weight_layer="features.17",
        )
        loader = DataLoader(tfm_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        for method_name, model in [("baseline", baseline), ("patch_weighted", weighted)]:
            start = time.perf_counter()
            with torch.no_grad():
                for batch in loader:
                    _ = model(batch["ref_img"].to(device), batch["dist_img"].to(device))
            total = time.perf_counter() - start
            results.append(
                {
                    "device": device_name,
                    "method": method_name,
                    "avg_seconds_per_pair": total / len(tfm_dataset),
                }
            )
    safe_mkdir(os.path.join(out_dir, "phase3"))
    with open(os.path.join(out_dir, "phase3", "complexity_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_cross_dataset_generalization(
    datasets: Dict[str, Dataset], out_dir: str, device: torch.device, num_workers: int, batch_size: int
):
    tuning_dataset = datasets["TID2013"]
    candidates = [0.4, 0.5, 0.6, 0.7, 0.8]
    best = None
    for p in candidates:
        _, weighted = build_models(
            device=device,
            percent=p,
            window=4,
            patch_size=8,
            backbone="vgg16",
            feature_layer="features.24",
            weight_layer="features.17",
        )
        pred = infer_dataset(
            weighted,
            tuning_dataset,
            device,
            os.path.join(out_dir, "phase3", "cross_dataset", f"tuning_percent_{p:.1f}.csv"),
            num_workers,
            batch_size=batch_size,
        )
        srcc, plcc, _ = compute_srcc_plcc(pred["pred"], pred["gt"])
        if best is None or srcc > best["srcc"]:
            best = {"percent": p, "srcc": srcc, "plcc": plcc}

    test_results = []
    for name in ["LIVE", "CSIQ"]:
        _, weighted = build_models(
            device=device,
            percent=best["percent"],
            window=4,
            patch_size=8,
            backbone="vgg16",
            feature_layer="features.24",
            weight_layer="features.17",
        )
        out_csv = os.path.join(out_dir, "phase3", "cross_dataset", f"test_{name}.csv")
        pred = infer_dataset(weighted, datasets[name], device, out_csv, num_workers, batch_size=batch_size)
        srcc, plcc, _ = compute_srcc_plcc(pred["pred"], pred["gt"])
        test_results.append({"dataset": name, "srcc": srcc, "plcc": plcc, "predictions": out_csv})
    summary = {"tuned_on": "TID2013", "best": best, "tests": test_results}
    with open(os.path.join(out_dir, "phase3", "cross_dataset_generalization.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_formula_justification(out_dir: str):
    text = (
        "The method computes similarity between reference and distorted feature Gram matrices, "
        "after selecting high-variance channels. The patch-weighted extension computes local "
        "scores over feature-space patches and combines them with saliency-like spatial weights "
        "from shallower features. This prioritizes spatially informative regions while preserving "
        "the original global-statistics objective."
    )
    safe_mkdir(os.path.join(out_dir, "phase3"))
    path = os.path.join(out_dir, "phase3", "formula_justification.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    return {"file": path}


def main():
    parser = argparse.ArgumentParser(description="Resumable experiment runner for variance-guided IQA.")
    parser.add_argument("--output-dir", type=str, default="experiment_results")
    parser.add_argument("--state-file", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", nargs="+", default=["phase1", "phase2", "phase3"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-optional", action="store_true")
    parser.add_argument("--jpeg-aic-root", type=str, default=None, help="Root path for JPEG AIC-4 files.")
    parser.add_argument(
        "--jpeg-aic-manifest",
        type=str,
        default=None,
        help="CSV with columns: ref_img,dist_img,score (relative to --jpeg-aic-root).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.abspath(args.output_dir)
    state_file = os.path.abspath(args.state_file or os.path.join(output_dir, "run_state.json"))
    safe_mkdir(output_dir)
    state = load_state(state_file)

    datasets: Dict[str, Dataset] = {}
    for name in BASE_IQADATASETS:
        datasets[name] = load_iqadataset(name)
    if args.jpeg_aic_root and args.jpeg_aic_manifest:
        datasets["JPEG AIC-4"] = load_jpeg_aic_dataset(args.jpeg_aic_root, args.jpeg_aic_manifest)

    tasks = []
    if "phase1" in args.phase:
        for name in PHASE1_DATASETS:
            if name in datasets:
                tasks.append(
                    (
                        "phase1",
                        f"phase1_{name}",
                        functools.partial(
                            run_phase1_dataset,
                            name,
                            datasets[name],
                            output_dir,
                            device,
                            args.num_workers,
                            args.batch_size,
                        ),
                    )
                )
        tasks.append(("phase1", "phase1_main_table", None))

    if "phase2" in args.phase:
        tasks.append(("phase2", "phase2_weight_map_ablation_tid2013", lambda: run_weight_map_ablation(
            datasets["TID2013"], output_dir, device, args.num_workers, args.batch_size
        )))
        tasks.append(("phase2", "phase2_patch_window_sensitivity_tid2013", lambda: run_patch_window_sensitivity(
            datasets["TID2013"], output_dir, device, args.num_workers, args.batch_size
        )))
        tasks.append(("phase2", "phase2_threshold_ablation_tid2013", lambda: run_threshold_ablation(
            datasets["TID2013"], output_dir, device, args.num_workers, args.batch_size
        )))
        if args.run_optional:
            tasks.append(("phase2", "phase2_backbone_comparison_tid2013", lambda: run_backbone_comparison(
                datasets["TID2013"], output_dir, device, args.num_workers, args.batch_size
            )))

    if "phase3" in args.phase:
        tasks.append(("phase3", "phase3_geometric_robustness_tid2013", lambda: run_geometric_robustness(
            datasets["TID2013"], output_dir, device, args.num_workers, args.batch_size
        )))
        tasks.append(("phase3", "phase3_complexity_analysis_tid2013", lambda: run_complexity_analysis(
            datasets["TID2013"], output_dir, args.num_workers
        )))
        if args.run_optional:
            tasks.append(("phase3", "phase3_cross_dataset_generalization", lambda: run_cross_dataset_generalization(
                datasets, output_dir, device, args.num_workers, args.batch_size
            )))
        tasks.append(("phase3", "phase3_formula_justification", lambda: write_formula_justification(output_dir)))

    for phase, task_name, fn in tasks:
        if task_name in state and state[task_name].get("status") == "done" and not args.force:
            print(f"[SKIP] {task_name} (already completed)")
            continue
        if fn is None and task_name == "phase1_main_table":
            rows = []
            for name in PHASE1_DATASETS:
                p = os.path.join(output_dir, "phase1", f"{name}_summary.json")
                if os.path.exists(p):
                    with open(p, "r") as f:
                        rows.append(json.load(f))
            safe_mkdir(os.path.join(output_dir, "phase1"))
            with open(os.path.join(output_dir, "phase1", "main_results_table.json"), "w") as f:
                json.dump(rows, f, indent=2)
            state[task_name] = {"status": "done", "phase": phase, "result": "phase1/main_results_table.json"}
            save_state(state_file, state)
            continue

        print(f"[RUN] {task_name}")
        result = fn()
        state[task_name] = {"status": "done", "phase": phase, "result": result}
        save_state(state_file, state)

    print(f"Completed. State file: {state_file}")


if __name__ == "__main__":
    main()
