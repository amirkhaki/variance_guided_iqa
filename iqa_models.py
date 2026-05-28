import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class BackboneSpec:
    backbone_name: str
    feature_layer: str
    weight_layer: Optional[str] = None


def get_backbone_extractors(
    spec: BackboneSpec,
) -> Tuple[nn.Module, Optional[nn.Module], object, str]:
    if spec.backbone_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        normalize = weights.transforms()
    elif spec.backbone_name == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        normalize = weights.transforms()
    else:
        raise ValueError(f"Unsupported backbone: {spec.backbone_name}")

    base_extractor = create_feature_extractor(
        model, return_nodes={spec.feature_layer: "features"}
    )
    weight_extractor = None
    if spec.weight_layer is not None:
        weight_extractor = create_feature_extractor(
            model, return_nodes={spec.weight_layer: "weights"}
        )
    return base_extractor, weight_extractor, normalize, "features"


class IDFIQA(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        normalize,
        feature_node_key: str = "features",
        percent_features_to_keep: float = 0.6,
        window_size: int = 4,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.normalize = normalize
        self.feature_node_key = feature_node_key
        self.percent_features_to_keep = percent_features_to_keep
        self.window_size = window_size
        self.xi = 1e-8

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(self.normalize(x))
        if isinstance(out, dict):
            return out[self.feature_node_key]
        return out

    @staticmethod
    def _gram(feature_map: torch.Tensor) -> torch.Tensor:
        n, c, h, w = feature_map.shape
        feats = feature_map.view(n, c, h * w)
        gram = torch.bmm(feats, feats.transpose(1, 2))
        return gram / (h * w)

    def _select_channels(self, ref: torch.Tensor, dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, c, h_ref, w_ref = ref.shape
        k = max(1, int(c * self.percent_features_to_keep))
        ref_vars = torch.var(ref, dim=(2, 3), unbiased=False)
        _, top_idx = torch.topk(ref_vars, k, dim=1)
        idx_ref = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_ref, w_ref)
        ref_sel = torch.gather(ref, 1, idx_ref)
        _, _, h_dist, w_dist = dist.shape
        idx_dist = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h_dist, w_dist)
        dist_sel = torch.gather(dist, 1, idx_dist)
        return ref_sel, dist_sel

    def forward(self, ref_img: torch.Tensor, dist_img: torch.Tensor) -> torch.Tensor:
        ref = self._get_features(ref_img.to(self.device))
        dist = self._get_features(dist_img.to(self.device))
        ref_sel, dist_sel = self._select_channels(ref, dist)

        gram_ref = self._gram(ref_sel)
        gram_dist = self._gram(dist_sel)
        gram_ref_unf = F.unfold(gram_ref.unsqueeze(1), kernel_size=self.window_size, stride=1)
        gram_dist_unf = F.unfold(gram_dist.unsqueeze(1), kernel_size=self.window_size, stride=1)
        gram_ref_unf = gram_ref_unf.transpose(1, 2)
        gram_dist_unf = gram_dist_unf.transpose(1, 2)

        var_ref = torch.var(gram_ref_unf, dim=2, unbiased=False)
        var_dist = torch.var(gram_dist_unf, dim=2, unbiased=False)
        mean_ref = torch.mean(gram_ref_unf, dim=2, keepdim=True)
        mean_dist = torch.mean(gram_dist_unf, dim=2, keepdim=True)
        covar = torch.mean((gram_ref_unf - mean_ref) * (gram_dist_unf - mean_dist), dim=2)
        local_scores = (2 * covar + self.xi) / (var_ref + var_dist + self.xi)
        return torch.mean(local_scores, dim=1)


class WeightedPatchIDFIQA(IDFIQA):
    def __init__(
        self,
        feature_extractor: nn.Module,
        weight_extractor: nn.Module,
        normalize,
        feature_node_key: str = "features",
        weight_node_key: str = "weights",
        percent_features_to_keep: float = 0.6,
        window_size: int = 4,
        patch_size: int = 8,
        aggregation: str = "max",
        softmax_temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            normalize=normalize,
            feature_node_key=feature_node_key,
            percent_features_to_keep=percent_features_to_keep,
            window_size=window_size,
            device=device,
        )
        self.weight_extractor = weight_extractor.to(self.device).eval()
        for param in self.weight_extractor.parameters():
            param.requires_grad = False

        self.weight_node_key = weight_node_key
        self.patch_size = patch_size
        self.aggregation = aggregation
        self.softmax_temperature = softmax_temperature
        if self.aggregation == "softmax" and self.softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be > 0 for softmax aggregation.")

    def _get_weight_map(self, ref_img: torch.Tensor, shape_hw: Tuple[int, int]) -> torch.Tensor:
        out = self.weight_extractor(self.normalize(ref_img.to(self.device)))
        weight_feats = out[self.weight_node_key] if isinstance(out, dict) else out
        weight_map = torch.norm(weight_feats, p=2, dim=1, keepdim=True)
        return F.interpolate(weight_map, size=shape_hw, mode="bilinear", align_corners=False).squeeze(1)

    def _aggregate_patch_weight(self, patch_weight: torch.Tensor) -> torch.Tensor:
        flat = patch_weight.reshape(patch_weight.shape[0], -1)
        if self.aggregation == "max":
            return torch.max(flat, dim=1)[0]
        if self.aggregation == "average":
            return torch.mean(flat, dim=1)
        if self.aggregation == "uniform":
            return torch.ones(flat.shape[0], device=flat.device)
        if self.aggregation == "softmax":
            scaled = flat / self.softmax_temperature
            return torch.logsumexp(scaled, dim=1) / math.log(flat.shape[1] + 1.0)
        raise ValueError(f"Unsupported aggregation: {self.aggregation}")

    def forward(self, ref_img: torch.Tensor, dist_img: torch.Tensor) -> torch.Tensor:
        ref_feats = self._get_features(ref_img.to(self.device))
        dist_feats = self._get_features(dist_img.to(self.device))
        _, _, h, w = ref_feats.shape
        weight_map = self._get_weight_map(ref_img, (h, w))

        num_patches_h = max(1, h // self.patch_size)
        num_patches_w = max(1, w // self.patch_size)
        patch_h = h // num_patches_h
        patch_w = w // num_patches_w

        patch_scores = []
        patch_weights = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                hs = i * patch_h
                he = (i + 1) * patch_h if i < num_patches_h - 1 else h
                ws = j * patch_w
                we = (j + 1) * patch_w if j < num_patches_w - 1 else w

                ref_patch = ref_feats[:, :, hs:he, ws:we]
                dist_patch = dist_feats[:, :, hs:he, ws:we]
                if ref_patch.shape[2] < self.window_size or ref_patch.shape[3] < self.window_size:
                    continue
                ref_sel, dist_sel = self._select_channels(ref_patch, dist_patch)
                gram_ref = self._gram(ref_sel)
                gram_dist = self._gram(dist_sel)
                gram_ref_unf = F.unfold(gram_ref.unsqueeze(1), kernel_size=self.window_size, stride=1)
                gram_dist_unf = F.unfold(gram_dist.unsqueeze(1), kernel_size=self.window_size, stride=1)
                gram_ref_unf = gram_ref_unf.transpose(1, 2)
                gram_dist_unf = gram_dist_unf.transpose(1, 2)
                var_ref = torch.var(gram_ref_unf, dim=2, unbiased=False)
                var_dist = torch.var(gram_dist_unf, dim=2, unbiased=False)
                mean_ref = torch.mean(gram_ref_unf, dim=2, keepdim=True)
                mean_dist = torch.mean(gram_dist_unf, dim=2, keepdim=True)
                covar = torch.mean((gram_ref_unf - mean_ref) * (gram_dist_unf - mean_dist), dim=2)
                local_scores = (2 * covar + self.xi) / (var_ref + var_dist + self.xi)
                patch_scores.append(torch.mean(local_scores, dim=1))

                patch_weight = weight_map[:, hs:he, ws:we]
                patch_weights.append(self._aggregate_patch_weight(patch_weight))

        if not patch_scores:
            return super().forward(ref_img, dist_img)

        scores = torch.stack(patch_scores, dim=1)
        weights = torch.stack(patch_weights, dim=1)
        weights = weights / (weights.sum(dim=1, keepdim=True) + self.xi)
        return torch.sum(scores * weights, dim=1)
