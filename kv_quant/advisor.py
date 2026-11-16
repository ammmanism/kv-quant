"""Automatic configuration advisor for optimal quantization settings."""

from dataclasses import dataclass
from typing import Literal

from kv_quant.config import CacheConfig, QuantConfig
from kv_quant.vram import VRAMBreakdown


@dataclass(frozen=True)
class AdvisorRecommendation:
    """Recommended configuration with rationale."""
    quant_config: QuantConfig
    cache_config: CacheConfig
    estimated_vram: VRAMBreakdown
    rationale: str
    confidence: float  # 0.0 - 1.0


def recommend_config(
    model_name: str,
    target_vram_mib: int | None = None,
    target_compression: float | None = None,
    quality_priority: Literal["speed", "balanced", "accuracy"] = "balanced",
    device: str = "cuda",
) -> AdvisorRecommendation:
    """Recommend quantization config for a model.

    Args:
        model_name: HuggingFace model identifier.
        target_vram_mib: Target VRAM budget in MiB.
        target_compression: Target compression ratio (vs FP16).
        quality_priority: Optimization priority.
        device: Target device.

    Returns:
        AdvisorRecommendation with config and rationale.
    """
    raise NotImplementedError("recommend_config not implemented")


def auto_tune_group_size(
    model: "torch.nn.Module",
    sample_inputs: dict,
    candidate_sizes: list[int] = [64, 128, 256, 512],
    metric: Literal["perplexity", "latency", "vram"] = "perplexity",
) -> int:
    """Auto-tune group size by evaluating candidates.

    Args:
        model: Model to evaluate.
        sample_inputs: Sample inputs for evaluation.
        candidate_sizes: Group sizes to try.
        metric: Metric to optimize.

    Returns:
        Best group size.
    """
    raise NotImplementedError("auto_tune_group_size not implemented")


def profile_quantization_impact(
    model: "torch.nn.Module",
    sample_inputs: dict,
    configs: list[QuantConfig],
) -> list[tuple[QuantConfig, dict]]:
    """Profile multiple quantization configs.

    Args:
        model: Model to evaluate.
        sample_inputs: Sample inputs.
        configs: List of configs to profile.

    Returns:
        List of (config, metrics) tuples.
    """
    raise NotImplementedError("profile_quantization_impact not implemented")
