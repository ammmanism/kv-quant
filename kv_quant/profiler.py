"""Profiling utilities for kernel performance measurement."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileResult:
    """Result of a kernel profile run."""

    mean_us: float
    std_us: float
    min_us: float
    max_us: float
    iterations: int


def profile_kernel(
    fn: Callable[[], None],
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda",
) -> ProfileResult:
    """Profile a kernel function.

    Args:
        fn: Callable taking no arguments (kernel launch).
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.
        device: Device to synchronize.

    Returns:
        ProfileResult with timing statistics.
    """
    raise NotImplementedError("profile_kernel not implemented")


def profile_memory(
    fn: Callable[[], None],
    device: str = "cuda",
) -> dict[str, int]:
    """Profile memory usage of a function.

    Args:
        fn: Callable to profile.
        device: CUDA device.

    Returns:
        Dict with 'allocated_bytes', 'reserved_bytes', 'peak_bytes'.
    """
    raise NotImplementedError("profile_memory not implemented")


def benchmark_throughput(
    fn: Callable[[], None],
    batch_size: int,
    warmup: int = 5,
    iterations: int = 20,
    device: str = "cuda",
) -> float:
    """Benchmark throughput in tokens/second.

    Args:
        fn: Callable that processes batch_size tokens.
        batch_size: Number of tokens per call.
        warmup: Warmup iterations.
        iterations: Timed iterations.
        device: CUDA device.

    Returns:
        Throughput in tokens/second.
    """
    raise NotImplementedError("benchmark_throughput not implemented")
