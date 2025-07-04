import timeit

import torch


def get_device(index: int = 0) -> torch.device:
    """Use GPU if available, otherwise fallback to CPU."""
    return torch.device(f"cuda:{index}" if torch.cuda.is_available() else "cpu")


def time_matmul(a: torch.Tensor, b: torch.Tensor, trials: int = 5) -> float:
    """Return the average time (in seconds) to compute a @ b over `trials` runs."""
    # Warm-up (avoid cold start effects)
    for _ in range(3):
        _ = a @ b
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def run():
        _ = a @ b
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    total_time = timeit.timeit(run, number=trials)
    return total_time / trials


def main():
    device = get_device()
    print(f"Using device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}")

    # Use larger sizes on GPU
    if device.type == "cuda":
        B, D, K = 16384, 32768, 8192
        dtype = torch.bfloat16  # Can try torch.float16 or torch.bfloat16
        torch.set_float32_matmul_precision("high")  # Enable TF32 on A100
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        B, D, K = 1024, 256, 64
        dtype = torch.float32

    x = torch.ones((B, D), dtype=dtype, device=device)
    w = torch.randn((D, K), dtype=dtype, device=device)

    avg_time = time_matmul(x, w)
    total_flops = 2 * B * D * K
    flops_per_sec = total_flops / avg_time
    tflops = flops_per_sec / 1e12

    print(f"Matrix size: {B}x{D} @ {D}x{K}")
    print(f"Average matmul time: {avg_time:.6f} seconds")
    print(f"Estimated throughput: {tflops:.2f} TFLOP/s")
    print("Compared to A100 throughput: 312 TFLOP/s")


if __name__ == "__main__":
    main()
