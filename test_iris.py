import torch
import os
import time
import numpy as np

import iris_hamming as ih


def main():
    device = "cuda"
    # Verification size (keep small; numpy is O(M^2 * shifts))
    M = 16

    # Dummy iris codes/masks in the "real" shape: (M, 16, 200, 2, 2) bits
    code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
    mask = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

    data_words = ih.pack_theta_major(code)
    mask_words = ih.pack_theta_major(mask)

    data_t = torch.from_numpy(data_words).to(device=device)
    mask_t = torch.from_numpy(mask_words).to(device=device)

    # Single run (GPU) with output + pair collection
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    D, pairs, match_count = ih.masked_hamming_cuda(
        data_t,
        mask_t,
        write_output=True,
        collect_pairs=True,
        threshold=0.5,
        max_pairs=1 << 16,
    )
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    pairs_total = 0.5 * M * (M - 1)
    pairs_per_s = pairs_total / (elapsed_ms * 1e-3) / 1e6
    print(f"M={M}, time={elapsed_ms:.3f} ms, pairs/s={pairs_per_s:.3f} M (31 shifts)")

    # Fetch count and a few pairs
    count = match_count.cpu().item() if match_count.numel() else 0
    print(f"Matches found: {count}")
    if count > 0:
        capped = min(count, pairs.size(0))
        print("Sample pairs:", pairs[: min(8, capped)].cpu())

    # ----------------- NumPy reference verification -----------------
    # Compute min FHD over theta-roll shifts for the lower triangle.
    ref = np.zeros((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(i):
            mi = mask[i].astype(bool)
            di = code[i].astype(bool)
            best = 2.0
            for s in range(-15, 16):
                djr = np.roll(code[j], s, axis=1).astype(bool)   # axis=1 of IrisCode => theta
                mjr = np.roll(mask[j], s, axis=1).astype(bool)
                inter = mi & mjr
                valid = int(inter.sum())
                if valid == 0:
                    continue
                ham = int(((di ^ djr) & inter).sum())
                fhd = ham / valid
                if fhd < best:
                    best = fhd
            ref[i, j] = best if best <= 1.0 else 0.0

    D_cpu = D.cpu().numpy()
    max_err = np.max(np.abs(D_cpu - ref))
    print(f"NumPy check: max_abs_err={max_err:.3e}")
    if max_err > 1e-5:
        # Print a few mismatches
        mism = np.argwhere(np.abs(D_cpu - ref) > 1e-5)
        print("Mismatches (up to 8):")
        for k in range(min(8, mism.shape[0])):
            ii, jj = mism[k]
            print(f"  ({ii},{jj}): gpu={D_cpu[ii,jj]:.6f} numpy={ref[ii,jj]:.6f}")

    # ----------------- Performance-only check (no NumPy validation) -----------------
    # This uses packed int32 words directly (shape [M_perf, 400]) to avoid Python/NumPy overhead.
    M_perf = int(os.environ.get("M_PERF", "4096"))
    print(f"\nPerf run (no validation): M_PERF={M_perf}")
    data_perf = torch.randint(
        low=0, high=2**31, size=(M_perf, 400), dtype=torch.int32, device=device
    ).contiguous()
    mask_perf = torch.full(
        (M_perf, 400), fill_value=0x7FFFFFFF, dtype=torch.int32, device=device
    ).contiguous()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    # No output matrix, no pair collection for perf timing.
    _D, _pairs, _count = ih.masked_hamming_cuda(
        data_perf,
        mask_perf,
        write_output=False,
        collect_pairs=False,
        threshold=1.0,
        max_pairs=0,
    )
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    pairs_total = 0.5 * M_perf * (M_perf - 1)
    pairs_per_s = pairs_total / (elapsed_ms * 1e-3) / 1e6
    print(f"M={M_perf}, time={elapsed_ms:.3f} ms, pairs/s={pairs_per_s:.3f} M (31 shifts)")


if __name__ == "__main__":
    main()

