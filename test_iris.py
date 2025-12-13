import torch
import os
import time
import multiprocessing as mp
import numpy as np

import cuda_iris_matcher as ih


def _perf_worker(
    device_idx: int,
    m_perf: int,
    warmup: int,
    repeats: int,
    start_evt: "mp.synchronize.Event",
    ready_q: "mp.Queue",
    result_q: "mp.Queue",
) -> None:
    """
    Run the performance kernel on a single CUDA device in a subprocess.
    Uses per-device CUDA events for kernel timing; parent aggregates with wall time.
    """
    try:
        if not torch.cuda.is_available():
            result_q.put({"device_idx": device_idx, "error": "CUDA not available"})
            return

        torch.cuda.set_device(device_idx)
        device = torch.device(f"cuda:{device_idx}")

        # This uses packed int32 words directly (shape [M_perf, 400]) to avoid Python/NumPy overhead.
        data_perf = torch.randint(
            low=0, high=2**31, size=(m_perf, 400), dtype=torch.int32, device=device
        ).contiguous()
        mask_perf = torch.full(
            (m_perf, 400), fill_value=0x7FFFFFFF, dtype=torch.int32, device=device
        ).contiguous()

        # Warmup (init kernels, caches, allocator).
        for _ in range(max(0, warmup)):
            ih.masked_hamming_cuda(
                data_perf,
                mask_perf,
                write_output=False,
                collect_pairs=False,
                threshold=1.0,
                max_pairs=0,
            )
        torch.cuda.synchronize(device)

        # Signal readiness after warmup + allocations, then wait for a coordinated start.
        ready_q.put({"device_idx": device_idx, "ready": True})
        start_evt.wait()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(max(1, repeats)):
            ih.masked_hamming_cuda(
                data_perf,
                mask_perf,
                write_output=False,
                collect_pairs=False,
                threshold=1.0,
                max_pairs=0,
            )
        end_event.record()
        torch.cuda.synchronize(device)

        elapsed_ms = start_event.elapsed_time(end_event) / max(1, repeats)
        pairs_total = 0.5 * m_perf * (m_perf - 1)
        pairs_per_s_m = pairs_total / (elapsed_ms * 1e-3) / 1e6

        result_q.put(
            {
                "device_idx": device_idx,
                "device_name": torch.cuda.get_device_name(device_idx),
                "m_perf": m_perf,
                "elapsed_ms": float(elapsed_ms),
                "pairs_total": float(pairs_total),
                "pairs_per_s_m": float(pairs_per_s_m),
            }
        )
    except Exception as e:
        result_q.put({"device_idx": device_idx, "error": f"{type(e).__name__}: {e}"})


def main():
    device = "cuda"
    # Verification size (keep small; numpy is O(M^2 * shifts))
    M = 16

    # Dummy iris codes/masks in the "real" shape: (M, 16, 200, 2, 2) bits
    code = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)
    mask = np.random.randint(0, 2, (M, 16, 200, 2, 2), dtype=np.uint8)

    # --- Timing: Loading onto GPU ---
    t0_load = time.perf_counter()
    code_t = torch.from_numpy(code).to(device=device)
    mask_t_raw = torch.from_numpy(mask).to(device=device)
    torch.cuda.synchronize()
    t1_load = time.perf_counter()
    load_ms = (t1_load - t0_load) * 1e3
    print(f"Loading onto GPU: {load_ms:.3f} ms for M={M}")

    # --- Timing: Packing (CUDA kernel, in-place) ---
    t0_pack = time.perf_counter()
    data_t = ih.pack_theta_major(code_t)
    mask_t = ih.pack_theta_major(mask_t_raw)
    torch.cuda.synchronize()
    t1_pack = time.perf_counter()
    pack_ms = (t1_pack - t0_pack) * 1e3
    print(f"Packing (CUDA): {pack_ms:.3f} ms for M={M}")

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
    print(f"Kernel compute: {elapsed_ms:.3f} ms, pairs/s={pairs_per_s:.3f} M (31 shifts)")

    # --- Timing: Copying back results ---
    # Note: pairs and match_count are already pinned CPU tensors (async copied)
    t0_copy = time.perf_counter()
    D_cpu = D.cpu()
    count = match_count.item() if match_count.numel() else 0
    pairs_cpu = pairs if count > 0 else None
    t1_copy = time.perf_counter()
    copy_ms = (t1_copy - t0_copy) * 1e3
    print(f"Copying back results: {copy_ms:.3f} ms (D only; pairs already on CPU)")

    # Summary
    total_ms = pack_ms + load_ms + elapsed_ms + copy_ms
    print(f"\nTotal pipeline time: {total_ms:.3f} ms")
    print(f"  - Packing:     {pack_ms:7.3f} ms ({100*pack_ms/total_ms:5.1f}%)")
    print(f"  - GPU load:    {load_ms:7.3f} ms ({100*load_ms/total_ms:5.1f}%)")
    print(f"  - Kernel:      {elapsed_ms:7.3f} ms ({100*elapsed_ms/total_ms:5.1f}%)")
    print(f"  - Copy back:   {copy_ms:7.3f} ms ({100*copy_ms/total_ms:5.1f}%)")

    # Fetch count and a few pairs
    print(f"\nMatches found: {count}")
    if count > 0:
        capped = min(count, pairs.size(0))
        print("Sample pairs:", pairs_cpu[: min(8, capped)])

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

    D_cpu_np = D_cpu.numpy()
    max_err = np.max(np.abs(D_cpu_np - ref))
    print(f"NumPy check: max_abs_err={max_err:.3e}")
    if max_err > 1e-5:
        # Print a few mismatches
        mism = np.argwhere(np.abs(D_cpu_np - ref) > 1e-5)
        print("Mismatches (up to 8):")
        for k in range(min(8, mism.shape[0])):
            ii, jj = mism[k]
            print(f"  ({ii},{jj}): gpu={D_cpu_np[ii,jj]:.6f} numpy={ref[ii,jj]:.6f}")

    # ----------------- Performance-only check (no NumPy validation) -----------------
    # Run performance test on all visible CUDA devices concurrently and aggregate throughput.
    M_perf = int(os.environ.get("M_PERF", "4096"))
    warmup = int(os.environ.get("PERF_WARMUP", "0"))
    repeats = int(os.environ.get("PERF_REPEATS", "1"))

    if not torch.cuda.is_available():
        print("\nPerf run skipped: CUDA not available")
        return

    ndev = torch.cuda.device_count()

    # Measure packing overhead at scale
    print(f"\n--- Performance run: M_PERF={M_perf} ---")
    code_perf = np.random.randint(0, 2, (M_perf, 16, 200, 2, 2), dtype=np.uint8)
    mask_perf = np.random.randint(0, 2, (M_perf, 16, 200, 2, 2), dtype=np.uint8)

    # Load onto GPU
    t0_load_perf = time.perf_counter()
    code_cuda = torch.from_numpy(code_perf).cuda()
    mask_cuda = torch.from_numpy(mask_perf).cuda()
    torch.cuda.synchronize()
    t1_load_perf = time.perf_counter()
    load_perf_ms = (t1_load_perf - t0_load_perf) * 1e3
    print(f"Loading onto GPU: {load_perf_ms:.3f} ms for M={M_perf}")

    # CUDA packing (in-place)
    t0_pack_perf = time.perf_counter()
    data_words_cuda = ih.pack_theta_major(code_cuda)
    mask_words_cuda = ih.pack_theta_major(mask_cuda)
    torch.cuda.synchronize()
    t1_pack_perf = time.perf_counter()
    pack_perf_ms = (t1_pack_perf - t0_pack_perf) * 1e3
    print(f"Packing (CUDA): {pack_perf_ms:.3f} ms for M={M_perf}")

    # Clean up to free GPU memory before spawning workers
    del code_perf, mask_perf, code_cuda, mask_cuda, data_words_cuda, mask_words_cuda
    torch.cuda.empty_cache()

    print(
        f"\nKernel benchmark (concurrent): devices={ndev}, "
        f"warmup={warmup}, repeats={repeats}"
    )

    ctx = mp.get_context("spawn")
    start_evt = ctx.Event()
    ready_q: "mp.Queue" = ctx.Queue()
    result_q: "mp.Queue" = ctx.Queue()
    procs = []

    for di in range(ndev):
        p = ctx.Process(
            target=_perf_worker, args=(di, M_perf, warmup, repeats, start_evt, ready_q, result_q)
        )
        p.start()
        procs.append(p)

    # Wait until all workers have allocated/warmed up, then release them together.
    for _ in range(ndev):
        ready_q.get()
    wall_t0 = time.time()
    start_evt.set()

    results = []
    for _ in range(ndev):
        results.append(result_q.get())

    for p in procs:
        p.join()
    wall_t1 = time.time()

    # Print per-device and aggregate.
    results_sorted = sorted(results, key=lambda r: r.get("device_idx", -1))
    ok = [r for r in results_sorted if "error" not in r]
    bad = [r for r in results_sorted if "error" in r]

    for r in ok:
        print(
            f"  cuda:{r['device_idx']} ({r['device_name']}): "
            f"time={r['elapsed_ms']:.3f} ms, pairs/s={r['pairs_per_s_m']:.3f} M"
        )
    for r in bad:
        print(f"  cuda:{r['device_idx']}: ERROR: {r['error']}")

    if ok:
        wall_s = max(1e-9, wall_t1 - wall_t0)
        total_pairs = sum(r["pairs_total"] for r in ok)
        agg_pairs_per_s_m_wall = total_pairs / wall_s / 1e6
        agg_pairs_per_s_m_sum = sum(r["pairs_per_s_m"] for r in ok)
        max_elapsed_s = max(r["elapsed_ms"] for r in ok) * 1e-3
        agg_pairs_per_s_m_concurrent = total_pairs / max(1e-9, max_elapsed_s) / 1e6
        max_kernel_ms = max(r["elapsed_ms"] for r in ok)
        print(
            f"\nAggregate kernel: devices_ok={len(ok)}/{ndev}, wall={wall_s*1e3:.1f} ms, "
            f"pairs/s={agg_pairs_per_s_m_sum:.3f} M (sum of per-device), "
            f"{agg_pairs_per_s_m_concurrent:.3f} M (total/max_device_time), "
            f"{agg_pairs_per_s_m_wall:.3f} M (wall)"
        )

        # Full pipeline summary (packing + load + kernel)
        total_pipeline_ms = pack_perf_ms + load_perf_ms + max_kernel_ms
        print(f"\nFull pipeline breakdown for M={M_perf}:")
        print(f"  - Packing:     {pack_perf_ms:9.3f} ms ({100*pack_perf_ms/total_pipeline_ms:5.1f}%)")
        print(f"  - GPU load:    {load_perf_ms:9.3f} ms ({100*load_perf_ms/total_pipeline_ms:5.1f}%)")
        print(f"  - Kernel:      {max_kernel_ms:9.3f} ms ({100*max_kernel_ms/total_pipeline_ms:5.1f}%)")
        print(f"  - Total:       {total_pipeline_ms:9.3f} ms")


if __name__ == "__main__":
    main()

