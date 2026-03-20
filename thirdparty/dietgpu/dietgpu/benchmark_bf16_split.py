# Benchmark: split bfloat16 into exponent / sign+mantissa bytes,
# then compress each part with ANS raw byte-wise codec.
#
# bfloat16 layout (16 bits): [sign(1)][exponent(8)][mantissa(7)]
# Split into:
#   exponent byte:       (uint16 >> 7) & 0xFF
#   sign+mantissa byte:  ((uint16 >> 8) & 0x80) | (uint16 & 0x7F)

import torch

import glob as _glob, os as _os
_so_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "build")
_so_files = _glob.glob(_os.path.join(_so_dir, "lib.*", "p2p_dietgpu*.so"))
if not _so_files:
    raise RuntimeError(f"Could not find p2p_dietgpu*.so under {_so_dir}")
torch.ops.load_library(_so_files[0])
dev = torch.device("cuda:0")


def calc_comp_ratio(input_ts, out_sizes):
    total_input_size = 0
    total_comp_size = 0
    for t, s in zip(input_ts, out_sizes):
        total_input_size += t.numel() * t.element_size()
        total_comp_size += s
    return total_input_size, total_comp_size, total_comp_size / total_input_size


def ans_compress_and_report(label, ts, num_runs=3):
    """Compress with raw ANS byte-wise codec and report ratio + throughput."""
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)

    comp_time = 0
    decomp_time = 0

    for i in range(1 + num_runs):
        # --- compress ---
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        rows, cols = torch.ops.dietgpu.max_any_compressed_output_size(ts)
        comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
        sizes = torch.zeros([len(ts)], dtype=torch.int, device=dev)

        start.record()
        comp, sizes, memUsed = torch.ops.dietgpu.compress_data(
            False, ts, False, tempMem, comp, sizes
        )
        end.record()
        torch.cuda.synchronize()
        if i > 0:
            comp_time += start.elapsed_time(end)

        total_size, comp_size, _ = calc_comp_ratio(ts, sizes)

        # --- decompress ---
        out_ts = [torch.empty(t.size(), dtype=t.dtype, device=t.device) for t in ts]
        comp_ts = [*comp]
        out_status = torch.empty([len(ts)], dtype=torch.uint8, device=dev)
        out_sizes = torch.empty([len(ts)], dtype=torch.int32, device=dev)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        torch.ops.dietgpu.decompress_data(
            False, comp_ts, out_ts, False, tempMem, out_status, out_sizes
        )
        end.record()
        torch.cuda.synchronize()
        if i > 0:
            decomp_time += start.elapsed_time(end)

        # validate
        for a, b in zip(ts, out_ts):
            assert torch.equal(a, b), "Decompression mismatch!"

    comp_time /= num_runs
    decomp_time /= num_runs

    ratio = comp_size / total_size
    c_bw = (total_size / 1e9) / (comp_time * 1e-3)
    dc_bw = (total_size / 1e9) / (decomp_time * 1e-3)

    print(f"  {label}:")
    print(f"    orig {total_size} bytes -> comp {comp_size} bytes  (ratio {ratio:.4f}x)")
    print(f"    comp {comp_time:.3f} ms ({c_bw:.1f} GB/s)  decomp {decomp_time:.3f} ms ({dc_bw:.1f} GB/s)")
    return total_size, comp_size, ratio


def split_bf16(tensor):
    """Split bfloat16 tensor into exponent bytes and sign+mantissa bytes on GPU."""
    # Use int32 for bit ops since uint16 doesn't support shifts on CUDA
    raw = tensor.view(torch.uint16).to(torch.int32)
    exponent = ((raw >> 7) & 0xFF).to(torch.uint8)
    sign_mantissa = (((raw >> 8) & 0x80) | (raw & 0x7F)).to(torch.uint8)
    return exponent, sign_mantissa


def merge_bf16(exponent, sign_mantissa):
    """Reconstruct bfloat16 tensor from exponent and sign+mantissa bytes."""
    exp32 = exponent.to(torch.int32)
    sm32 = sign_mantissa.to(torch.int32)
    # sign is bit 7 of sign_mantissa -> bit 15 of uint16
    # exponent is 8 bits -> bits 14-7
    # mantissa is bits 6-0 of sign_mantissa -> bits 6-0
    raw = ((sm32 & 0x80) << 8) | (exp32 << 7) | (sm32 & 0x7F)
    return raw.to(torch.uint16).view(torch.bfloat16)


# ---- main ----
for n_elems in [512 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024, 128 * 512 * 1024]:
    size_bytes = n_elems * 2  # bfloat16 = 2 bytes
    if size_bytes >= 1024 * 1024 * 1024:
        size_str = f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024 * 1024:
        size_str = f"{size_bytes / (1024**2):.0f} MB"
    elif size_bytes >= 1024:
        size_str = f"{size_bytes / 1024:.0f} KB"
    else:
        size_str = f"{size_bytes} B"

    print(f"\n{'='*70}")
    print(f"bfloat16 uniform(-1, 1)  elements={n_elems}  size={size_str}")
    print(f"{'='*70}")

    # Generate random bfloat16 data ~ uniform(-1, 1)
    t = torch.empty(n_elems, dtype=torch.float32, device=dev).uniform_(-1, 1).to(torch.bfloat16)

    # --- Baseline: compress whole bfloat16 as raw bytes ---
    print("\n[Baseline] Whole bfloat16 as raw bytes (ANS byte-wise):")
    t_bytes = t.view(torch.uint8)
    ans_compress_and_report("whole", [t_bytes])

    # --- Baseline: compress whole bfloat16 with float codec ---
    print("\n[Baseline] Whole bfloat16 with float codec:")
    tempMem = torch.empty([384 * 1024 * 1024], dtype=torch.uint8, device=dev)
    rows, cols = torch.ops.dietgpu.max_float_compressed_output_size([t])
    comp = torch.empty([rows, cols], dtype=torch.uint8, device=dev)
    sizes = torch.zeros([1], dtype=torch.int, device=dev)
    comp, sizes, _ = torch.ops.dietgpu.compress_data(True, [t], False, tempMem, comp, sizes)
    total_size = t.numel() * t.element_size()
    comp_size = sizes[0].item()
    ratio = comp_size / total_size
    print(f"  float codec: orig {total_size} bytes -> comp {comp_size} bytes  (ratio {ratio:.4f}x)")

    # --- Split: exponent vs sign+mantissa ---
    exponent, sign_mantissa = split_bf16(t)

    # Verify reconstruction
    reconstructed = merge_bf16(exponent, sign_mantissa)
    assert torch.equal(t, reconstructed), "Reconstruction failed!"

    print("\n[Split] Exponent bytes (ANS byte-wise):")
    _, exp_comp, exp_ratio = ans_compress_and_report("exponent", [exponent])

    print("\n[Split] Sign+Mantissa bytes (ANS byte-wise):")
    _, sm_comp, sm_ratio = ans_compress_and_report("sign+mantissa", [sign_mantissa])

    # Combined split ratio
    combined_comp = exp_comp + sm_comp
    combined_ratio = combined_comp / total_size
    print(f"\n[Split Combined] orig {total_size} -> exp {exp_comp} + sm {sm_comp} = {combined_comp} bytes  (ratio {combined_ratio:.4f}x)")

    # --- Also try: high byte / low byte split (for comparison) ---
    raw_u8 = t.view(torch.uint8)
    high_bytes = raw_u8[0::2].contiguous()  # byte containing sign + upper exponent
    low_bytes = raw_u8[1::2].contiguous()   # byte containing lower exponent + mantissa

    print("\n[Alt Split] High byte vs Low byte (byte-interleave):")
    _, hi_comp, _ = ans_compress_and_report("high byte", [high_bytes])
    _, lo_comp, _ = ans_compress_and_report("low byte", [low_bytes])
    alt_combined = hi_comp + lo_comp
    alt_ratio = alt_combined / total_size
    print(f"[Alt Combined] orig {total_size} -> hi {hi_comp} + lo {lo_comp} = {alt_combined} bytes  (ratio {alt_ratio:.4f}x)")
