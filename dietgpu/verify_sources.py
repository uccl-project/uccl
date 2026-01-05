#!/usr/bin/env python3
"""
Verify that source file filtering is working correctly for ROCm/CUDA builds.
"""
from glob import glob
import torch

# Determine if we're building for ROCm/HIP or CUDA
is_rocm = not torch.version.cuda

print(f"Platform: {'ROCm' if is_rocm else 'CUDA'}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.version.hip: {torch.version.hip}")
print()

# Select the appropriate main source file
if is_rocm:
    main_source = "./DietGpu_hip.cpp"
else:
    main_source = "./DietGpu.cpp"

# Collect all source files
all_sources = (
    glob("./utils/*.cu") + glob("./utils/*.cpp") + glob("./utils/*.cc") +
    glob("./float/*.cu") + glob("./float/*.cpp") + glob("./float/*.cc") +
    glob("./ans/*.cu") + glob("./ans/*.cpp") + glob("./ans/*.cc")
)

# Filter sources based on platform
if is_rocm:
    # For ROCm: only include files ending with _hip.cpp or _hip.cc, exclude non-hip .cpp/.cc
    filtered_sources = [
        src for src in all_sources
        if not (src.endswith('.cpp') or src.endswith('.cc')) or '_hip.' in src
    ]
else:
    # For CUDA: exclude files with _hip in the name
    filtered_sources = [
        src for src in all_sources
        if '_hip.' not in src
    ]

sources = [main_source] + filtered_sources

print(f"Main source: {main_source}")
print(f"\nTotal sources: {len(sources)}")
print(f"\nUtils sources:")
utils_sources = [s for s in sources if 'utils/' in s]
for src in sorted(utils_sources):
    print(f"  - {src}")

# Verify no conflicts
if is_rocm:
    # Should NOT have DeviceUtils.cpp or StackDeviceMemory.cpp
    conflicts = [s for s in sources if 'DeviceUtils.cpp' in s and '_hip' not in s]
    conflicts += [s for s in sources if 'StackDeviceMemory.cpp' in s and '_hip' not in s]

    if conflicts:
        print(f"\n❌ ERROR: Found conflicting CUDA sources in ROCm build:")
        for c in conflicts:
            print(f"  - {c}")
    else:
        print(f"\n✅ PASS: No conflicting CUDA sources in ROCm build")
else:
    # Should NOT have _hip files
    conflicts = [s for s in sources if '_hip.' in s]

    if conflicts:
        print(f"\n❌ ERROR: Found conflicting HIP sources in CUDA build:")
        for c in conflicts:
            print(f"  - {c}")
    else:
        print(f"\n✅ PASS: No conflicting HIP sources in CUDA build")
