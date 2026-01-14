/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <glog/logging.h>
#include <torch/types.h>
#include <vector>
#include "dietgpu/ans/GpuANSCodec.h"
#include "dietgpu/float/GpuFloatCodec.h"
#include "dietgpu/utils/StackDeviceMemory.h"

namespace dietgpu {

namespace {

FloatType getFloatTypeFromDtype(at::ScalarType t) {
  switch (t) {
    case at::ScalarType::Half:
      return FloatType::kFloat16;
    case at::ScalarType::BFloat16:
      return FloatType::kBFloat16;
    case at::ScalarType::Float:
      return FloatType::kFloat32;
    default:
      TORCH_CHECK(
          t == at::ScalarType::Half || t == at::ScalarType::BFloat16 ||
          t == at::ScalarType::Float);
      return FloatType::kUndefined;
  }
}

at::ScalarType getDtypeFromFloatType(FloatType ft) {
  switch (ft) {
    case FloatType::kFloat16:
      return at::ScalarType::Half;
    case FloatType::kBFloat16:
      return at::ScalarType::BFloat16;
    case FloatType::kFloat32:
      return at::ScalarType::Float;
    default:
      TORCH_CHECK(
          ft == FloatType::kFloat16 || ft == FloatType::kBFloat16 ||
          ft == FloatType::kFloat32);
      return at::ScalarType::Half;
  }
}

FloatType getFloatTypeFromTensor(const torch::Tensor& t) {
  return getFloatTypeFromDtype(t.dtype().toScalarType());
}

// returns (totalSize, maxSize)
std::tuple<int64_t, int64_t> getTotalAndMaxSize(
    const std::vector<torch::Tensor>& tIns) {
  int64_t totalSize = 0;
  int64_t maxSize = 0;

  for (auto& t : tIns) {
    auto curSize = t.numel();
    // FIXME: due to int indexing, it's really total size
    TORCH_CHECK(
        curSize * t.element_size() <= std::numeric_limits<uint32_t>::max());

    totalSize += curSize;
    maxSize = std::max(maxSize, curSize);
  }

  TORCH_CHECK(maxSize <= std::numeric_limits<uint32_t>::max());

  return std::make_tuple(totalSize, maxSize);
}

// Convert a compressed matrix into a list of tensors that are views into the
// compressed row pieces
std::vector<torch::Tensor> compressedMatrixToTensors(
    int numInBatch,
    torch::Tensor& matrix_dev,
    torch::Tensor& sizes_dev) {
  auto stream = at::cuda::getCurrentCUDAStream();

  // We wish to return narrowed tensors with a view into the matrix
  auto sizes_host = std::vector<uint32_t>(numInBatch);

  CUDA_VERIFY(cudaMemcpyAsync(
      sizes_host.data(),
      sizes_dev.data_ptr(),
      sizeof(uint32_t) * numInBatch,
      cudaMemcpyDeviceToHost,
      stream));

  auto out = std::vector<torch::Tensor>(numInBatch);

  auto matrix1d = matrix_dev.view({matrix_dev.numel()});
  auto cols = matrix_dev.size(1);

  for (int i = 0; i < numInBatch; ++i) {
    out[i] = matrix1d.narrow(0, i * cols, sizes_host[i]);
  }

  return out;
}

} // namespace

//
// External API
//

constexpr int kDefaultPrecision = 10;

std::tuple<int64_t, int64_t> max_float_compressed_output_size(
    const std::vector<torch::Tensor>& ts) {
  auto sizes = getTotalAndMaxSize(ts);

  auto maxCompSize = getMaxFloatCompressedSize(
      getFloatTypeFromTensor(ts[0]), std::get<1>(sizes));

  return std::make_tuple(ts.size(), maxCompSize);
}

// FIXME: can we pass a dtype somehow instead?
int64_t max_float_compressed_size(const torch::Tensor& dtype, int64_t size) {
  return getMaxFloatCompressedSize(getFloatTypeFromTensor(dtype), size);
}

std::tuple<int64_t, int64_t> max_any_compressed_output_size(
    const std::vector<torch::Tensor>& ts) {
  auto sizes = getTotalAndMaxSize(ts);
  int64_t maxBytes = std::get<1>(sizes) * ts[0].element_size();

  return std::make_tuple(ts.size(), getMaxCompressedSize(maxBytes));
}

int64_t max_any_compressed_size(int64_t bytes) {
  return getMaxCompressedSize(bytes);
}

//////////////////////
//
// Compress
//
//////////////////////

std::tuple<torch::Tensor, torch::Tensor, int64_t> compress_data_res(
    bool compressAsFloat,
    StackDeviceMemory& res,
    const std::vector<torch::Tensor>& tIns,
    bool checksum,
    const std::optional<torch::Tensor>& outCompressed,
    const std::optional<torch::Tensor>& outCompressedSizes) {
  TORCH_CHECK(!tIns.empty());

  // All computation will take place on this device
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  auto maxOutputSize = compressAsFloat ? max_float_compressed_output_size(tIns)
                                       : max_any_compressed_output_size(tIns);

  //
  // Validate input and validate / construct output
  //
  for (auto& t : tIns) {
    TORCH_CHECK(t.device().type() == at::kCUDA);
    TORCH_CHECK(t.is_contiguous());

    // device must be consistent
    TORCH_CHECK(t.get_device() == dev);

    // must be all the same type unless we are compressing bytewise
    if (compressAsFloat) {
      TORCH_CHECK(t.dtype() == tIns[0].dtype());

      // must be a supported float type
      TORCH_CHECK(
          getFloatTypeFromDtype(t.dtype().toScalarType()) !=
          FloatType::kUndefined);
    }
  }

  torch::Tensor comp;
  if (outCompressed) {
    TORCH_CHECK(outCompressed->dtype() == torch::kByte);
    TORCH_CHECK(outCompressed->device().type() == at::kCUDA);
    TORCH_CHECK(outCompressed->is_contiguous());
    TORCH_CHECK(outCompressed->dim() == 2);
    TORCH_CHECK(outCompressed->size(0) >= tIns.size());
    TORCH_CHECK(outCompressed->size(1) >= std::get<1>(maxOutputSize));
    TORCH_CHECK(outCompressed->get_device() == dev);

    comp = *outCompressed;
  } else {
    comp = torch::empty(
        {(int64_t)tIns.size(), std::get<1>(maxOutputSize)},
        at::TensorOptions()
            .device(tIns[0].device())
            .dtype(at::ScalarType::Byte));
  }

  auto inPtrs = std::vector<const void*>(tIns.size());
  auto inSize = std::vector<uint32_t>(tIns.size());
  auto compPtrs = std::vector<void*>(tIns.size());

  for (size_t i = 0; i < tIns.size(); ++i) {
    auto& t = tIns[i];

    inPtrs[i] = t.data_ptr();
    inSize[i] = compressAsFloat ? t.numel() : (t.numel() * t.element_size());
    compPtrs[i] = (uint8_t*)comp.data_ptr() + i * comp.size(1);
  }

  //
  // Validate / construct output sizes
  //
  torch::Tensor sizes;
  if (outCompressedSizes) {
    TORCH_CHECK(outCompressedSizes->dtype() == torch::kInt);
    TORCH_CHECK(outCompressedSizes->device().type() == at::kCUDA);
    TORCH_CHECK(outCompressedSizes->dim() == 1);
    TORCH_CHECK(outCompressedSizes->is_contiguous());
    TORCH_CHECK(outCompressedSizes->size(0) >= tIns.size());
    TORCH_CHECK(outCompressedSizes->get_device() == dev);

    sizes = *outCompressedSizes;
  } else {
    // FIXME: no uint32 in torch
    sizes = torch::empty(
        {(int64_t)tIns.size()},
        at::TensorOptions()
            .device(tIns[0].device())
            .dtype(at::ScalarType::Int));
  }

  if (compressAsFloat) {
    auto config = FloatCompressConfig(
        getFloatTypeFromTensor(tIns[0]),
        ANSCodecConfig(kDefaultPrecision, false),
        false /* we'll figure this out later */,
        checksum);

    floatCompress(
        res,
        config,
        tIns.size(),
        inPtrs.data(),
        inSize.data(),
        compPtrs.data(),
        // FIXME: int32_t versus uint32_t
        (uint32_t*)sizes.data_ptr(),
        at::cuda::getCurrentCUDAStream());
  } else {
    auto config = ANSCodecConfig(kDefaultPrecision, checksum);

    ansEncodeBatchPointer(
        res,
        config,
        tIns.size(),
        inPtrs.data(),
        inSize.data(),
        nullptr,
        compPtrs.data(),
        // FIXME: int32_t versus uint32_t
        (uint32_t*)sizes.data_ptr(),
        at::cuda::getCurrentCUDAStream());
  }

  // how much temporary memory we actually used
  int64_t tempMemUsage = res.getMaxMemoryUsage();
  return std::make_tuple(std::move(comp), std::move(sizes), tempMemUsage);
}

std::tuple<torch::Tensor, torch::Tensor, int64_t> compress_data(
    bool compressAsFloat,
    const std::vector<torch::Tensor>& tIns,
    bool checksum,
    const std::optional<torch::Tensor>& tempMem,
    const std::optional<torch::Tensor>& outCompressed,
    const std::optional<torch::Tensor>& outCompressedSizes) {
  TORCH_CHECK(!tIns.empty());

  // All computation will take place on this device; set before creating the
  // GpuResources object
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  // Validate temp memory if passed
  if (tempMem) {
    TORCH_CHECK(tempMem->device().type() == at::kCUDA);
    TORCH_CHECK(tempMem->is_contiguous());

    // Should be on the same device as the first tensor passed
    TORCH_CHECK(tempMem->get_device() == tIns.front().get_device());
  }

  auto res = StackDeviceMemory(
      getCurrentDevice(),
      tempMem ? tempMem->data_ptr() : nullptr,
      tempMem ? tempMem->numel() * tempMem->element_size() : 0);

  // The rest of the validation takes place here
  return compress_data_res(
      compressAsFloat, res, tIns, checksum, outCompressed, outCompressedSizes);
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, int64_t>
compress_data_split_size(
    bool compressAsFloat,
    const torch::Tensor& tIn,
    const torch::Tensor& tSplitSizes,
    bool checksum,
    const std::optional<torch::Tensor>& tempMem,
    const std::optional<torch::Tensor>& outCompressed,
    const std::optional<torch::Tensor>& outCompressedSizes) {
  // All computation will take place on this device; set before creating the
  // GpuResources object
  int dev = tIn.get_device();
  DeviceScope device(dev);
  auto stream = at::cuda::getCurrentCUDAStream();

  // Validate temp memory if passed
  if (tempMem) {
    TORCH_CHECK(tempMem->device().type() == at::kCUDA);
    TORCH_CHECK(tempMem->is_contiguous());

    // Should be on the same device as the first tensor passed
    TORCH_CHECK(tempMem->get_device() == dev);
  }

  // Validate input
  auto floatType = compressAsFloat
      ? getFloatTypeFromDtype(tIn.dtype().toScalarType())
      : FloatType::kUndefined;

  TORCH_CHECK(tIn.device().type() == at::kCUDA);
  TORCH_CHECK(tIn.is_contiguous());
  TORCH_CHECK(tIn.get_device() == dev);
  if (compressAsFloat) {
    TORCH_CHECK(floatType != FloatType::kUndefined);
  } else {
    // All input data must meet ANS alignment
    TORCH_CHECK(
        uintptr_t(tIn.data_ptr()) % kANSRequiredAlignment == 0,
        "All splits should start on a 16 byte boundary; "
        "start pointer is not aligned");
  }

  // Validate split sizes
  auto numInBatch = tSplitSizes.numel();
  TORCH_CHECK(tSplitSizes.is_contiguous());
  TORCH_CHECK(tSplitSizes.device().type() == at::kCPU);
  TORCH_CHECK(tSplitSizes.dtype() == torch::kInt);

  uint32_t maxSize = 0;
  for (size_t i = 0; i < numInBatch; ++i) {
    auto size = ((const int32_t*)tSplitSizes.data_ptr())[i];
    TORCH_CHECK(size > 0);
    maxSize = std::max((uint32_t)size, maxSize);

    if (!compressAsFloat && i != (numInBatch - 1)) {
      // All input data starts for direct ANS compression must meet ANS
      // alignment
      TORCH_CHECK(
          size % kANSRequiredAlignment == 0,
          "All splits should start on a 16 byte boundary; the size of an interior "
          "split is not a multiple of 16 bytes");
    }
  }

  auto maxCompressedBytes = compressAsFloat
      ? getMaxFloatCompressedSize(floatType, maxSize)
      : getMaxCompressedSize(maxSize);

  // Validate output
  torch::Tensor comp;
  if (outCompressed) {
    TORCH_CHECK(outCompressed->dtype() == torch::kByte);
    TORCH_CHECK(outCompressed->device().type() == at::kCUDA);
    TORCH_CHECK(outCompressed->is_contiguous());
    TORCH_CHECK(outCompressed->dim() == 2);
    TORCH_CHECK(outCompressed->size(0) >= numInBatch);
    TORCH_CHECK(outCompressed->size(1) >= maxCompressedBytes);
    TORCH_CHECK(outCompressed->get_device() == dev);

    comp = *outCompressed;
  } else {
    comp = torch::empty(
        {(int64_t)numInBatch, maxCompressedBytes},
        at::TensorOptions().device(tIn.device()).dtype(at::ScalarType::Byte));
  }

  torch::Tensor sizes;
  if (outCompressedSizes) {
    TORCH_CHECK(outCompressedSizes->dtype() == torch::kInt);
    TORCH_CHECK(outCompressedSizes->device().type() == at::kCUDA);
    TORCH_CHECK(outCompressedSizes->dim() == 1);
    TORCH_CHECK(outCompressedSizes->is_contiguous());
    TORCH_CHECK(outCompressedSizes->size(0) >= numInBatch);
    TORCH_CHECK(outCompressedSizes->get_device() == dev);

    sizes = *outCompressedSizes;
  } else {
    // FIXME: no uint32 in torch
    sizes = torch::empty(
        {(int64_t)numInBatch},
        at::TensorOptions().device(tIn.device()).dtype(at::ScalarType::Int));
  }

  auto res = StackDeviceMemory(
      getCurrentDevice(),
      tempMem ? tempMem->data_ptr() : nullptr,
      tempMem ? tempMem->numel() * tempMem->element_size() : 0);

  if (compressAsFloat) {
    auto config = FloatCompressConfig(
        floatType,
        ANSCodecConfig(kDefaultPrecision, false),
        false /* we'll figure this out later */,
        checksum);

    floatCompressSplitSize(
        res,
        config,
        numInBatch,
        tIn.data_ptr(),
        // FIXME: int32_t versus uint32_t
        (const uint32_t*)tSplitSizes.data_ptr(),
        comp.data_ptr(),
        maxCompressedBytes,
        // FIXME: int32_t versus uint32_t
        (uint32_t*)sizes.data_ptr(),
        stream);
  } else {
    auto config = ANSCodecConfig(kDefaultPrecision, checksum);

    ansEncodeBatchSplitSize(
        res,
        config,
        numInBatch,
        tIn.data_ptr(),
        // FIXME: int32_t versus uint32_t
        (const uint32_t*)tSplitSizes.data_ptr(),
        nullptr,
        comp.data_ptr(),
        maxCompressedBytes,
        // FIXME: int32_t versus uint32_t
        (uint32_t*)sizes.data_ptr(),
        stream);
  }

  auto compList = compressedMatrixToTensors(numInBatch, comp, sizes);

  // how much temporary memory we actually used
  int64_t tempMemUsage = res.getMaxMemoryUsage();
  return std::make_tuple(std::move(compList), std::move(sizes), tempMemUsage);
}

std::vector<torch::Tensor> compress_data_simple(
    bool compressAsFloat,
    const std::vector<torch::Tensor>& tIns,
    bool checksum,
    const std::optional<int64_t>& tempMem) {
  TORCH_CHECK(!tIns.empty());

  std::tuple<torch::Tensor, torch::Tensor, int64_t> comp;

  if (tempMem && *tempMem > 0) {
    torch::Tensor scratch = torch::empty(
        {*tempMem},
        at::TensorOptions()
            .device(tIns[0].device())
            .dtype(at::ScalarType::Byte));

    // rest of validation takes place here
    comp = compress_data(
        compressAsFloat, tIns, checksum, scratch, std::nullopt, std::nullopt);
  } else {
    // rest of validation takes place here
    comp = compress_data(
        compressAsFloat,
        tIns,
        checksum,
        std::nullopt,
        std::nullopt,
        std::nullopt);
  }

  auto& compMatrix_dev = std::get<0>(comp);
  auto& size_dev = std::get<1>(comp);

  torch::Tensor size_host = size_dev.to(torch::kCPU);
  TORCH_CHECK(size_host.size(0) == tIns.size());

  auto compMatrixRowStride = compMatrix_dev.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto out = std::vector<torch::Tensor>();
  for (int i = 0; i < tIns.size(); ++i) {
    auto compSize = ((int32_t*)size_host.data_ptr())[i];

    out.emplace_back(torch::empty(
        {compSize},
        at::TensorOptions()
            .device(tIns[0].device())
            .dtype(at::ScalarType::Byte)));

    // FIXME: custom batch kernel to avoid N cudaMemcpy calls?
    CUDA_VERIFY(cudaMemcpyAsync(
        out[i].data_ptr(),
        (uint8_t*)compMatrix_dev.data_ptr() + i * compMatrixRowStride,
        compSize,
        cudaMemcpyDeviceToDevice,
        stream));
  }

  return out;
}

//////////////////////
//
// Decompress
//
//////////////////////

int64_t decompress_data_res(
    bool compressAsFloat,
    StackDeviceMemory& res,
    const std::vector<torch::Tensor>& tIns,
    const std::vector<torch::Tensor>& tOuts,
    bool checksum,
    const std::optional<torch::Tensor>& outStatus,
    const std::optional<torch::Tensor>& outSizes) {
  TORCH_CHECK(!tIns.empty());
  TORCH_CHECK(tIns.size() == tOuts.size());

  // All computation will take place on this device
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  // Validate input and output
  auto inPtrs = std::vector<const void*>(tIns.size());
  auto outPtrs = std::vector<void*>(tIns.size());
  auto outCapacity = std::vector<uint32_t>(tOuts.size());

  for (size_t i = 0; i < tIns.size(); ++i) {
    auto& tIn = tIns[i];
    auto& tOut = tOuts[i];

    TORCH_CHECK(tIn.device().type() == at::kCUDA);
    TORCH_CHECK(tIn.get_device() == dev);
    TORCH_CHECK(tIn.is_contiguous());

    TORCH_CHECK(tOut.device().type() == at::kCUDA);
    TORCH_CHECK(tOut.get_device() == dev);
    TORCH_CHECK(tOut.is_contiguous());

    TORCH_CHECK(tIn.dtype() == torch::kByte);
    if (compressAsFloat) {
      TORCH_CHECK(
          tOut.dtype() == torch::kFloat16 || tOut.dtype() == torch::kBFloat16 ||
          tOut.dtype() == torch::kFloat32);
    }

    inPtrs[i] = tIn.data_ptr();
    outPtrs[i] = tOut.data_ptr();

    auto outSize =
        compressAsFloat ? tOut.numel() : (tOut.numel() * tOut.element_size());

    // FIXME: total range checking
    TORCH_CHECK(outSize <= std::numeric_limits<uint32_t>::max());
    outCapacity[i] = outSize;
  }

  // Validate outStatus, if passed
  if (outStatus) {
    TORCH_CHECK(outStatus->is_contiguous());
    TORCH_CHECK(outStatus->device().type() == at::kCUDA);
    TORCH_CHECK(outStatus->dtype() == torch::kByte);
    TORCH_CHECK(outStatus->numel() == tIns.size());
    TORCH_CHECK(outStatus->get_device() == dev);
  }

  // Validate outSizes, if passed
  if (outSizes) {
    TORCH_CHECK(outSizes->is_contiguous());
    TORCH_CHECK(outSizes->device().type() == at::kCUDA);
    TORCH_CHECK(outSizes->dtype() == torch::kInt32);
    TORCH_CHECK(outSizes->numel() == tIns.size());
    TORCH_CHECK(outSizes->get_device() == dev);
  }

  if (compressAsFloat) {
    auto config = FloatDecompressConfig(
        getFloatTypeFromTensor(tOuts[0]),
        ANSCodecConfig(kDefaultPrecision, false),
        false /* we'll figure this out later */,
        checksum);

    auto decStatus = floatDecompress(
        res,
        config,
        tIns.size(),
        inPtrs.data(),
        outPtrs.data(),
        outCapacity.data(),
        outStatus ? (uint8_t*)outStatus->data_ptr() : nullptr,
        // FIXME: int32_t versus uint32_t
        outSizes ? (uint32_t*)outSizes->data_ptr() : nullptr,
        at::cuda::getCurrentCUDAStream());

    TORCH_CHECK(
        decStatus.error != FloatDecompressError::ChecksumMismatch,
        "floatDecompress: checksum mismatch seen on decoded data; "
        "archive cannot be unpacked");
  } else {
    auto config = ANSCodecConfig(kDefaultPrecision, checksum);

    auto decStatus = ansDecodeBatchPointer(
        res,
        config,
        tIns.size(),
        inPtrs.data(),
        outPtrs.data(),
        outCapacity.data(),
        outStatus ? (uint8_t*)outStatus->data_ptr() : nullptr,
        // FIXME: int32_t versus uint32_t
        outSizes ? (uint32_t*)outSizes->data_ptr() : nullptr,
        at::cuda::getCurrentCUDAStream());

    TORCH_CHECK(
        decStatus.error != ANSDecodeError::ChecksumMismatch,
        "ANSDecode: checksum mismatch seen on decoded data; "
        "archive cannot be unpacked");
  }

  // how much temporary memory we actually used
  return res.getMaxMemoryUsage();
}

int64_t decompress_data(
    bool compressAsFloat,
    const std::vector<torch::Tensor>& tIns,
    const std::vector<torch::Tensor>& tOuts,
    bool checksum,
    const std::optional<torch::Tensor>& tempMem,
    const std::optional<torch::Tensor>& outStatus,
    const std::optional<torch::Tensor>& outSizes) {
  TORCH_CHECK(!tIns.empty());

  // All computation will take place on this device; set before creating the
  // GpuResources object
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  // Validate temp memory if passed
  if (tempMem) {
    TORCH_CHECK(tempMem->device().type() == at::kCUDA);
    TORCH_CHECK(tempMem->is_contiguous());
    TORCH_CHECK(tempMem->get_device() == tIns.front().get_device());
    // we don't care about data type, we just care about memory
  }

  auto res = StackDeviceMemory(
      getCurrentDevice(),
      tempMem ? tempMem->data_ptr() : nullptr,
      tempMem ? tempMem->numel() * tempMem->element_size() : 0);

  // Rest of validation happens here
  return decompress_data_res(
      compressAsFloat, res, tIns, tOuts, checksum, outStatus, outSizes);
}

int64_t decompress_data_split_size(
    bool compressAsFloat,
    const std::vector<torch::Tensor>& tIns,
    torch::Tensor& tOut,
    const torch::Tensor& tSplitSizes,
    bool checksum,
    const std::optional<torch::Tensor>& tempMem,
    const std::optional<torch::Tensor>& outStatus,
    const std::optional<torch::Tensor>& outSizes) {
  TORCH_CHECK(!tIns.empty());

  // All computation will take place on this device; set before creating the
  // GpuResources object
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  auto numInBatch = tSplitSizes.numel();

  // Validate temp memory if passed
  if (tempMem) {
    TORCH_CHECK(tempMem->device().type() == at::kCUDA);
    TORCH_CHECK(tempMem->is_contiguous());
    TORCH_CHECK(tempMem->get_device() == tIns.front().get_device());
    // we don't care about data type, we just care about memory
  }

  // Validate input, split sizes and output
  auto inPtrs = std::vector<void*>(tIns.size());
  auto splitSizes = std::vector<uint32_t>(tIns.size());

  // Validate split sizes
  TORCH_CHECK(tSplitSizes.is_contiguous());
  TORCH_CHECK(tSplitSizes.device().type() == at::kCPU);
  TORCH_CHECK(tSplitSizes.dtype() == torch::kInt);
  // Should be a size for each of the input tensors
  TORCH_CHECK(numInBatch == tIns.size());

  for (size_t i = 0; i < numInBatch; ++i) {
    auto& tIn = tIns[i];

    TORCH_CHECK(tIn.device().type() == at::kCUDA);
    TORCH_CHECK(tIn.get_device() == dev);
    TORCH_CHECK(tIn.is_contiguous());

    TORCH_CHECK(tIn.dtype() == torch::kByte);

    inPtrs[i] = tIn.data_ptr();

    auto size = ((const int32_t*)tSplitSizes.data_ptr())[i];
    TORCH_CHECK(size > 0);
    splitSizes[i] = size;
  }

  // Validate output
  TORCH_CHECK(tOut.device().type() == at::kCUDA);
  TORCH_CHECK(tOut.get_device() == dev);
  TORCH_CHECK(tOut.is_contiguous());
  if (compressAsFloat) {
    TORCH_CHECK(
        tOut.dtype() == torch::kFloat16 || tOut.dtype() == torch::kBFloat16 ||
        tOut.dtype() == torch::kFloat32);
  }

  auto outSize =
      compressAsFloat ? tOut.numel() : (tOut.numel() * tOut.element_size());

  // FIXME: total range checking
  TORCH_CHECK(outSize <= std::numeric_limits<uint32_t>::max());

  // Validate outStatus, if passed
  if (outStatus) {
    TORCH_CHECK(outStatus->is_contiguous());
    TORCH_CHECK(outStatus->device().type() == at::kCUDA);
    TORCH_CHECK(outStatus->dtype() == torch::kByte);
    TORCH_CHECK(outStatus->numel() == numInBatch);
    TORCH_CHECK(outStatus->get_device() == dev);
  }

  // Validate outSizes, if passed
  if (outSizes) {
    TORCH_CHECK(outSizes->is_contiguous());
    TORCH_CHECK(outSizes->device().type() == at::kCUDA);
    TORCH_CHECK(outSizes->dtype() == torch::kInt32);
    TORCH_CHECK(outSizes->numel() == numInBatch);
    TORCH_CHECK(outSizes->get_device() == dev);
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  auto res = StackDeviceMemory(
      getCurrentDevice(),
      tempMem ? tempMem->data_ptr() : nullptr,
      tempMem ? tempMem->numel() * tempMem->element_size() : 0);

  if (compressAsFloat) {
    auto config = FloatDecompressConfig(
        getFloatTypeFromTensor(tOut),
        ANSCodecConfig(kDefaultPrecision, false),
        false /* we figure this out later */,
        checksum);

    auto decStatus = floatDecompressSplitSize(
        res,
        config,
        numInBatch,
        (const void**)inPtrs.data(),
        tOut.data_ptr(),
        splitSizes.data(),
        (uint8_t*)(outStatus ? outStatus->data_ptr() : nullptr),
        // FIXME: int32_t vs uint32_t
        (uint32_t*)(outSizes ? outSizes->data_ptr() : nullptr),
        stream);

    TORCH_CHECK(
        decStatus.error != FloatDecompressError::ChecksumMismatch,
        "floatDecompress: checksum mismatch seen on decoded data; "
        "archive cannot be unpacked");
  } else {
    auto config = ANSCodecConfig(kDefaultPrecision, checksum);

    auto decStatus = ansDecodeBatchSplitSize(
        res,
        config,
        numInBatch,
        (const void**)inPtrs.data(),
        tOut.data_ptr(),
        splitSizes.data(),
        (uint8_t*)(outStatus ? outStatus->data_ptr() : nullptr),
        // FIXME: int32_t vs uint32_t
        (uint32_t*)(outSizes ? outSizes->data_ptr() : nullptr),
        stream);

    TORCH_CHECK(
        decStatus.error != ANSDecodeError::ChecksumMismatch,
        "ANSDecode: checksum mismatch seen on decoded data; "
        "archive cannot be unpacked");
  }

  // how much temporary memory we actually used
  return res.getMaxMemoryUsage();
}

std::vector<torch::Tensor> decompress_data_simple(
    bool compressAsFloat,
    const std::vector<torch::Tensor>& tIns,
    bool checksum,
    const std::optional<int64_t>& tempMem) {
  TORCH_CHECK(!tIns.empty());
  auto stream = at::cuda::getCurrentCUDAStream();

  // All computation will take place on this device
  int dev = tIns.front().get_device();
  DeviceScope device(dev);

  size_t tempMemToUse = 0;
  if (tempMem && *tempMem >= kSDMAlignment) {
    tempMemToUse = *tempMem;
  }

  torch::Tensor scratch;
  if (tempMemToUse) {
    scratch = torch::empty(
        {(int64_t)tempMemToUse},
        at::TensorOptions()
            .device(tIns[0].device())
            .dtype(at::ScalarType::Byte));
  }

  auto res = StackDeviceMemory(
      getCurrentDevice(),
      tempMemToUse ? scratch.data_ptr() : nullptr,
      tempMemToUse);

  auto sizes_dev = res.alloc<uint32_t>(stream, tIns.size());
  auto types_dev = res.alloc<uint32_t>(stream, tIns.size());

  auto inPtrs = std::vector<const void*>(tIns.size());
  for (int i = 0; i < tIns.size(); ++i) {
    auto& tIn = tIns[i];

    inPtrs[i] = tIn.data_ptr();

    TORCH_CHECK(tIn.device().type() == at::kCUDA);
    TORCH_CHECK(tIn.get_device() == dev);
    TORCH_CHECK(tIn.is_contiguous());
  }

  if (compressAsFloat) {
    floatGetCompressedInfo(
        res,
        inPtrs.data(),
        tIns.size(),
        sizes_dev.data(),
        types_dev.data(),
        nullptr,
        stream);
  } else {
    ansGetCompressedInfo(
        res, inPtrs.data(), tIns.size(), sizes_dev.data(), nullptr, stream);
  }

  auto sizes = sizes_dev.copyToHost(stream);
  auto types = types_dev.copyToHost(stream);

  auto tOuts = std::vector<torch::Tensor>();
  for (int i = 0; i < tIns.size(); ++i) {
    auto size = sizes[i];
    auto type = types[i];

    torch::Tensor tOut;

    if (compressAsFloat) {
      TORCH_CHECK(type == types[0]); // must be consistent dtype

      tOut = torch::empty(
          {static_cast<int>(size)},
          at::TensorOptions()
              .device(tIns[0].device())
              .dtype(getDtypeFromFloatType((FloatType)type)));
    } else {
      tOut = torch::empty(
          {static_cast<int>(size)},
          at::TensorOptions().device(tIns[0].device()).dtype(torch::kByte));
    }

    tOuts.emplace_back(std::move(tOut));
  }

  decompress_data_res(
      compressAsFloat, res, tIns, tOuts, checksum, std::nullopt, std::nullopt);

  return tOuts;
}

} // namespace dietgpu

TORCH_LIBRARY_FRAGMENT(dietgpu, m) {
  // compression sizes
  m.def("max_float_compressed_output_size(Tensor[] ts) -> (int, int)");
  m.def("max_float_compressed_size(Tensor dtype, int size) -> int");
  m.def("max_any_compressed_output_size(Tensor[] ts) -> (int, int)");
  m.def("max_any_compressed_size(int bytes) -> int");

  // data compress
  m.def(
      "compress_data(bool compress_as_float, Tensor[] ts_in, bool checksum=False, Tensor? temp_mem=None, Tensor? out_compressed=None, Tensor? out_compressed_bytes=None) -> (Tensor, Tensor, int)");
  m.def(
      "compress_data_split_size(bool compress_as_float, Tensor t_in, Tensor t_in_split_sizes, bool checksum=False, Tensor? temp_mem=None, Tensor? out_compressed=None, Tensor? out_compressed_bytes=None) -> (Tensor[], Tensor, int)");
  m.def(
      "compress_data_simple(bool compress_as_float, Tensor[] ts_in, bool checksum=False, int? temp_mem=67108864) -> Tensor[]");

  // data decompress
  m.def(
      "decompress_data(bool compress_as_float, Tensor[] ts_in, Tensor[] ts_out, bool checksum=False, Tensor? temp_mem=None, Tensor? out_status=None, Tensor? out_decompressed_words=None) -> (int)");
  m.def(
      "decompress_data_split_size(bool compress_as_float, Tensor[] ts_in, Tensor t_out, Tensor t_out_split_sizes, bool checksum=False, Tensor? temp_mem=None, Tensor? out_status=None, Tensor? out_decompressed_words=None) -> (int)");
  m.def(
      "decompress_data_simple(bool compress_as_float, Tensor[] ts_in, bool checksum=False, int? temp_mem=67108864) -> Tensor[]");
}

TORCH_LIBRARY(dietgpu, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::max_float_compressed_output_size"),
      TORCH_FN(dietgpu::max_float_compressed_output_size));
  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::max_float_compressed_size"),
      TORCH_FN(dietgpu::max_float_compressed_size));
  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::max_any_compressed_output_size"),
      TORCH_FN(dietgpu::max_any_compressed_output_size));
  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::max_any_compressed_size"),
      TORCH_FN(dietgpu::max_any_compressed_size));

  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::compress_data"),
      TORCH_FN(dietgpu::compress_data));
  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::compress_data_split_size"),
      TORCH_FN(dietgpu::compress_data_split_size));
  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::compress_data_simple"),
      TORCH_FN(dietgpu::compress_data_simple));

  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::decompress_data"),
      TORCH_FN(dietgpu::decompress_data));
  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::decompress_data_split_size"),
      TORCH_FN(dietgpu::decompress_data_split_size));
  m.impl(
      TORCH_SELECTIVE_NAME("dietgpu::decompress_data_simple"),
      TORCH_FN(dietgpu::decompress_data_simple));
}
