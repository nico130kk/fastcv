#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <nvtx3/nvToolsExt.h>
#include "utils.cuh"

__device__ int checkBorder(int pos, int maxBorder) {
    if (pos < 0) return 0;
    if (pos > maxBorder) return maxBorder;
    return pos;
}

template <typename scalar_t>
__global__ void pyrDownKernel(scalar_t* in, scalar_t* out, int width, int height, int channels, int outWidth, int outHeight) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {

        __shared__ int sharedWeights[5];

        if (threadIdx.x < 5 && threadIdx.y == 0) {
            int weights[] = { 1, 4, 6, 4, 1 };
            sharedWeights[threadIdx.x] = weights[threadIdx.x];
        }
        __syncthreads();

        for (int c = 0; c < channels; ++c) {
            float sum = 0;
            float weightSum = 0;

            for (int fy = -2; fy <= 2; ++fy) {
                for (int fx = -2; fx <= 2; ++fx) {

                    int closeX = checkBorder(col * 2 + fx, width - 1);
                    int closeY = checkBorder(row * 2 + fy, height - 1);

                    int idx = (closeY * width + closeX) * channels + c;
                    scalar_t pixelValue = in[idx];

                    int weights = sharedWeights[fx + 2] * sharedWeights[fy + 2];

                    sum += static_cast<float>(pixelValue) * weights;
                    weightSum += static_cast<float>(weights);
                }
            }
            int out_idx = (row * outWidth + col) * channels + c;
            out[out_idx] = static_cast<unsigned char>((sum / weightSum) + 0.5);
        }
    }
}








torch::Tensor pyrDown(torch::Tensor img) {
    nvtxRangePushA("pyrDown_Host_Launch");

    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);
    const int outHeight = (height + 1) / 2;
    const int outWidth = (width + 1) / 2;

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(outWidth, dimBlock.x), cdiv(outHeight, dimBlock.y));

    auto result = torch::empty({ outHeight, outWidth, channels },
        torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    AT_DISPATCH_ALL_TYPES(img.scalar_type(), "pyrDown", ([&] {
        pyrDownKernel<scalar_t> << <dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream() >> > (
            img.data_ptr<scalar_t>(),
            result.data_ptr<scalar_t>(),
            width, height, channels, outWidth, outHeight);
        }));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    nvtxRangePop();

    return result;
}