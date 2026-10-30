// Bind expert transfers to the same SHMEM library used by MegaMoE.
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "shmem.h"
#include "ascend_moonep/kernels/launch.h"

extern "C" int moonep_shared_grad_return_bf16(
    void* stream, const void* source, void* output, void* slots,
    const int32_t* plan, int32_t experts, int32_t hidden, int32_t projection,
    int32_t ranks, int32_t rank, int32_t blocks);

static void check_runtime(int64_t rank, int64_t ranks) {
    TORCH_CHECK(rank == aclshmem_my_pe(), "expert-transfer rank differs from active MegaMoE SHMEM PE");
    TORCH_CHECK(ranks == aclshmem_n_pes(), "expert-transfer world differs from active MegaMoE SHMEM root");
}

static void prefetch(torch::Tensor home, torch::Tensor slots, torch::Tensor copies,
                     int64_t rank, int64_t ranks) {
    check_runtime(rank, ranks);
    TORCH_CHECK(home.is_contiguous() && slots.is_contiguous() && copies.is_contiguous(), "expert transfer requires contiguous tensors");
    TORCH_CHECK(home.scalar_type() == at::kBFloat16 && slots.scalar_type() == at::kBFloat16, "prefetch weights must be BF16");
    TORCH_CHECK(copies.scalar_type() == at::kInt && copies.dim() == 1, "prefetch copies must be INT32 slots");
    TORCH_CHECK(home.dim() == 3 && slots.dim() == 3 && home.size(0) == 2*slots.size(0), "prefetch home/slot geometry changed");
    TORCH_CHECK(home.size(1) == slots.size(1) && home.size(2) == slots.size(2) && copies.numel() == slots.size(0), "prefetch projection shape changed");
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // OpCommand may execute after this binding returns; own the tensor handles.
    auto run = [=]() -> int {
        // The device kernel publishes home writes and completes all peer reads.
        prefetch_weight_kernel_do(stream, static_cast<uint8_t*>(home.data_ptr()),
            static_cast<uint8_t*>(slots.data_ptr()), static_cast<uint8_t*>(copies.data_ptr()),
            ranks*slots.size(0), home.size(1), home.size(2), slots.size(0), 24);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MoonEPSharedPrefetch", run);
}

static void reduce(torch::Tensor source, torch::Tensor home, torch::Tensor slots,
                   torch::Tensor copies, int64_t rank) {
    check_runtime(rank, copies.size(0));
    TORCH_CHECK(source.is_contiguous() && home.is_contiguous() && slots.is_contiguous() && copies.is_contiguous(), "gradient transfer requires contiguous tensors");
    TORCH_CHECK(source.scalar_type() == at::kBFloat16 && home.scalar_type() == at::kBFloat16 && slots.scalar_type() == at::kBFloat16, "expert gradients must be BF16; accumulation uses FP32 inside the kernel");
    TORCH_CHECK(copies.scalar_type() == at::kInt && copies.dim() == 2, "gradient copy plan must be INT32 [root, slots]");
    TORCH_CHECK(home.sizes() == source.sizes() && home.sizes() == slots.sizes() && home.dim() == 3 && home.size(0) == copies.size(1), "gradient transfer shape changed");
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        return moonep_shared_grad_return_bf16(stream, source.data_ptr(), home.data_ptr(), slots.data_ptr(),
            copies.data_ptr<int32_t>(), home.size(0), home.size(1), home.size(2),
            copies.size(0), rank, 24);
    };
    at_npu::native::OpCommand::RunOpApi("MoonEPSharedGradReduce", run);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("prefetch", &prefetch);
    module.def("reduce", &reduce);
    module.def("runtime_state", []() { return std::vector<int>{aclshmem_my_pe(), aclshmem_n_pes()}; });
}
