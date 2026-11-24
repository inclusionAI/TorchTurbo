#include <ATen/ATen.h>
#include <torch/library.h>
#include <c10/core/DeviceGuard.h>
#include <acl/acl_rt.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

extern "C" void launch_gated(uint32_t blocks, void* stream, void* x, void* dy,
                             void* out, int64_t rows, int64_t width, int dtype,
                             bool backward, bool gelu, float limit, bool after);

namespace {
int CheckInput(const at::Tensor& x)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "Expected NPU input");
    TORCH_CHECK(x.is_contiguous() && x.dim() > 0 && x.size(-1) > 0 && x.size(-1) % 2 == 0,
                "Expected contiguous input with positive even last dimension");
    switch (x.scalar_type()) {
        case at::kFloat: return 0;
        case at::kHalf: return 1;
        case at::kBFloat16: return 2;
        default: TORCH_CHECK(false, "Expected float32, float16 or bfloat16");
    }
}

at::Tensor Run(const at::Tensor& x, const at::Tensor& dy, bool backward,
               bool gelu, double limit, bool after)
{
    const int dtype = CheckInput(x);
    TORCH_CHECK(limit >= 0, "limit must be non-negative");
    const c10::DeviceGuard guard(x.device());
    auto shape = x.sizes().vec();
    shape.back() /= 2;
    if (backward) {
        TORCH_CHECK(dy.device() == x.device() && dy.scalar_type() == x.scalar_type() &&
                    dy.is_contiguous() && dy.sizes().vec() == shape, "Invalid output gradient");
    }
    auto output = at::empty(backward ? x.sizes().vec() : shape, x.options());
    if (x.numel() == 0) return output;
    const int64_t width = shape.back();
    const int64_t rows = x.numel() / (2 * width);
    const int64_t rowsPerTile = width <= 4096 && width % 64 == 0 ? 4096 / width : 1;
    const int64_t tiles = ((rows + rowsPerTile - 1) / rowsPerTile) * ((width + 4095) / 4096);
    int64_t vectorCores = 0;
    const auto status = aclrtGetDeviceInfo(x.get_device(), ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCores);
    TORCH_CHECK(status == ACL_SUCCESS && vectorCores > 0, "Cannot query NPU vector core count: ", status);
    const uint32_t blocks = std::min<int64_t>(vectorCores, tiles);
    // The callback already runs in queue order. Do not drain that same queue
    // from its consumer thread when obtaining the underlying ACL stream.
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    // Queue with torch_npu so launches preserve producer/consumer order.
    // Tensor captures retain allocations until the host launch has executed.
    at_npu::native::OpCommand::RunOpApiV2("TorchTurboGatedActivation", [=]() -> int {
        launch_gated(blocks, stream, x.data_ptr(), backward ? dy.data_ptr() : nullptr,
                     output.data_ptr(), rows, width, dtype, backward, gelu,
                     static_cast<float>(limit), after);
        return 0;
    });
    return output;
}

at::Tensor Forward(const at::Tensor& x, bool gelu, double limit, bool after)
{
    return Run(x, at::Tensor(), false, gelu, limit, after);
}

at::Tensor Backward(const at::Tensor& x, const at::Tensor& dy, bool gelu, double limit, bool after)
{
    return Run(x, dy, true, gelu, limit, after);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(torchturbo_ascendc, m)
{
    m.def("gated_forward(Tensor x, bool gelu, float limit, bool after) -> Tensor");
    m.def("gated_backward(Tensor x, Tensor dy, bool gelu, float limit, bool after) -> Tensor");
}
TORCH_LIBRARY_IMPL(torchturbo_ascendc, PrivateUse1, m)
{
    m.impl("gated_forward", &Forward);
    m.impl("gated_backward", &Backward);
}
