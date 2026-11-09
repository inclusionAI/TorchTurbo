// Fused routing metadata and selective gradient initialization.
#pragma once

namespace routing_kernels {
using namespace AscendC;
constexpr int32_t kZeroBytes = 16384;

template <typename T>
__aicore__ inline void Load(LocalTensor<T> local, GM_ADDR address, int64_t count) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(local, global, copy, pad);
    SetFlag<HardEvent::MTE2_S>(0);
    WaitFlag<HardEvent::MTE2_S>(0);
}

template <typename T>
__aicore__ inline void Store(GM_ADDR address, LocalTensor<T> local, int64_t count) {
    GlobalTensor<T> global;
    global.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
    SetFlag<HardEvent::S_MTE3>(0);
    WaitFlag<HardEvent::S_MTE3>(0);
    DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPad(global, local, copy);
    SetFlag<HardEvent::MTE3_S>(0);
    WaitFlag<HardEvent::MTE3_S>(0);
}

__global__ __aicore__ void Metadata(GM_ADDR input, GM_ADDR offsets, GM_ADDR sizes,
                                  int32_t p, int32_t e, int32_t rank, int32_t h,
                                  int32_t stride, int32_t group, int32_t programs,
                                  int32_t bytes) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, bytes);
    auto memory = buffer.Get<int32_t>();
    auto c = memory;
    auto out = memory[device_aligned(p * e)].ReinterpretCast<int64_t>();
    auto length = memory[device_aligned(p * e) + 2 * stride];
    auto groups = memory[device_aligned(p * e) + 3 * stride].ReinterpretCast<int64_t>();
    Load(c, input, p * e);
    const int32_t local = e / p;
    const int32_t block = GetBlockIdx();
    if (block < programs) {
        for (int32_t source = block * group; source < p && source < (block + 1) * group; ++source) {
            int64_t prefix = 0;
            for (int32_t expert = 0; expert < e; ++expert) {
                int32_t n = c.GetValue(source * e + expert);
                out.SetValue(expert, prefix * h);
                if (source == rank) length.SetValue(expert, n * h);
                prefix += n;
            }
            if (source == rank) {
                Store(offsets, out, e);
                Store(sizes, length, e);
            }
            for (int32_t j = 0; j < local; ++j) groups.SetValue(j, out.GetValue(rank * local + j));
            Store(offsets + (3 * stride + source * local) * 8, groups, local);
        }
    } else {
        int32_t begin = (block - programs) * group;
        for (int32_t dest = begin; dest < p && dest < begin + group; ++dest) {
            int64_t prefix = 0;
            for (int32_t j = 0; j < local; ++j) {
                int64_t source_prefix = 0;
                for (int32_t source = 0; source < p; ++source) {
                    int32_t n = c.GetValue(source * e + dest * local + j);
                    out.SetValue(source * local + j, (prefix + source_prefix) * h);
                    if (dest == rank) length.SetValue(source * local + j, n * h);
                    source_prefix += n;
                }
                prefix += source_prefix;
                groups.SetValue(j, prefix);
            }
            if (dest == rank) {
                Store(offsets + 2 * stride * 8, out, e);
                Store(sizes + stride * 4, length, e);
                Store(offsets + 4 * stride * 8, groups, local);
            }
            for (int32_t j = 0; j < local; ++j) groups.SetValue(j, out.GetValue(rank * local + j));
            Store(offsets + (stride + dest * local) * 8, groups, local);
        }
    }
}

__global__ __aicore__ void Render(GM_ADDR input, GM_ADDR boundaries, GM_ADDR placements,
                                GM_ADDR counts_out, GM_ADDR repeats_out, GM_ADDR labels_out,
                                int32_t p, int32_t e, int32_t rank, int32_t width,
                                int32_t source_group, int32_t bytes) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, bytes);
    auto memory = buffer.Get<int32_t>();
    int32_t pos = 0;
    auto c = memory[pos]; pos += device_aligned(p * e);
    auto cumulative = memory[pos]; pos += device_aligned(p * e);
    auto copies = memory[pos]; pos += device_aligned(e);
    auto lookup = memory[pos]; pos += device_aligned(p * e);
    auto row = memory[pos]; pos += device_aligned(p * width);
    auto repeats = memory[pos]; pos += device_aligned(p * e);
    auto labels = memory[pos];
    Load(c, input, p * e);
    Load(cumulative, boundaries, p * e);
    Load(copies, placements, e);
    const int32_t local = e / p;
    for (int32_t i = 0; i < p * e; ++i) lookup.SetValue(i, -1);
    for (int32_t dest = 0; dest < p; ++dest) {
        for (int32_t j = 0; j < local; ++j) lookup.SetValue(dest * e + dest * local + j, j);
        for (int32_t j = 0; j < width - local; ++j) {
            int32_t expert = copies.GetValue(dest * local + j);
            if (expert >= 0) lookup.SetValue(dest * e + expert, local + j);
        }
    }
    const int32_t begin = GetBlockIdx() * source_group;
    for (int32_t source = begin; source < p && source < begin + source_group; ++source) {
        for (int32_t i = 0; i < p * width; ++i) row.SetValue(i, 0);
        for (int32_t expert = 0; expert < e; ++expert) {
            int32_t start = 0;
            for (int32_t s = 0; s < source; ++s) start += c.GetValue(s * e + expert);
            int32_t end = start + c.GetValue(source * e + expert), before = 0;
            for (int32_t dest = 0; dest < p; ++dest) {
                int32_t after = cumulative.GetValue(expert * p + dest);
                int32_t left = start > before ? start : before;
                int32_t right = end < after ? end : after;
                int32_t n = right > left ? right - left : 0;
                int32_t slot = lookup.GetValue(dest * e + expert);
                if (slot >= 0) row.SetValue(dest * width + slot, n);
                if (source == rank) {
                    repeats.SetValue(expert * p + dest, n);
                    labels.SetValue(expert * p + dest, n ? dest * width + slot : 0);
                }
                before = after;
            }
        }
        Store(counts_out + source * p * width * 4, row, p * width);
        if (source == rank) {
            Store(repeats_out, repeats, p * e);
            Store(labels_out, labels, p * e);
        }
    }
}

__global__ __aicore__ void ZeroEmpty(GM_ADDR first, GM_ADDR second, GM_ADDR group_address,
                                   int64_t first_row, int64_t second_row, int64_t experts) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, kZeroBytes);
    auto zeros = buffer.Get<int32_t>();
    Duplicate(zeros, int32_t(0), kZeroBytes / 4);
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<int64_t> groups;
    groups.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(group_address));
    for (int32_t which = 0; which < 2; ++which) {
        GM_ADDR address = which == 0 ? first : second;
        int64_t row_bytes = which == 0 ? first_row : second_row;
        int64_t total = experts * row_bytes;
        for (int64_t tile = GetBlockIdx() * kZeroBytes; tile < total;
             tile += GetBlockNum() * kZeroBytes) {
            int64_t stop = tile + kZeroBytes < total ? tile + kZeroBytes : total;
            for (int64_t pos = tile; pos < stop;) {
                int64_t expert = pos / row_bytes;
                int64_t end = (expert + 1) * row_bytes;
                if (end > stop) end = stop;
                int64_t before = expert == 0 ? 0 : groups.GetValue(expert - 1);
                if (groups.GetValue(expert) == before) {
                    auto bytes = zeros.ReinterpretCast<uint8_t>();
                    Store(address + pos, bytes, end - pos);
                }
                pos = end;
            }
        }
    }
}

std::vector<torch::Tensor> metadata(torch::Tensor counts, int64_t rank, int64_t hidden) {
    const int64_t p = counts.size(0), e = counts.size(1), local = e / p;
    const int64_t stride = (e + 31) / 32 * 32;
    auto offsets = torch::empty({4 * stride + local}, counts.options().dtype(at::kLong));
    auto sizes = torch::empty({2 * stride}, counts.options());
    const int64_t bytes = (aligned(p * e) + 3 * stride + 2 * aligned(local)) * 4;
    int64_t group = 1;
    while ((group * local) % 16 != 0) ++group;
    const int64_t programs = (p + group - 1) / group;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        Metadata<<<2 * programs, nullptr, stream>>>((GM_ADDR)counts.data_ptr(), (GM_ADDR)offsets.data_ptr(),
            (GM_ADDR)sizes.data_ptr(), p, e, rank, hidden, stride, group, programs, bytes);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MegaMoeMetadata", run);
    return {offsets, sizes};
}

std::vector<torch::Tensor> render(torch::Tensor gathered, torch::Tensor cumulative,
                                torch::Tensor copies, int64_t rank, int64_t width) {
    const int64_t p = gathered.size(0), e = gathered.size(1);
    auto counts = torch::empty({p, p * width}, gathered.options());
    auto repeats = torch::empty({e * p}, gathered.options());
    auto labels = torch::empty({e * p}, gathered.options());
    int64_t group = 1;
    while ((group * p * width) % 32 != 0) ++group;
    const int64_t bytes = (5 * aligned(p * e) + aligned(e) + aligned(p * width)) * 4;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        Render<<<(p + group - 1) / group, nullptr, stream>>>((GM_ADDR)gathered.data_ptr(),
            (GM_ADDR)cumulative.data_ptr(), (GM_ADDR)copies.data_ptr(), (GM_ADDR)counts.data_ptr(),
            (GM_ADDR)repeats.data_ptr(), (GM_ADDR)labels.data_ptr(), p, e, rank, width, group, bytes);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MoonEPVirtualCounts", run);
    return {counts, repeats, labels};
}

void zero_empty(torch::Tensor first, torch::Tensor second, torch::Tensor groups) {
    TORCH_CHECK(first.is_contiguous() && second.is_contiguous() && groups.is_contiguous(),
                "expert gradients and group list must be contiguous");
    TORCH_CHECK(groups.scalar_type() == at::kLong, "group list must be INT64");
    const int64_t experts = groups.numel();
    const int64_t a = first.nbytes() / experts, b = second.nbytes() / experts;
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto run = [=]() -> int {
        ZeroEmpty<<<24, nullptr, stream>>>((GM_ADDR)first.data_ptr(), (GM_ADDR)second.data_ptr(),
            (GM_ADDR)groups.data_ptr(), a, b, experts);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("MegaMoeZeroEmptyGradients", run);
}
}  // namespace routing_kernels
