#include "gated_activation.hpp"

using namespace AscendC;
constexpr uint32_t TILE = 4096;

template <typename T>
class GatedKernel {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR dy, GM_ADDR out, int64_t rows, int64_t width,
                               bool backward, bool gelu, float limit, bool after)
    {
        GlobalTensor<T> input, grad, output;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        grad.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dy));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        pipe.InitBuffer(floats, 9 * TILE * sizeof(float));
        pipe.InitBuffer(inputQueue, 1, TILE * sizeof(T));
        pipe.InitBuffer(outputQueue, 1, TILE * sizeof(T));
        pipe.InitBuffer(bits, TILE / 4);
        auto base = floats.Get<float>();
        auto g = base;
        auto u = base[TILE];
        auto cg = base[2 * TILE];
        auto cu = base[3 * TILE];
        auto a = base[4 * TILE];
        auto derivative = base[5 * TILE];
        auto t = base[6 * TILE];
        auto s = base[7 * TILE];
        auto dout = base[8 * TILE];
        auto result = t;
        auto mask = bits.Get<uint8_t>();
        auto gateMask = mask[TILE / 8];
        const int64_t colTiles = (width + TILE - 1) / TILE;
        // Aligned rows pack into one vector tile with strided GM copies.
        // Unaligned widths retain the padded single-row path.
        const int64_t rowsPerTile = width <= TILE && width % 64 == 0 ? TILE / width : 1;
        const int64_t rowTiles = (rows + rowsPerTile - 1) / rowsPerTile;
        for (int64_t tile = GetBlockIdx(); tile < rowTiles * colTiles; tile += GetBlockNum()) {
            const int64_t row = tile / colTiles * rowsPerTile;
            const int64_t col = (tile % colTiles) * TILE;
            const uint16_t numRows = static_cast<uint16_t>(rows - row < rowsPerTile ? rows - row : rowsPerTile);
            const uint32_t rowCount = static_cast<uint32_t>(width - col < TILE ? width - col : TILE);
            const uint32_t validCount = rowCount * numRows;
            // Compare/Select consume packed masks in full 64-float repeats.
            // Keep arithmetic padded while global copies use the exact length.
            const uint32_t count = (validCount + 63) / 64 * 64;
            const int64_t offset = row * 2 * width + col;
            const int64_t yOffset = row * width + col;
            Load(g, input[offset], rowCount, numRows, width);
            Load(u, input[offset + width], rowCount, numRows, width);
            if (limit > 0) {
                Mins(cg, g, limit, count);
                Mins(cu, u, limit, count);
                PipeBarrier<PIPE_V>();
                Maxs(cu, cu, -limit, count);
                PipeBarrier<PIPE_V>();
            }
            TorchTurbo::Activation(a, derivative, limit > 0 && !after ? cg : g, t, s, gelu, count, backward);
            if (limit > 0) {
                // clamp passes gradients at both endpoints, including equality.
                if (backward) CompareScalar(gateMask, after ? a : g, limit, CMPMODE::LE, count);
                if (after) Mins(a, a, limit, count);
                PipeBarrier<PIPE_V>();
            } else {
                Round(a, count);
            }
            if (!backward) {
                Mul(result, a, limit > 0 ? cu : u, count);
                PipeBarrier<PIPE_V>();
                Store(output[yOffset], result, validCount);
                continue;
            }
            Load(dout, grad[yOffset], validCount);
            Mul(result, dout, limit > 0 ? cu : u, count);
            PipeBarrier<PIPE_V>();
            if (!(limit > 0)) Round(result, count);
            // Apply clamp backward in graph order, rather than masking the
            // derivative: zero times a NaN upstream gradient is still NaN.
            if (limit > 0 && after) {
                Select(result, gateMask, result, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
                PipeBarrier<PIPE_V>();
            }
            Mul(result, result, derivative, count);
            PipeBarrier<PIPE_V>();
            if (limit > 0 && !after) {
                Select(result, gateMask, result, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
                PipeBarrier<PIPE_V>();
            }
            Store(output[offset], result, rowCount, numRows, width);
            Mul(result, dout, a, count);
            PipeBarrier<PIPE_V>();
            if (limit > 0) {
                CompareScalar(mask, u, limit, CMPMODE::LE, count);
                PipeBarrier<PIPE_V>();
                Select(result, mask, result, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
                PipeBarrier<PIPE_V>();
                CompareScalar(mask, u, -limit, CMPMODE::GE, count);
                PipeBarrier<PIPE_V>();
                Select(result, mask, result, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
                PipeBarrier<PIPE_V>();
            }
            Store(output[offset + width], result, rowCount, numRows, width);
        }
    }

private:
    __aicore__ inline void Load(const LocalTensor<float>& dst, const GlobalTensor<T>& src,
                                uint32_t rowCount, uint16_t numRows = 1, uint32_t stride = 0)
    {
        const uint32_t count = rowCount * numRows;
        auto buf = inputQueue.AllocTensor<T>();
        DataCopyExtParams params{numRows, static_cast<uint32_t>(rowCount * sizeof(T)),
                                 static_cast<uint32_t>(stride * sizeof(T)), 0, 0};
        DataCopyPad(buf, src, params, DataCopyPadExtParams<T>{false, 0, 0, 0});
        inputQueue.EnQue(buf);
        buf = inputQueue.DeQue<T>();
        if constexpr (std::is_same_v<T, float>) Adds(dst, buf, 0.0f, count);
        else Cast(dst, buf, RoundMode::CAST_NONE, count);
        inputQueue.FreeTensor(buf);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Round(const LocalTensor<float>& value, uint32_t count)
    {
        if constexpr (!std::is_same_v<T, float>) {
            // The unclamped path does not use the clipped-gate scratch.
            auto buf = floats.Get<float>()[2 * TILE].template ReinterpretCast<T>();
            Cast(buf, value, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Cast(value, buf, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Store(const GlobalTensor<T>& dst, const LocalTensor<float>& src,
                                 uint32_t rowCount, uint16_t numRows = 1, uint32_t stride = 0)
    {
        const uint32_t count = rowCount * numRows;
        auto buf = outputQueue.AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) Adds(buf, src, 0.0f, count);
        else Cast(buf, src, RoundMode::CAST_RINT, count);
        outputQueue.EnQue(buf);
        buf = outputQueue.DeQue<T>();
        DataCopyExtParams params{numRows, static_cast<uint32_t>(rowCount * sizeof(T)),
                                 0, static_cast<uint32_t>(stride * sizeof(T)), 0};
        DataCopyPad(dst, buf, params);
        outputQueue.FreeTensor(buf);
    }

    TPipe pipe;
    TBuf<TPosition::VECCALC> floats, bits;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
};

__global__ __aicore__ void gated_fp32(GM_ADDR x, GM_ADDR dy, GM_ADDR out,
                                          int64_t rows, int64_t width, bool backward,
                                          bool gelu, float limit, bool after)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GatedKernel<float> kernel;
    kernel.Run(x, dy, out, rows, width, backward, gelu, limit, after);
}

__global__ __aicore__ void gated_fp16(GM_ADDR x, GM_ADDR dy, GM_ADDR out,
                                    int64_t rows, int64_t width, bool backward,
                                    bool gelu, float limit, bool after)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GatedKernel<half> kernel;
    kernel.Run(x, dy, out, rows, width, backward, gelu, limit, after);
}

__global__ __aicore__ void gated_bf16(GM_ADDR x, GM_ADDR dy, GM_ADDR out,
                                    int64_t rows, int64_t width, bool backward,
                                    bool gelu, float limit, bool after)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GatedKernel<bfloat16_t> kernel;
    kernel.Run(x, dy, out, rows, width, backward, gelu, limit, after);
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void launch_gated(uint32_t blocks, void* stream, void* x, void* dy, void* out,
                             int64_t rows, int64_t width, int dtype, bool backward,
                             bool gelu, float limit, bool after)
{
    if (dtype == 0) {
        gated_fp32<<<blocks, nullptr, stream>>>(
            (GM_ADDR)x, (GM_ADDR)dy, (GM_ADDR)out, rows, width, backward, gelu, limit, after);
    } else if (dtype == 1) {
        gated_fp16<<<blocks, nullptr, stream>>>(
            (GM_ADDR)x, (GM_ADDR)dy, (GM_ADDR)out, rows, width, backward, gelu, limit, after);
    } else {
        gated_bf16<<<blocks, nullptr, stream>>>(
            (GM_ADDR)x, (GM_ADDR)dy, (GM_ADDR)out, rows, width, backward, gelu, limit, after);
    }
}
#endif
