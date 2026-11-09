/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swi_glu_grad_bf16.hpp
 * \brief
 */
#ifndef OPP_SWI_GLU_GRAD_BF16_HPP
#define OPP_SWI_GLU_GRAD_BF16_HPP
#include "kernel_operator.h"

using namespace AscendC;
template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
class SwiGluGradBF16 {
  public:
    __aicore__ inline SwiGluGradBF16() {}

  protected:
    __aicore__ inline void InitUbBuffer(uint64_t tileLength);
    __aicore__ inline void Compute(uint64_t curTileLen);
    __aicore__ inline void ComputeSigLocal(uint64_t curTileLen);
    __aicore__ inline void ComputeGradN(uint64_t curTileLen);

    calcType beta = 1.0;
    float clampLimit = 0.0;
    bool clampBeforeActivation = false;
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueA;
    TQue<QuePosition::VECIN, bufferNum> inQueueB;
    TQue<QuePosition::VECIN, bufferNum> inQueueL;
    TQue<QuePosition::VECOUT, bufferNum> outQueueM;
    TQue<QuePosition::VECOUT, bufferNum> outQueueN;
    TBuf<TPosition::VECCALC> tmpQueue;
    TBuf<TPosition::VECCALC> sigQueue;
    LocalTensor<calcType> tempLocal;
    LocalTensor<calcType> sigLocal;
    LocalTensor<calcType> aLocal;
    LocalTensor<calcType> nLocal;
    LocalTensor<calcType> lLocal;

    TBuf<TPosition::VECCALC> aTempBuffer;
    TBuf<TPosition::VECCALC> lTempBuffer;
    TBuf<TPosition::VECCALC> outputTempBuffer;
    LocalTensor<inType> aInput;
    LocalTensor<inType> bInput;

    GlobalTensor<inType> aGm;
    GlobalTensor<inType> bGm;
    GlobalTensor<inType> lGm;
    GlobalTensor<outType> mGm;
    GlobalTensor<outType> nGm;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    RoundMode downcastRoundMode_ = RoundMode::CAST_RINT;
#else
    RoundMode downcastRoundMode_ = RoundMode::CAST_NONE;
#endif
};

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::InitUbBuffer(uint64_t tileLength)
{
    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueA, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueB, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(inQueueL, bufferNum, tileLength * sizeof(inType));
    pipe.InitBuffer(outQueueM, bufferNum, tileLength * sizeof(outType)); // The length must be an integer multiple of 32
    pipe.InitBuffer(outQueueN, bufferNum, tileLength * sizeof(outType)); // The length must be an integer multiple of 32

    pipe.InitBuffer(tmpQueue, tileLength * sizeof(calcType));
    pipe.InitBuffer(sigQueue, tileLength * sizeof(calcType));

    pipe.InitBuffer(aTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(lTempBuffer, tileLength * sizeof(calcType));
    pipe.InitBuffer(outputTempBuffer, tileLength * sizeof(calcType));
    tempLocal = tmpQueue.Get<calcType>();
    sigLocal = sigQueue.Get<calcType>();
    aLocal = aTempBuffer.Get<calcType>();
    nLocal = outputTempBuffer.Get<calcType>();
    lLocal = lTempBuffer.Get<calcType>();
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::Compute(uint64_t tileLength)
{
    ComputeSigLocal(tileLength);

    if (clampLimit <= 0.0f) {
        //----------------N
        Mul(nLocal, sigLocal, aLocal, tileLength);
        ComputeGradN(tileLength);

        //----------------M
        Muls(tempLocal, sigLocal, (calcType)(-1.0), tileLength);
        PipeBarrier<PIPE_V>();
        Adds(tempLocal, tempLocal, (calcType)(1.0), tileLength);
        PipeBarrier<PIPE_V>();

        auto& mLocal = nLocal;
        Mul(mLocal, sigLocal, tempLocal, tileLength);
        PipeBarrier<PIPE_V>();
        Mul(mLocal, mLocal, aLocal, tileLength);
        PipeBarrier<PIPE_V>();

        Muls(mLocal, mLocal, -beta, tileLength);
        PipeBarrier<PIPE_V>();
        Add(mLocal, mLocal, sigLocal, tileLength);

        LocalTensor<inType> bLocal_ = inQueueB.template DeQue<inType>();
        auto& bLocal = aLocal;
        Cast(bLocal, bLocal_, RoundMode::CAST_NONE, tileLength);
        PipeBarrier<PIPE_V>();

        Mul(mLocal, mLocal, lLocal, tileLength);
        PipeBarrier<PIPE_V>();

        LocalTensor<outType> mLocal_ = outQueueM.template AllocTensor<outType>();

        Mul(mLocal, mLocal, bLocal, tileLength);
        inQueueB.template FreeTensor(bLocal_);

        Cast(mLocal_, mLocal, downcastRoundMode_, tileLength);
        PipeBarrier<PIPE_V>();
        outQueueM.template EnQue<outType>(mLocal_);
        return;
    }

    // Keep intermediate gradients in FP32, matching the native gated activation.
    LocalTensor<inType> lLocal_ = inQueueL.template DeQue<inType>();
    Cast(lLocal, lLocal_, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
    inQueueL.template FreeTensor(lLocal_);
    bInput = inQueueB.template DeQue<inType>();

    // SiLU derivative at the original or pre-clamped gate.
    Muls(tempLocal, sigLocal, (calcType)(-1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Adds(tempLocal, tempLocal, (calcType)(1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Mul(nLocal, sigLocal, tempLocal, tileLength);
    PipeBarrier<PIPE_V>();
    Mul(nLocal, nLocal, aLocal, tileLength);
    PipeBarrier<PIPE_V>();
    Muls(nLocal, nLocal, -beta, tileLength);
    PipeBarrier<PIPE_V>();
    Add(nLocal, nLocal, sigLocal, tileLength);
    PipeBarrier<PIPE_V>();

    // Multiply backward remains FP32 until the gate gradient is complete.
    Cast(aLocal, bInput, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
    Mins(aLocal, aLocal, clampLimit, tileLength);
    PipeBarrier<PIPE_V>();
    Maxs(aLocal, aLocal, -clampLimit, tileLength);
    PipeBarrier<PIPE_V>();
    Mul(aLocal, aLocal, lLocal, tileLength);
    PipeBarrier<PIPE_V>();
    LocalTensor<outType> mOut = outQueueM.template AllocTensor<outType>();
    Mul(nLocal, nLocal, aLocal, tileLength);
    PipeBarrier<PIPE_V>();

    // Clamp backward includes values exactly on the boundary, as torch.clamp.
    Cast(aLocal, aInput, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
    if (!clampBeforeActivation) {
        Mul(aLocal, aLocal, sigLocal, tileLength);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<uint8_t> mask = tmpQueue.Get<uint8_t>();
    constexpr int64_t kCompareAlign = 64;
    int64_t alignedCount = (tileLength + kCompareAlign - 1) / kCompareAlign * kCompareAlign;
    CompareScalar(mask, aLocal, clampLimit, CMPMODE::LE, alignedCount);
    PipeBarrier<PIPE_V>();
    Select(nLocal, mask, nLocal, (calcType)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLength);
    PipeBarrier<PIPE_V>();
    Cast(mOut, nLocal, downcastRoundMode_, tileLength);
    PipeBarrier<PIPE_V>();
    outQueueM.template EnQue<outType>(mOut);

    // The up-branch gradient uses the FP32 activation and its own clamp mask.
    Cast(aLocal, aInput, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
    if (clampBeforeActivation) {
        Mins(aLocal, aLocal, clampLimit, tileLength);
        PipeBarrier<PIPE_V>();
    }
    Mul(nLocal, aLocal, sigLocal, tileLength);
    PipeBarrier<PIPE_V>();
    LocalTensor<outType> nOut = outQueueN.template AllocTensor<outType>();
    if (!clampBeforeActivation) {
        Mins(nLocal, nLocal, clampLimit, tileLength);
        PipeBarrier<PIPE_V>();
    }
    Mul(nLocal, nLocal, lLocal, tileLength);
    PipeBarrier<PIPE_V>();
    Cast(tempLocal, bInput, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> upMask = sigQueue.Get<uint8_t>();
    CompareScalar(upMask, tempLocal, clampLimit, CMPMODE::LE, alignedCount);
    PipeBarrier<PIPE_V>();
    Select(nLocal, upMask, nLocal, (calcType)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLength);
    PipeBarrier<PIPE_V>();
    CompareScalar(upMask, tempLocal, -clampLimit, CMPMODE::GE, alignedCount);
    PipeBarrier<PIPE_V>();
    Select(nLocal, upMask, nLocal, (calcType)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLength);
    PipeBarrier<PIPE_V>();
    Cast(nOut, nLocal, downcastRoundMode_, tileLength);
    PipeBarrier<PIPE_V>();
    outQueueN.template EnQue<outType>(nOut);
    inQueueA.template FreeTensor(aInput);
    inQueueB.template FreeTensor(bInput);
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::ComputeSigLocal(uint64_t tileLength)
{
    // tbuf::templocaltensor
    // deque input tensors from VECIN queue
    //calc sigLocal
    LocalTensor<inType> aLocal_ = inQueueA.template DeQue<inType>(); //input a
    if (clampLimit > 0.0f) {
        aInput = aLocal_;
    }
    Cast(aLocal, aLocal_, RoundMode::CAST_NONE, tileLength);
    PipeBarrier<PIPE_V>();
    if (clampLimit <= 0.0f) {
        inQueueA.template FreeTensor(aLocal_);
    }

    if (clampBeforeActivation) {
        Mins(aLocal, aLocal, clampLimit, tileLength);
        PipeBarrier<PIPE_V>();
    }
    Muls(sigLocal, aLocal, beta, tileLength);
    PipeBarrier<PIPE_V>();
    Exp(sigLocal, sigLocal, tileLength);
    PipeBarrier<PIPE_V>();
    Adds(sigLocal, sigLocal, (calcType)(1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Duplicate<calcType>(tempLocal, (calcType)(1.0), tileLength);
    PipeBarrier<PIPE_V>();
    Div(sigLocal, tempLocal, sigLocal, tileLength);
    PipeBarrier<PIPE_V>();
}

template<typename inType, typename calcType, typename outType, uint16_t bufferNum>
__aicore__ inline void SwiGluGradBF16<inType, calcType, outType, bufferNum>::ComputeGradN(uint64_t tileLength)
{
    if (clampLimit <= 0.0f) {
        LocalTensor<inType> lLocal_ = inQueueL.template DeQue<inType>(); // input l
        Cast(lLocal, lLocal_, RoundMode::CAST_NONE, tileLength);
        PipeBarrier<PIPE_V>();
        inQueueL.template FreeTensor(lLocal_);
    }

    LocalTensor<outType> nLocal_ = outQueueN.template AllocTensor<outType>(); // lb
    if (clampLimit > 0.0f) {
        Cast(nLocal_, nLocal, downcastRoundMode_, tileLength);
        PipeBarrier<PIPE_V>();
        Cast(nLocal, nLocal_, RoundMode::CAST_NONE, tileLength);
        PipeBarrier<PIPE_V>();
        Mins(nLocal, nLocal, clampLimit, tileLength);
        PipeBarrier<PIPE_V>();
    }

    Mul(nLocal, nLocal, lLocal, tileLength);
    if (clampLimit > 0.0f) {
        LocalTensor<uint8_t> mask = sigQueue.Get<uint8_t>();
        constexpr int64_t kCompareAlign = 64;
        int64_t alignedCount = (tileLength + kCompareAlign - 1) / kCompareAlign * kCompareAlign;
        CompareScalar(mask, tempLocal, clampLimit, CMPMODE::LE, alignedCount);
        PipeBarrier<PIPE_V>();
        Select(nLocal, mask, nLocal, (calcType)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLength);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, tempLocal, -clampLimit, CMPMODE::GE, alignedCount);
        PipeBarrier<PIPE_V>();
        Select(nLocal, mask, nLocal, (calcType)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, tileLength);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();

    Cast(nLocal_, nLocal, downcastRoundMode_, tileLength); // todo nLocal最后使用位置
    PipeBarrier<PIPE_V>();
    outQueueN.template EnQue<outType>(nLocal_);
}
#endif // OPP_SWI_GLU_GRAD_BF16_HPP
