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
 * \file glu_tiling_kernel.hpp
 * \brief
 */
#ifndef OPP_GLU_TILING_KERNEL_HPP
#define OPP_GLU_TILING_KERNEL_HPP
#include "glu_tiling.hpp"


constexpr uint32_t DEFAULT_MIN_BLOCK_SIZE = 32; // 最小的数据块长度，32Bytes

// 单输入场景，一个tile需要的偏置参数
struct SwiGluSingleTileOffsetParam {
    uint64_t splitVecGmOffset1 = 0; // 拼接的vector，第一个vector gm上的偏移
    uint64_t splitVecGmOffset2 = 0; // 拼接的vector，第er个vector gm上的偏移
    uint64_t indepVecGmoffset = 0; // 独立的vector gm上的偏移，一般用于 反向的gradout和正向的out
};

struct SwiGluCopyParam {
    uint16_t blockCount = 0; // 指定该指令包含的连续传输数据块个数，取值范围：blockCount∈[1, 4095]。
    uint16_t blockLen = 0; // 指定该指令每个连续传输数据块长度。取值范围：blockLen∈[1, 65535]。32B对齐时，单位为data block(32Bytes); 非32B对齐时，单位为Byte
    uint16_t stride = 0; // 源操作数/目的操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔）。32B对齐时，单位为data block(32Bytes); 非32B对齐时，单位为Byte
};

struct SwiGluSinlgeTileCopyParam {
    SwiGluCopyParam splitVecCopyParam;
    SwiGluCopyParam indepVecCopyParam;
};

// tiling for SwiGlu Vector on one VectorCore
struct SwigluSingleTilingKernel {
    uint32_t is32BAligned = 1; // Is 32-byte aligned for split colLen?
    uint64_t totalBlockLen = 0; // 输入的2个input vector的总长度，单位：元素个数
    uint64_t combColLen = 0; // 输入的2个input vector内存间隔（交织场景 in1 in2算是一行），第1行的首地址和第2行首地址之间的跨度间隔 , Unit:element
    uint64_t colLen = 0; // 输入的2个input vector内存间隔（交织场景 in1 in2算是一个行），in1首地址和in2首地址之间的跨度间隔 , Unit:element
    uint64_t rowLen = 0; // 输入的2个input vector的row Len

    uint32_t baseRowLen = 0; // for one tile in one core, Unit:element
    uint32_t baseColLen = 0; // for one tile in one core, Unit:element
    uint32_t tailRowLen = 0; // number of tail row in one core, Unit:element
    uint32_t tailColLen = 0; // number of column in one core, Unit:element

    uint32_t tileLength = 0;  // baseRowLen * baseColLen

    uint64_t rowTileNum = 0; // row的方向一共分为几片，包含尾块
    uint64_t colTileNum = 0; // col的方向一共分为几片，包含尾块
    uint64_t totalTileNum = 0; // 输入的vector 以baseRowLen*baseColLen分割，总个分为几片

    uint64_t baseRowTileNum = 0; // row的方向一共分为几片，不包含尾块
    uint64_t baseColTileNum = 0; // row的方向一共分为几片，不包含尾块

    // 每一个tile计算的长度，单位:元素个数. 当colLen非32B对齐是，计算的CalLen时colLen需要与32B向上对齐
    uint64_t baseRowBaseColCalLen = 0;
    uint64_t baseRowTailColCalLen = 0;
    uint64_t tailRowBaseColCalLen = 0;
    uint64_t tailRowTailColCalLen = 0;
    SwiGluSinlgeTileCopyParam baseRowBaseColCopyParam;
    SwiGluSinlgeTileCopyParam baseRowTailColCopyParam;
    SwiGluSinlgeTileCopyParam tailRowBaseColCopyParam;
    SwiGluSinlgeTileCopyParam tailRowTailColCopyParam;

    // 每一片的tile参数，临时变量，会不停刷新
    uint64_t curCalLen;
    SwiGluSingleTileOffsetParam offsetParam;
    SwiGluSinlgeTileCopyParam *curTileCopyParam = nullptr;

  // calc tiling data
  __aicore__ void GetTilingAndOffset(GM_ADDR tiling_gm, uint32_t inputDTypeLen) {
    GET_TILING_DATA_WITH_STRUCT(SwiGluTilingData, tempTilingGm, tiling_gm);

    is32BAligned = tempTilingGm.is32BAligned;
    rowLen = tempTilingGm.rowLen;
    colLen = tempTilingGm.colLen;
    // 合并后的combColLen长度为colLen的2倍
    combColLen = colLen * 2;
    totalBlockLen = rowLen * combColLen;

    uint32_t minInputDTypeLen = 2;
    uint32_t isZero = 1;

    baseRowLen = tempTilingGm.baseRowLen;
    baseColLen = tempTilingGm.baseColLen;
    // 申请UB的TQUE和TBUF时需要使用
    tileLength = (is32BAligned == 1) ? (baseRowLen * baseColLen) :
        baseRowLen * AlignUp<uint32_t>(baseColLen,
        ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));

    // 计算分片信息
    baseRowTileNum = rowLen / baseRowLen;
    baseColTileNum = colLen / baseColLen;
    tailRowLen = rowLen % baseRowLen;
    tailColLen = colLen % baseColLen;
    rowTileNum = (tailRowLen > 0) ? (baseRowTileNum + 1) : baseRowTileNum;
    colTileNum = (tailColLen > 0) ? (baseColTileNum + 1) : baseColTileNum;
    totalTileNum = rowTileNum * colTileNum;

    // 计算每个tile的copy参数，分4个区间
    CaclTileCopyParams(inputDTypeLen);
  }

  __aicore__ inline void CaclOneTileCopyParam(uint64_t calRowLen, uint64_t calColLen, uint32_t inputDTypeLen, SwiGluSinlgeTileCopyParam &SwiGluCopyParam)
  {
    // 32B对齐时，单位为data block(32Bytes); 非32B对齐时，单位为Byte
    uint16_t blockUnit = (is32BAligned == 1) ? DEFAULT_MIN_BLOCK_SIZE : 1;
    SwiGluCopyParam.splitVecCopyParam.blockCount = calRowLen;
    SwiGluCopyParam.splitVecCopyParam.blockLen = calColLen * inputDTypeLen / blockUnit;
    SwiGluCopyParam.splitVecCopyParam.stride =
        calRowLen == 1 ? 0 : ((combColLen - calColLen) * inputDTypeLen / blockUnit);

    SwiGluCopyParam.indepVecCopyParam.blockCount = calRowLen;
    SwiGluCopyParam.indepVecCopyParam.blockLen = calColLen * inputDTypeLen / blockUnit;
    SwiGluCopyParam.indepVecCopyParam.stride =
        calRowLen == 1 ? 0 : ((colLen - calColLen) * inputDTypeLen / blockUnit);
  }

  // 计算每片（tile）的copy参数，总共分为4类
  __aicore__ inline void CaclTileCopyParams(uint32_t inputDTypeLen) {
    // 将整个GM区域分为4个区域，分别计算他们的copy参数:
    // zone1:baseRow-baseCol  zone2:baseRow-tailCol  zone3:tailRow-baseCol  zone4:tailRow-tailCol
    // base row , base col
    uint32_t minInputDTypeLen = 2;
    uint32_t isZero = 1;
    baseRowBaseColCalLen = (is32BAligned == 1) ? (baseRowLen * baseColLen) :
      (baseRowLen * AlignUp<uint32_t>(baseColLen,
      ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen)))));
    CaclOneTileCopyParam(baseRowLen, baseColLen, inputDTypeLen, baseRowBaseColCopyParam);

    // base row , tail col
    baseRowTailColCalLen = (is32BAligned == 1) ? (baseRowLen * tailColLen) :
      baseRowLen * AlignUp<uint32_t>(tailColLen,
      ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
    CaclOneTileCopyParam(baseRowLen, tailColLen, inputDTypeLen, baseRowTailColCopyParam);

    // tail row , base col
    tailRowBaseColCalLen = (is32BAligned == 1) ? (tailRowLen * baseColLen) :
      tailRowLen * AlignUp<uint32_t>(baseColLen,
      ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
    CaclOneTileCopyParam(tailRowLen, baseColLen, inputDTypeLen, tailRowBaseColCopyParam);

    // tail row , tail col
    tailRowTailColCalLen = (is32BAligned == 1) ? (tailRowLen * tailColLen) :
      tailRowLen * AlignUp<uint32_t>(tailColLen,
      ISMAX<uint32_t>(isZero, static_cast<uint32_t>(DEFAULT_MIN_BLOCK_SIZE / (inputDTypeLen == 0 ? minInputDTypeLen : inputDTypeLen))));
    CaclOneTileCopyParam(tailRowLen, tailColLen, inputDTypeLen, tailRowTailColCopyParam);
  }

  // 计算每片（tile）的offset参数，每次for循环都会刷新，是临时使用的
  __aicore__ inline void CaclOneTileOffsetParam(uint64_t gmRowOffset, uint64_t colIdx)
  {
    // 加上col的偏置，则得到本loop计算的数据块的offset
    // 第一个input vector的偏移
    offsetParam.splitVecGmOffset1 = gmRowOffset * combColLen + colIdx * baseColLen;
    // 第二个input vector的偏移，col需要整体偏移colLen
    offsetParam.splitVecGmOffset2 = offsetParam.splitVecGmOffset1 + colLen;
    // 独立的vector gm上的偏移，一般用于 反向的gradout和正向的out
    offsetParam.indepVecGmoffset = gmRowOffset * colLen + colIdx * baseColLen;
  }

  __aicore__ inline void CaclOneTileParam(uint64_t tileIdx)
  {
    uint64_t rowTileIdx = tileIdx / colTileNum;
    uint64_t colTileIdx = tileIdx % colTileNum;
    CaclOneTileOffsetParam(rowTileIdx * baseRowLen, colTileIdx);
    if (rowTileIdx < baseRowTileNum) {
      if (colTileIdx < baseColTileNum) {
        // base row, base col
        curCalLen = baseRowBaseColCalLen;
        curTileCopyParam = &baseRowBaseColCopyParam;
      } else {
        // base row, tail col
        curCalLen = baseRowTailColCalLen;
        curTileCopyParam = &baseRowTailColCopyParam;
      }
    } else {
      if (colTileIdx < baseColTileNum) {
        // tail row, base col
        curCalLen = tailRowBaseColCalLen;
        curTileCopyParam = &tailRowBaseColCopyParam;
      } else {
        // tail row, tail col
        curCalLen = tailRowTailColCalLen;
        curTileCopyParam = &tailRowTailColCopyParam;
      }
    }
  }
};
#define SWIGLU_SINGLE_PROCESS_TILE(offsetParam, SwiGluCopyParam, calLen) \
do {                                \
    CopyIn(offsetParam, SwiGluCopyParam); \
    this->Compute(calLen); \
    CopyOut(offsetParam, SwiGluCopyParam); \
} while (0)
#define SWIGLU_SINGLE_PROCESS(kernelTiling) \
do {                                       \
    uint64_t blockNum = 1;       \
	  for(uint64_t tileIdx = 0; tileIdx < (kernelTiling).totalTileNum; tileIdx += blockNum) { \
        (kernelTiling).CaclOneTileParam(tileIdx); \
        SWIGLU_SINGLE_PROCESS_TILE((kernelTiling).offsetParam, *((kernelTiling).curTileCopyParam), (kernelTiling).curCalLen); \
	} \
} while(0)
#define SWIGLU_SINGLE_PROCESS_TILE_NON32BALIGNED(offsetParam, SwiGluCopyParam, calLen) \
do {                                              \
    CopyIn_Non32BAligned(offsetParam, SwiGluCopyParam); \
    this->Compute(calLen); \
    CopyOut_Non32BAligned(offsetParam, SwiGluCopyParam); \
} while(0)
#define SWIGLU_SINGLE_PROCESS_NON32BALIGNED(kernelTiling) \
do {                                       \
    uint64_t blockNum = 1;       \
	  for(uint64_t tileIdx = 0; tileIdx < (kernelTiling).totalTileNum; tileIdx += blockNum) { \
        (kernelTiling).CaclOneTileParam(tileIdx); \
        SWIGLU_SINGLE_PROCESS_TILE_NON32BALIGNED((kernelTiling).offsetParam, *((kernelTiling).curTileCopyParam), (kernelTiling).curCalLen); \
	} \
} while(0)

#endif  // OPP_GLU_TILING_KERNEL_HPP
