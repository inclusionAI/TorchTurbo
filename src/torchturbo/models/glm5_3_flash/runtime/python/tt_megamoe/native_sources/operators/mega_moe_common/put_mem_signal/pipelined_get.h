/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "shmem.h"
namespace multicore {
// One worker owns both UB halves until its final MTE3 write has completed.
template <typename T>
__aicore__ inline void get_pipelined(GM_ADDR destination, GM_ADDR source, int64_t elements, int source_pe) {
  if (elements == 0) {
    return;
  }
  auto *state = aclshmemi_get_state();
  if (!(state->topo_list[source_pe] & ACLSHMEM_TRANSPORT_MTE)) {
    AscendC::Trap();
  }
  auto *remote = reinterpret_cast<__gm__ T *>(aclshmem_ptr(source, source_pe));
  if (remote == nullptr) {
    AscendC::Trap();
  }
  auto *local = reinterpret_cast<__gm__ T *>(destination);
  uint64_t ub = state->mte_config.aclshmem_ub;
  uint64_t block_bytes = state->mte_config.ub_size / 2;
  // Larger remote-read bursts lengthen concurrent GMM execution.
  constexpr uint64_t max_block_bytes = 4 * 1024;
  if (block_bytes > max_block_bytes) {
    block_bytes = max_block_bytes;
  }
  uint64_t total_bytes = static_cast<uint64_t>(elements) * sizeof(T);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
  for (uint64_t offset = 0, block = 0; offset < total_bytes; offset += block_bytes, ++block) {
    AscendC::TEventID event = (block & 1) ? EVENT_ID1 : EVENT_ID0;
    auto *buffer = reinterpret_cast<__ubuf__ T *>(ub + (block & 1) * block_bytes);
    uint64_t bytes = total_bytes - offset < block_bytes ? total_bytes - offset : block_bytes;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    aclshmemi_copy_gm2ub(buffer, remote + offset / sizeof(T), bytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
    aclshmemi_copy_ub2gm(local + offset / sizeof(T), buffer, bytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
  }
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

}
