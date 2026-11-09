/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_UDMA_H
#define SHMEM_DEVICE_UDMA_H

#include <stdint.h>

#include "kernel_operator.h"
#include "host_device/shmem_common_types.h"

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 * PE to address on the local device.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations
 * to the same PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE to the SQ ring. PIPE_MTE3 (default)
 *                            stages the WQE in the caller-provided UB scratch (see @p buf)
 *                            and DataCopyPads it to HBM in one shot, useful for hot loops.
 *                            PIPE_S scalar-writes the SQE/SGE block directly to HBM and
 *                            ignores @p buf / @p sync_id. Other pipe values are not supported.
 *
 * @param dst                 [in] Pointer on local device of the destination data.
 * @param src                 [in] Pointer on Symmetric memory of the source data.
 * @param buf                 [in] Pointer on local UB. Used as WQE staging scratch when
 *                                 WQE_PIPE == PIPE_MTE3 (must hold one full WQE block;
 *                                 128 B is safe for all current opcodes); ignored when
 *                                 WQE_PIPE == PIPE_S.
 * @param elem_size           [in] Number of elements in the destination and source arrays.
 * @param pe                  [in] PE number of the remote PE.
 * @param sync_id             [in] Hardware event ID used by the MTE3->S sync after
 *                                 DataCopyPad in the PIPE_MTE3 path. Ignored when
 *                                 WQE_PIPE == PIPE_S. Defaults to 0 for backward
 *                                 compatibility with existing callers.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id = 0);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 * PE to address on the local PE.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations
 * to the same PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE to the SQ ring. See the bare-pointer
 *                            overload for semantics.
 *
 * @param dst                 [in] GlobalTensor on local device of the destination data.
 * @param src                 [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf                 [in] LocalTensor on local UB. WQE staging scratch when
 *                                 WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size           [in] Number of elements in the destination and source arrays.
 * @param pe                  [in] PE number of the remote PE.
 * @param sync_id             [in] Hardware event ID used by the MTE3->S sync after
 *                                 DataCopyPad in the PIPE_MTE3 path. Ignored when
 *                                 WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id = 0);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE to the SQ ring. See the get_nbi
 *                            overload for semantics.
 *
 * @note A single UDMA put request transfers at most 256 MB
 *       (256 * 1024 * 1024 bytes). @p elem_size is the number of T
 *       elements, so the transfer size is @p elem_size * sizeof(T) bytes.
 *       Split larger transfers into multiple put requests whose byte size
 *       does not exceed 256 MB. After submitting one or more asynchronous
 *       requests, call the matching completion/synchronization interface,
 *       such as @ref aclshmemx_udma_quiet, before reading the destination
 *       data or reusing buffers whose contents must remain stable.
 *
 * @param dst                 [in] Pointer on Symmetric memory of the destination data.
 * @param src                 [in] Pointer on local device of the source data.
 * @param buf                 [in] Pointer on local UB. WQE staging scratch when
 *                                 WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size           [in] Number of elements in the destination and source arrays.
 * @param pe                  [in] PE number of the remote PE.
 * @param sync_id             [in] Hardware event ID used by the MTE3->S sync after
 *                                 DataCopyPad in the PIPE_MTE3 path. Ignored when
 *                                 WQE_PIPE == PIPE_S. Defaults to 0 for backward
 *                                 compatibility with existing callers.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, uint32_t sync_id = 0);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE to the SQ ring. See the get_nbi
 *                            overload for semantics.
 *
 * @note A single UDMA put request transfers at most 256 MB
 *       (256 * 1024 * 1024 bytes). @p elem_size is the number of T
 *       elements, so the transfer size is @p elem_size * sizeof(T) bytes.
 *       Split larger transfers into multiple put requests whose byte size
 *       does not exceed 256 MB. After submitting one or more asynchronous
 *       requests, call the matching completion/synchronization interface,
 *       such as @ref aclshmemx_udma_quiet, before reading the destination
 *       data or reusing buffers whose contents must remain stable.
 *
 * @param dst                 [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src                 [in] GlobalTensor on local device of the source data.
 * @param buf                 [in] LocalTensor on local UB. WQE staging scratch when
 *                                 WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size           [in] Number of elements in the destination and source arrays.
 * @param pe                  [in] PE number of the remote PE.
 * @param sync_id             [in] Hardware event ID used by the MTE3->S sync after
 *                                 DataCopyPad in the PIPE_MTE3 path. Ignored when
 *                                 WQE_PIPE == PIPE_S.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous relay-mode UDMA Put. Copy contiguous data from local PE to symmetric address on
 *        pe, while egressing on the local source EID that reaches relay_pe. The fabric forwards
 *        the packet via relay_pe to pe. Use this to spread traffic across multiple physical
 *        links between the same pair of nodes for higher aggregate bandwidth.
 *        Requires building with ACLSHMEM_RELAY_SUPPORT=ON.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        pe are not supported.
 *
 * @note  Preconditions on the PE arguments (all must hold, where myPe is the calling PE and
 *        rankCount is the total number of PEs):
 *          - 0 <= pe < rankCount and 0 <= relay_pe < rankCount
 *          - pe != relay_pe
 *          - pe != myPe and relay_pe != myPe (self is not a valid actual/relay target)
 *        If any precondition is violated the call submits nothing and returns immediately; in a
 *        debug build it aborts the kernel instead. Because the return type is void, the caller
 *        cannot detect a skipped submission at runtime -- validate the arguments before calling.
 * @note  Completion semantics: this is a non-blocking (_nbi) call. A normal return only means the
 *        WQE was published to the send queue; it does NOT mean the transfer finished or that the
 *        data is visible on pe. Call aclshmemx_udma_quiet(pe) (or a higher-level barrier) before
 *        reading the result or reusing @p src to guarantee completion and remote visibility.
 * @note  A single UDMA put request transfers at most 256 MB (256 * 1024 * 1024 bytes). @p elem_size
 *        is the number of T elements, so the transfer size is @p elem_size * sizeof(T) bytes. Split
 *        larger transfers into multiple put requests whose byte size does not exceed 256 MB.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE to the SQ ring. See @ref aclshmemx_udma_put_nbi
 *                            for semantics. PIPE_MTE3 (default) stages the WQE in the caller-provided
 *                            UB scratch (see @p buf) and DataCopyPads it to HBM in one shot; PIPE_S
 *                            scalar-writes the SQE/SGE block directly to HBM and ignores @p buf /
 *                            @p sync_id. Other pipe values are not supported.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data (on pe).
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB. Used as WQE staging scratch when
 *                               WQE_PIPE == PIPE_MTE3 (must hold one full WQE block; 256 B is safe
 *                               for all current opcodes); ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the actual destination. Must satisfy the preconditions above.
 * @param relay_pe          [in] PE whose port path is used to forward the packet. Must satisfy the
 *                               preconditions above.
 * @param sync_id           [in] Hardware event ID used by the MTE3->S sync after DataCopyPad in the
 *                               PIPE_MTE3 path. Ignored when WQE_PIPE == PIPE_S. Defaults to 0 for
 *                               backward compatibility with existing callers.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id = 0);

/**
 * @brief GlobalTensor/LocalTensor overload of @ref aclshmemx_udma_relay_put_nbi.
 *
 * @note  Same PE-argument preconditions and non-blocking completion semantics as the bare-pointer
 *        overload: invalid (pe, relay_pe) combinations submit nothing (abort in debug) and cannot be
 *        detected via the void return; a normal return only means the WQE was published, so call
 *        aclshmemx_udma_quiet(pe) before consuming the result or reusing @p src.
 * @note  A single UDMA put request transfers at most 256 MB (256 * 1024 * 1024 bytes); split larger
 *        transfers into multiple put requests whose byte size (@p elem_size * sizeof(T)) does not
 *        exceed 256 MB.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE. See the bare-pointer overload for semantics.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data (on pe).
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB. WQE staging scratch when
 *                               WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the actual destination. Must satisfy the preconditions above.
 * @param relay_pe          [in] PE whose port path is used to forward the packet. Must satisfy the
 *                               preconditions above.
 * @param sync_id           [in] Hardware event ID used by the MTE3->S sync in the PIPE_MTE3 path.
 *                               Ignored when WQE_PIPE == PIPE_S. Defaults to 0.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id = 0);

/**
 * @brief Asynchronous relay-mode UDMA Get. Symmetric to @ref aclshmemx_udma_relay_put_nbi.
 *        Requires building with ACLSHMEM_RELAY_SUPPORT=ON.
 *
 * @note  Preconditions on the PE arguments (all must hold, where myPe is the calling PE and
 *        rankCount is the total number of PEs):
 *          - 0 <= pe < rankCount and 0 <= relay_pe < rankCount
 *          - pe != relay_pe
 *          - pe != myPe and relay_pe != myPe (self is not a valid actual/relay target)
 *        If any precondition is violated the call submits nothing and returns immediately; in a
 *        debug build it aborts the kernel instead. Because the return type is void, the caller
 *        cannot detect a skipped submission at runtime -- validate the arguments before calling.
 * @note  Completion semantics: this is a non-blocking (_nbi) call. A normal return only means the
 *        WQE was published to the send queue; it does NOT mean the transfer finished or that @p dst
 *        holds the fetched data. Call aclshmemx_udma_quiet(pe) (or a higher-level barrier) before
 *        reading @p dst to guarantee completion.
 * @note  A single UDMA get request transfers at most 256 MB (256 * 1024 * 1024 bytes). @p elem_size
 *        is the number of T elements, so the transfer size is @p elem_size * sizeof(T) bytes. Split
 *        larger transfers into multiple get requests whose byte size does not exceed 256 MB.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE. See @ref aclshmemx_udma_relay_put_nbi.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data (on pe).
 * @param buf               [in] Pointer on local UB. WQE staging scratch when
 *                               WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the actual source. Must satisfy the preconditions above.
 * @param relay_pe          [in] PE whose port path is used to forward the packet. Must satisfy the
 *                               preconditions above.
 * @param sync_id           [in] Hardware event ID used by the MTE3->S sync in the PIPE_MTE3 path.
 *                               Ignored when WQE_PIPE == PIPE_S. Defaults to 0.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    __gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id = 0);

/**
 * @brief GlobalTensor/LocalTensor overload of @ref aclshmemx_udma_relay_get_nbi.
 *
 * @note  Same PE-argument preconditions and non-blocking completion semantics as the bare-pointer
 *        overload: invalid (pe, relay_pe) combinations submit nothing (abort in debug) and cannot be
 *        detected via the void return; a normal return only means the WQE was published, so call
 *        aclshmemx_udma_quiet(pe) before reading @p dst.
 * @note  A single UDMA get request transfers at most 256 MB (256 * 1024 * 1024 bytes); split larger
 *        transfers into multiple get requests whose byte size (@p elem_size * sizeof(T)) does not
 *        exceed 256 MB.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           Pipe used to publish the WQE. See the bare-pointer overload for semantics.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data (on pe).
 * @param buf               [in] LocalTensor on local UB. WQE staging scratch when
 *                               WQE_PIPE == PIPE_MTE3; ignored when WQE_PIPE == PIPE_S.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the actual source. Must satisfy the preconditions above.
 * @param relay_pe          [in] PE whose port path is used to forward the packet. Must satisfy the
 *                               preconditions above.
 * @param sync_id           [in] Hardware event ID used by the MTE3->S sync in the PIPE_MTE3 path.
 *                               Ignored when WQE_PIPE == PIPE_S. Defaults to 0.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_relay_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, int relay_pe, uint32_t sync_id = 0);

/**
 * @brief Asynchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and
 *        updating a remote signal flag on completion using UDMA.
 *        Template function for different data types.
 *        No-buf overload: not templated on WQE_PIPE and takes no UB scratch. Submits the WQE
 *        through the direct (non-staged) path, behaviorally equivalent to the buf-taking
 *        overload with WQE_PIPE = PIPE_S. Provided to match the original signature for
 *        backward compatibility with existing callers; new callers should use the buf-taking
 *        overload (which defaults to WQE_PIPE = PIPE_MTE3).
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 *
 * @note A single UDMA put request transfers at most 256 MB
 *       (256 * 1024 * 1024 bytes). @p elem_size is the number of T
 *       elements, so the transfer size is @p elem_size * sizeof(T) bytes.
 *       Split larger transfers into multiple put requests whose byte size
 *       does not exceed 256 MB. After submitting one or more asynchronous
 *       requests, call the matching completion/synchronization interface,
 *       such as @ref aclshmemx_udma_quiet or the corresponding signal wait
 *       protocol, before reading the destination data or reusing buffers
 *       whose contents must remain stable.
 *
 * @param dst                 [in] Pointer on Symmetric memory of the destination data.
 * @param src                 [in] Pointer on local device of the source data.
 * @param elem_size           [in] Number of elements in the dest and source arrays.
 * @param sig_addr            [in] Symmetric address of the signal word to be updated.
 * @param signal              [in] The value used to update sig_addr.
 * @param pe                  [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe);

/**
 * @brief Buf-taking overload of @ref aclshmemx_udma_put_signal_nbi. PIPE_MTE3 (default)
 *        stages the WRITE_WITH_NOTIFY WQE in the caller-provided UB scratch and
 *        DataCopyPads it to the SQ ring. PIPE_S falls through to the no-buf path and
 *        ignores @p buf and @p sync_id; provided so a single call site can be
 *        templated on WQE_PIPE.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @tparam T                  Element type of the transfer.
 * @tparam WQE_PIPE           PIPE_MTE3 (default) or PIPE_S; other pipes are not supported.
 *
 * @note A single UDMA put request transfers at most 256 MB
 *       (256 * 1024 * 1024 bytes). @p elem_size is the number of T
 *       elements, so the transfer size is @p elem_size * sizeof(T) bytes.
 *       Split larger transfers into multiple put requests whose byte size
 *       does not exceed 256 MB. After submitting one or more asynchronous
 *       requests, call the matching completion/synchronization interface,
 *       such as @ref aclshmemx_udma_quiet or the corresponding signal wait
 *       protocol, before reading the destination data or reusing buffers
 *       whose contents must remain stable.
 *
 * @param dst                 [in] Pointer on Symmetric memory of the destination data.
 * @param src                 [in] Pointer on local device of the source data.
 * @param elem_size           [in] Number of elements in the dest and source arrays.
 * @param sig_addr            [in] Symmetric address of the signal word to be updated.
 * @param signal              [in] The value used to update sig_addr.
 * @param pe                  [in] PE number of the remote PE.
 * @param buf                 [in] Pointer on local UB used as WQE staging scratch
 *                                 (must hold one full WRITE_WITH_NOTIFY WQE block;
 *                                 128 B is safe). Ignored when WQE_PIPE == PIPE_S.
 * @param sync_id             [in] Hardware event ID used by the MTE3->S sync after
 *                                 DataCopyPad in the PIPE_MTE3 path. Ignored when
 *                                 WQE_PIPE == PIPE_S. Defaults to 0.
 */
template <typename T, pipe_t WQE_PIPE = PIPE_MTE3>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe,
    __ubuf__ uint8_t* buf, uint32_t sync_id = 0);

/**
 * @brief UDMA Quiet function. This synchronous function ensures all previous UDMA WQEs are completed
 * (data has arrived at the destination PE).
 *
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_udma_quiet(int pe);

/**
 * @brief Asynchronous interface. Add value to dst (remote symmetric address) on the specified PE pe,
 * and atomically update the dst without returning the value. Supported types: int32, uint32, int64, uint64, float.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Add value to dst (remote symmetric address) on the specified PE pe,
 * and return the previous content of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous content of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Conditionally update dst (remote symmetric address) on the specified PE pe
 * and return the previous content of dst. If cond and the remote dst value are equal,
 * then value is swapped into the remote dst; otherwise, the remote dst is unchanged. In either case, the old
 * value of the remote dest is returned. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param cond              [in] condition for swap
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous content of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe);

/**
 * @brief Synchronous interface. Fetch the contents of dst (remote symmetric address) on the specified PE pe
 * and return the contents. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch(__gm__ T* dst, int32_t pe);

/**
 * @brief Synchronous interface. Set value to dst (remote symmetric address) on the specified PE pe
 * without returning a value. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Value to be atomically written to the remote PE.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_set(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Swap value to dst (remote symmetric address) on the specified PE pe
 * and return the previous contents of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Value to be atomically written to the remote PE.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_swap(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Increment dst (remote symmetric address) on the specified PE pe by one
 * and return the previous contents of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Synchronous interface. Increment dst (remote symmetric address) on the specified PE pe by one
 * without returning a value. Supported types: int32, uint32, int64, uint64, float.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_inc(__gm__ T* dst, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_and(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_and(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_or(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_or(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_xor(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_xor(__gm__ T* dst, T value, int32_t pe);

#include "gm2gm/engine/shmem_device_udma.hpp" // IWYU pragma: keep

#endif
