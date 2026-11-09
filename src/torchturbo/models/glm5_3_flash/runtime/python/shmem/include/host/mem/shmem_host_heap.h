/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_HOST_HEAP_H
#define SHMEM_HOST_HEAP_H

#include "host/shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory is not initialized.
 *        If <i>size</i> is 0, then <b>aclshmem_malloc()</b> returns NULL.
 *
 * @param size             [in] bytes to be allocated
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void *aclshmem_malloc(size_t size);
#define shmem_malloc aclshmem_malloc


/**
 * @brief allocate memory for an array of <i>nmemb</i> elements of <i>size</i> bytes each and returns a pointer to the
 *        allocated memory. The memory is set to zero. If <i>nmemb</i> or <i>size</i> is 0, then <b>aclshmem_calloc()</b>
 *        returns either NULL.
 *
 * @param nmemb            [in] elements count
 * @param size             [in] each element size in bytes
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void *aclshmem_calloc(size_t nmemb, size_t size);
#define shmem_calloc aclshmem_calloc


/**
 * @brief allocate <i>size</i> bytes and returns a pointer to the allocated memory. The memory address will be a
 *        multiple of <i>alignment</i>, which must be a power of two.
 *
 * @param alignment        [in] memory address alignment
 * @param size             [in] bytes allocated
 * @return pointer to the allocated memory.
 */
ACLSHMEM_HOST_API void *aclshmem_align(size_t alignment, size_t size);
#define shmem_align aclshmem_align

/**
 * @brief Free the memory space pointed to by <i>ptr</i>, which must have been returned by a previous call to
 *       <b>aclshmem_malloc()</b>, <b>aclshmem_calloc()</b> or <b>aclshmem_align()</b>. If <i>ptr</i> is NULL,
 *       no operation is performed.
 * @param ptr              [in] point to memory block to be free.
 */
ACLSHMEM_HOST_API void aclshmem_free(void *ptr);
#define shmem_free aclshmem_free

/**
 * @brief Allocates a block of ACLSHMEM symmetric memory, the data in this memory is uninitialized
 *
 * @param size          [in] Memory allocation size (in bytes)
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the symmetric memory
 *
 */
ACLSHMEM_HOST_API void *aclshmemx_malloc(size_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Allocates a block of SHMEM symmetric memory and initializes all content to zero
 *
 * @param count         [in] Number of elements
 * @param size          [in] Number of bytes occupied by each element
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the symmetric memory
 */
ACLSHMEM_HOST_API void *aclshmemx_calloc(size_t count, size_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Allocates a block of SHMEM symmetric memory with alignment to the specified length
 *
 * @param alignment     [in] Alignment length (in bytes)
 * @param size          [in] Memory allocation size (in bytes)
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the symmetric memory
 */
ACLSHMEM_HOST_API void *aclshmemx_align(size_t alignment, size_t size, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Frees the allocated symmetric memory
 *
 * @param ptr           [in] Pointer to the memory to be freed
 * @param mem_type      [in] Allocation location of the symmetric memory (Host/Device)
 */
ACLSHMEM_HOST_API void aclshmemx_free(void *ptr, aclshmem_mem_type_t mem_type = DEVICE_SIDE);

/**
 * @brief Returns the start address (heap_base) of the local symmetric memory heap.
 *        If <i>mem_type</i> is DEVICE_SIDE, returns the device heap base address.
 *        If <i>mem_type</i> is HOST_SIDE, returns the host heap base address.
 *        Must be called after ACLSHMEM initialization, otherwise returns NULL.
 *
 * @param mem_type      [in] Allocation location of symmetric memory (Host/Device)
 * @return Pointer to the start address of the symmetric memory heap, or NULL if not initialized.
 * @note Thread safety: This function is thread-safe for reading heap_base after initialization.
 *       The heap_base value is set during initialization and does not change afterwards.
 * @note Remote visibility: The returned address is local-only, not directly accessible by remote PEs.
 * @note Failure conditions: Returns NULL if ACLSHMEM is not initialized or the heap has not been allocated.
 */
ACLSHMEM_HOST_API void *aclshmemx_get_heap_base(aclshmem_mem_type_t mem_type = DEVICE_SIDE);

#ifdef __cplusplus
}
#endif
#endif