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
#ifndef SHMEM_HOST_INIT_H
#define SHMEM_HOST_INIT_H

#include "host/shmem_host_def.h"
#include "host_device/shmem_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Query the current initialization status.
 *
 * @return Returns initialization status. Returning ACLSHMEM_STATUS_IS_INITIALIZED indicates that initialization is
 *         complete. All return types can be found in <b>\ref aclshmemx_init_status_t</b>.
 */
ACLSHMEM_HOST_API int aclshmemx_init_status(void);
#define shmem_init_status aclshmemx_init_status

/**
 * @brief get the unique id and return it by input argument uid. This function need run with PTA.
 *
 * @param uid               [out] a ptr to uid generate by shmem
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int aclshmemx_get_uniqueid(aclshmemx_uniqueid_t *uid);
#define shmem_get_uniqueid aclshmemx_get_uniqueid

/**
 * @brief init process with unique id. This function need run with PTA.
 *
 * @param my_pe                 [in] Local PE ID, must be less than the maximum supported PEs (my_pe < ACLSHMEM_MAX_PES)
 * @param n_pes                 [in] Total number of PEs, must be less than or equal to the maximum supported PEs (n_pes <= ACLSHMEM_MAX_PES)
 * @param local_mem_size        [in] Allocated local memory size , must be less than the maximum supported local memory size (local_mem_size < ACLSHMEM_MAX_LOCAL_SIZE)
 * @param uid                   [in] Unique ID obtained from <b>aclshmemx_get_uniqueid()</b>
 * @param aclshmem_attr         [in/out] Attribute struct, output parameter of this interface and input parameter for subsequent initialization functions <b>aclshmemx_init_attr()</b>
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int aclshmemx_set_attr_uniqueid_args(int my_pe, int n_pes, int64_t local_mem_size,
                                    aclshmemx_uniqueid_t *uid,
                                    aclshmemx_init_attr_t *aclshmem_attr);
#define shmem_set_attr_uniqueid_args aclshmemx_set_attr_uniqueid_args


/**
 * @brief Initialize the resources required for ACLSHMEM task based on attributes.
 *        Attributes can be created by users or obtained by calling <b>aclshmemx_set_attr_uniqueid_args()</b>.
 *        if the self-created attr structure is incorrect, the initialization will fail.
 *        It is recommended to build the attributes by <b>aclshmemx_set_attr_uniqueid_args()</b>.
 *
 * @param bootstrap_flags   [in] bootstrap_flags for init.
 * @param attributes        [in] Pointer to the user-defined attributes.
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int aclshmemx_init_attr(aclshmemx_bootstrap_t bootstrap_flags, aclshmemx_init_attr_t *attributes);
#define shmem_init_attr aclshmemx_init_attr

/**
 * @brief Release all resources used by the CURRENT instance of the ACLSHMEM library.
 *
 *        Single-instance mode: finalizes instance 0, behaving identically to the
 *        legacy no-argument API.
 *
 *        Multi-instance mode: finalizes the instance currently active, i.e. the one
 *        most recently selected via aclshmemx_instance_ctx_set (or instance 0 if no
 *        instance has been set). To release a specific instance with this API, call
 *        aclshmemx_instance_ctx_set(instance_id) first, then call this function.
 *        After finalization, if the destroyed instance was non-zero, the context
 *        falls back to instance 0.
 *
 *        For multi-instance scenarios it is recommended to use aclshmemx_finalize
 *        directly, which releases a specific instance by id without an explicit
 *        context switch.
 *
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int aclshmem_finalize(void);
#define shmem_finalize aclshmem_finalize

/**
 * @brief Release resources of a SPECIFIC instance (multi-instance extension).
 *        Switches to the target instance context first (if it is not already
 *        active), then finalizes it. After finalization, if the destroyed
 *        instance was non-zero, the context falls back to instance 0.
 *        Recommended for multi-instance scenarios.
 *
 * @param instance_id the unique identifier of the instance to finalize.
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int aclshmemx_finalize(uint64_t instance_id);
#define shmemx_finalize aclshmemx_finalize


/**
 * @brief returns the major and minor version.
 *
 * @param major [out] major version
 *
 * @param minor [out] minor version
 */
ACLSHMEM_HOST_API void aclshmem_info_get_version(int *major, int *minor);
#define shmem_info_get_version aclshmem_info_get_version


/**
 * @brief returns the vendor defined name string.
 *
 * @param name [out] name
 */
ACLSHMEM_HOST_API void aclshmem_info_get_name(char *name);
#define shmem_info_get_name aclshmem_info_get_name


/**
 * @brief Set the TLS private key and password, and register a decrypt key password handler.
 *
 * @param tls_pk the content of tls private key
 * @param tls_pk_len length of tls private key
 * @param tls_pk_pw the content of tls private key password
 * @param tls_pk_pw_len length of tls private key password
 * @param decrypt_handler decrypt function pointer
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int32_t aclshmemx_set_config_store_tls_key(const char *tls_pk, const uint32_t tls_pk_len,
    const char *tls_pk_pw, const uint32_t tls_pk_pw_len, const aclshmem_decrypt_handler decrypt_handler);
#define shmem_set_config_store_tls_key aclshmemx_set_config_store_tls_key


/**
 * @brief exit all ranks.
 *
 * @param status [in] exit status code
 */
ACLSHMEM_HOST_API void aclshmem_global_exit(int status);
#define shmem_global_exit aclshmem_global_exit

/**
 * @brief aclshmemx_set_conf_store_tls.
 *
 * @param enable whether to enable tls
 * @param tls_info the format described in memfabric SECURITYNOTE.md, if disabled tls_info won't be use
 * @param tls_info_len length of tls_info, if disabled tls_info_len won't be use
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int32_t aclshmemx_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len);
#define shmem_set_conf_store_tls aclshmemx_set_conf_store_tls

/**
 * @brief Get the current instance context.
 *
 * @return Returns a pointer to the current instance context on success or NULL on failure.
 */
ACLSHMEM_HOST_API aclshmem_instance_ctx* aclshmemx_instance_ctx_get();

/**
 * @brief Set the current instance context by instance_id.
 *
 * @param instance_id the unique identifier of the instance to set as the current context.
 * @return Returns 0 on success or an error code on failure
 */
ACLSHMEM_HOST_API int aclshmemx_instance_ctx_set(uint64_t instance_id);

#ifdef __cplusplus
}
#endif

#endif