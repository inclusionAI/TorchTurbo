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
#ifndef SHMEM_HOST_DEF_H
#define SHMEM_HOST_DEF_H
#include <climits>
#include <stdint.h>


#include "host_device/shmem_common_types.h"
#include "host_device/shmem_common_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup group_constants Constants
 * @{
 */
 
/// \brief Inner length of the unique ID buffer for ACLSHMEM
constexpr uint16_t ACLSHMEM_UNIQUE_ID_INNER_LEN = 124;
/// \brief Default timeout value (in seconds) for ACLSHMEM operations
constexpr int DEFAULT_TIMEOUT = 120;

/**@} */

/**
 * @defgroup group_macros Macros
 * @{
*/
/// \def ACLSHMEM_HOST_API
/// \brief A macro that identifies a function on the host side.
#define ACLSHMEM_HOST_API __attribute__((visibility("default")))

/// \def ACLSHMEM_MAJOR_VERSION
/// \brief Macros that define major version info of ACLSHMEM
#define ACLSHMEM_MAJOR_VERSION 1

/// \def ACLSHMEM_MINOR_VERSION
/// \brief Macros that define minor version info of ACLSHMEM
#define ACLSHMEM_MINOR_VERSION 1

/// \def ACLSHMEM_MAX_NAME_LEN
/// \brief Maximum length of the name string in ACLSHMEM (including null terminator)
#define ACLSHMEM_MAX_NAME_LEN 256

/// \def ACLSHMEM_VENDOR_MAJOR_VER
/// \brief Macros that define vendor major version info of ACLSHMEM
#define ACLSHMEM_VENDOR_MAJOR_VER 1

/// \def ACLSHMEM_VENDOR_MINOR_VER
/// \brief Macros that define vendor minor version info of ACLSHMEM
#define ACLSHMEM_VENDOR_MINOR_VER 1

/// \def ACLSHMEM_VENDOR_PATCH_VER
/// \brief Macros that define vendor patch version info of ACLSHMEM
#define ACLSHMEM_VENDOR_PATCH_VER 1

/// \def ACLSHMEM_MAX_IP_PORT_LEN
/// \brief Maximum length of the IP and port string in ACLSHMEM (including null terminator)
#define ACLSHMEM_MAX_IP_PORT_LEN 64

/**@} */  // end of group_macros

/**
 * @defgroup group_enums Enumerations
 * @{
*/

using Result = int32_t;

/**
 * @brief Error code for the ACLSHMEM library.
*/
enum aclshmem_error_code_t : int {
    ACLSHMEM_SUCCESS = 0,         ///< Task execution was successful.
    ACLSHMEM_INVALID_PARAM = -1,  ///< There is a problem with the parameters.
    ACLSHMEM_INVALID_VALUE = -2,  ///< There is a problem with the range of the value of the parameter.
    ACLSHMEM_SMEM_ERROR = -3,     ///< There is a problem with SMEM.
    ACLSHMEM_INNER_ERROR = -4,    ///< This is a problem caused by an internal error.
    ACLSHMEM_NOT_INITED = -5,     ///< This is a problem caused by an uninitialization.
    ACLSHMEM_BOOTSTRAP_ERROR = -6,///< This is a problem with BOOTSTRAP.
    ACLSHMEM_TIMEOUT_ERROR = -7,  ///< This is a problem caused by TIMEOUT.
    ACLSHMEM_MALLOC_FAILED = -8,   ///< This is a problem when malloc.
    ACLSHMEM_DL_FUNC_FAILED = -9,  ///< This is a problem when dll func failed.
    ACLSHMEM_INNER_TIMEOUT = -10,  ///< This is a problem when inner timeout.
    ACLSHMEM_UNDER_API_UNLOAD = -11,  ///< This is a problem when under api lib load failed.
    ACLSHMEM_NOT_SUPPORTED = -100,    ///< This is a problem when function not supported.
};
#define shmem_error_code_t aclshmem_error_code_t


/**
 * @brief init flags
*/
enum aclshmemx_bootstrap_t : unsigned int {
    ACLSHMEMX_INIT_WITH_DEFAULT = 1U << 0,  ///< Default mode: Priority is ipport > uid (comm_args), if neither provided returns error
    ACLSHMEMX_INIT_WITH_MPI = 1U << 1,      ///< MPI mode
    ACLSHMEMX_INIT_WITH_UNIQUEID = 1U << 3,   ///< Unique ID mode
    ACLSHMEMX_INIT_MAX = 1U << 31
};
#define shmemx_bootstrap_t aclshmemx_bootstrap_t

/**
 * @brief The state of the ACLSHMEM library initialization.
*/
enum aclshmemx_init_status_t {
    ACLSHMEM_STATUS_NOT_INITIALIZED = 0,  ///< Uninitialized.
    ACLSHMEM_STATUS_SHM_CREATED,          ///< Shared memory heap creation is complete.
    ACLSHMEM_STATUS_IS_INITIALIZED,       ///< Initialization is complete.
    ACLSHMEM_STATUS_INVALID = INT_MAX,    ///< Invalid status code.
};
#define shmem_init_status_t aclshmemx_init_status_t

/**
 * @brief Different transports supported by ACLSHMEM library.
*/
enum aclshmem_transport_t : uint8_t {
    ACLSHMEM_TRANSPORT_MTE = 1 << 0,    ///< MTE Transport.
    ACLSHMEM_TRANSPORT_ROCE = 1 << 1,   ///< RDMA Transport (RoCE).
    ACLSHMEM_TRANSPORT_SDMA = 1 << 2,   ///< SDMA Transport.
    ACLSHMEM_TRANSPORT_UDMA = 1 << 3,   ///< UDMA Transport.
};
#define shmem_transport_t aclshmem_transport_t

/**@} */  // end of group_enums

/**
 * @defgroup group_structs Structs
 * @{
*/

/**
 * @struct aclshmem_init_optional_attr_t
 * @brief Optional parameter for the attributes used for initialization.
 *
 * - int version: version
 * - data_op_engine_type_t data_op_engine_type: data_op_engine_type
 * - uint32_t shm_init_timeout: shm_init_timeout
 * - uint32_t shm_create_timeout: shm_create_timeout
 * - uint32_t control_operation_timeout: control_operation_timeout
 * - int32_t sockFd: sock_fd for apply port in advance
*/
typedef struct {
    int version;
    data_op_engine_type_t data_op_engine_type;
    uint32_t shm_init_timeout;
    uint32_t shm_create_timeout;
    uint32_t control_operation_timeout;
    int32_t sockFd;
} aclshmem_init_optional_attr_t;
#define shmem_init_optional_attr_t aclshmem_init_optional_attr_t

/**
 * @struct aclshmemx_init_attr_t
 * @brief Mandatory parameter for attributes used for initialization.
 *
 * - int my_pe: The pe of the current process.
 * - int n_pes: The total pe number of all processes.
 * - char ip_port[ACLSHMEM_MAX_IP_PORT_LEN]: The ip and port of the communication server. The port must not conflict
 *   with other modules and processes.
 * - uint64_t local_mem_size: The size of shared memory currently occupied by current pe.
 * - aclshmem_init_optional_attr_t option_attr: Optional Parameters.
 * - void *comm_args: Parameters required for communication during the bootstrap phase when initializing different flags.
*/
typedef struct aclshmemx_init_attr_t {
    int my_pe;
    int n_pes;
    char ip_port[ACLSHMEM_MAX_IP_PORT_LEN] = {};
    uint64_t local_mem_size;
    aclshmem_init_optional_attr_t option_attr = {(1 << 16) + sizeof(aclshmem_init_optional_attr_t), ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    void *comm_args = nullptr;
    uint64_t instance_id = 0;
} aclshmemx_init_attr_t;
#define shmem_init_attr_t aclshmemx_init_attr_t


/**
 * @struct aclshmemx_uniqueid_t
 * @brief Structure required for SHMEM unique ID (uid) initialization
 *
 * - int32_t version: version.
 * - int my_pe: The pe of the current process.
 * - int n_pes: The total pe number of all processes.
 * - char internal[ACLSHMEM_UNIQUE_ID_INNER_LEN]: Internal information of uid.
*/
typedef struct {
    int32_t version;
    int my_pe;
    int n_pes;
    char internal[ACLSHMEM_UNIQUE_ID_INNER_LEN];
} aclshmemx_uniqueid_t;
#define shmem_uniqueid_t aclshmemx_uniqueid_t

/**@} */  // end of group_structs

/**
 * @brief Callback function of private key password decryptor, see aclshmemx_set_config_store_tls_key
 *
 * @param cipherText       [in] the encrypted text(private password)
 * @param cipherTextLen    [in] the length of encrypted text
 * @param plainText        [out] the decrypted text(private password)
 * @param plainTextLen     [out] the length of plainText
 */
typedef int (*aclshmem_decrypt_handler)(const char *cipherText, size_t cipherTextLen, char *plainText,
                                     size_t &plainTextLen);
#define shmem_decrypt_handler aclshmem_decrypt_handler
/**
 * @addtogroup group_constants
 * @{
*/
/// \brief Version number of the ACLSHMEM unique ID structure
constexpr int32_t ACLSHMEM_UNIQUEID_VERSION = (1 << 16) + sizeof(aclshmemx_uniqueid_t);
/**@} */ // end of group_constants

/**
 * @addtogroup group_macros
 * @{
*/
/// \def ACLSHMEM_UNIQUEID_INITIALIZER
/// \brief Initializer macro for the ACLSHMEM unique ID structure
#define ACLSHMEM_UNIQUEID_INITIALIZER                   \
    {                                                   \
        ACLSHMEM_UNIQUEID_VERSION,                      \
        {                                               \
            0                                           \
        }                                               \
    }

/**@} */ // end of group_macros
#ifdef __cplusplus
}
#endif

#endif