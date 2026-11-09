#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# SHMEM 安装前环境检测脚本
# 功能：在 SHMEM 包安装前检测目标环境的芯片平台、版本基线、拓扑链路、引擎能力和包内容
# 用法：./preinstall_check.sh [--package <pkg_dir>] [--help]

set -euo pipefail

# ============================================================
# 全局状态变量
# ============================================================
CHIP="UNKNOWN"
CHIP_MODEL="UNKNOWN"
CARD_COUNT=0
HDK_VERSION="UNKNOWN"
HDK_VERSION_CHECK="UNKNOWN"
CANN_VERSION="UNKNOWN"
CANN_VERSION_CHECK="UNKNOWN"
TOPOLOGY="UNKNOWN"
MTE_SUPPORT="UNKNOWN"
SDMA_SUPPORT="UNKNOWN"
UDMA_SUPPORT="UNKNOWN"
RDMA_SUPPORT="UNKNOWN"
RDMA_NIC_TYPE="UNKNOWN"
RDMA_HCCN_TOOL_OK="UNKNOWN"
PKG_HAS_SDM_TRANSPORT="UNKNOWN"
PKG_HAS_RDM_TRANSPORT="UNKNOWN"
PKG_HAS_UDM_TRANSPORT="UNKNOWN"
PKG_RDMA_BACKEND="UNKNOWN"
PKG_CANN_VERSION="UNKNOWN"
PKG_HAS_ROOT_INFO="UNKNOWN"
PKG_SOC_TYPE="UNKNOWN"
PKG_UDMA_SUPPORT="UNKNOWN"

PKG_DIR=""

# 最小版本要求：CANN >= 9.0.0.beta.2（910B SDMA 需要）
MIN_CANN_MAJOR=9
MIN_CANN_MINOR=0

# ============================================================
# 工具函数
# ============================================================
log_info()  { echo -e "\033[32m[INFO]\033[0m $1"; }
log_warn()  { echo -e "\033[33m[WARN]\033[0m $1"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $1"; }
log_step()  { echo -e "\n\033[34m============================================================\033[0m"; echo -e "\033[34m  $1\033[0m"; echo -e "\033[34m============================================================\033[0m"; }

# 版本号比较（格式：major.minor.patch[.beta.N]）
# 规则：先比三段数字，相等时 release > beta；都有 beta 时比 beta 数字
# 返回 0 表示 $1 >= $2
version_gte() {
    local v1="$1" v2="$2"
    # 提取纯数字部分
    local v1_clean=$(echo "$v1" | grep -oP '^\d+\.\d+\.\d+' || echo "0.0.0")
    local v2_clean=$(echo "$v2" | grep -oP '^\d+\.\d+\.\d+' || echo "0.0.0")

    local IFS=.
    local v1_arr=($v1_clean) v2_arr=($v2_clean)

    for i in 0 1 2; do
        local a=${v1_arr[$i]:-0} b=${v2_arr[$i]:-0}
        if [ "$a" -gt "$b" ]; then return 0; fi
        if [ "$a" -lt "$b" ]; then return 1; fi
    done

    # 三段数字相等时，比较 beta 后缀
    local v1_beta=$(echo "$v1" | grep -oP 'beta\.\K\d+' || echo "")
    local v2_beta=$(echo "$v2" | grep -oP 'beta\.\K\d+' || echo "")
    # v1 非 beta（release），v2 是 beta → v1 >= v2
    if [ -z "$v1_beta" ] && [ -n "$v2_beta" ]; then return 0; fi
    # v1 是 beta，v2 非 beta → v1 < v2
    if [ -n "$v1_beta" ] && [ -z "$v2_beta" ]; then return 1; fi
    # 都有 beta，比较数字
    if [ -n "$v1_beta" ] && [ -n "$v2_beta" ]; then
        if [ "$v1_beta" -ge "$v2_beta" ]; then return 0; else return 1; fi
    fi
    # 都没有 beta，相等
    return 0
}

# ============================================================
# 第一步：确定芯片平台
# ============================================================
step1_check_chip_platform() {
    log_step "第一步：确定芯片平台"

    if ! command -v npu-smi &>/dev/null; then
        log_error "npu-smi 命令不可用，无法检测芯片平台"
        CHIP="UNKNOWN"
        return 1
    fi

    local npu_info
    npu_info=$(npu-smi info 2>&1 | sed $'s/\033\[[0-9;]*[a-zA-Z]//g' | tr -d '\r') || true
    if [ -z "$npu_info" ]; then
        log_error "npu-smi info 无输出，无法检测芯片平台"
        CHIP="UNKNOWN"
        return 1
    fi

    # 提取芯片型号
    # 910B/C 系列芯片名可能为 Ascend910 / Ascend 910B / Ascend 910C 等
    if echo "$npu_info" | grep -qiE "Ascend.*910|910[BbCc]"; then
        CHIP="910B/C"
        log_info "检测到芯片型号：910B/C 系列"
    elif echo "$npu_info" | grep -qi "Ascend950\|Ascend 950\|950"; then
        CHIP="950"
        log_info "检测到芯片型号：Ascend 950"
    else
        # 尝试更宽泛的匹配
        local chip_line
        chip_line=$(echo "$npu_info" | grep -m1 -iE "(Chip|910|950|Ascend)")
        if [ -n "$chip_line" ]; then
            log_warn "芯片型号未能精确识别，原始输出：$chip_line"
            CHIP=$(echo "$chip_line" | awk '{print $NF}')
        else
            log_error "无法从 npu-smi info 输出中识别芯片型号"
            CHIP="UNKNOWN"
            return 1
        fi
    fi

    # 提取卡数：优先从 npu-smi info 输出统计唯一 NPU ID（设备编号）
    # 910B: 每卡 1 芯片，NPU 编号 = Phy-ID，如 0-7 共 8 卡
    # 910C: 每卡 2 芯片，NPU 编号 = 0-7 共 8 卡，Phy-ID = 0-15 共 16 个
    # 提取规则：第2列为数字(NPU ID)、第3列为字母开头(芯片名)的行
    #   如 "| 0     Ascend910  ..." → NPU ID=0
    #   排除子芯片行 "| 0     0  ..." (第3列为数字)
    local npu_card_count=0
    npu_card_count=$(echo "$npu_info" | awk '$2 ~ /^[0-9]+$/ && $3 ~ /^[A-Za-z]/ {print $2}' | sort -n | uniq | wc -l)

    if [ "$npu_card_count" -gt 0 ]; then
        CARD_COUNT=$npu_card_count
    else
        # 兜底：从 npu-smi info -t topo 的 header 行提取
        local topo_output
        topo_output=$(npu-smi info -t topo 2>&1 | sed $'s/\033\[[0-9;]*[a-zA-Z]//g' | tr -d '\r') || true
        if [ -n "$topo_output" ]; then
            # 910B 格式：NPU0, NPU1, ...
            # 仅统计唯一的 NPU 标签，避免 header 行 + 每行行标签导致重复计数
            CARD_COUNT=$(echo "$topo_output" | grep -oE 'NPU[0-9]+' | sort -u | wc -l)
            if [ "$CARD_COUNT" -eq 0 ]; then
                # 910C 格式：Phy-ID0, Phy-ID1, ...
                # 仅取 header 行（第一行）避免与行标签重复计数
                local phy_count
                phy_count=$(echo "$topo_output" | head -n 1 | grep -oE 'Phy-ID[0-9]+' | wc -l)
                CARD_COUNT=$((phy_count / 2))
            fi
        else
            CARD_COUNT=0
        fi
    fi
    log_info "卡数：$CARD_COUNT"

    return 0
}

# ============================================================
# 第二步：检查版本基线
# ============================================================
step2_check_version_baseline() {
    log_step "第二步：检查版本基线"

    # --- HDK 版本 ---
    local npu_info
    npu_info=$(npu-smi info 2>&1 | sed $'s/\033\[[0-9;]*[a-zA-Z]//g' | tr -d '\r') || true
    if [ -n "$npu_info" ]; then
        HDK_VERSION=$(echo "$npu_info" | grep -m1 -ioP 'Version:\s*\K\S+' || echo "UNKNOWN")
    fi
    log_info "HDK 版本：$HDK_VERSION"

    # --- CANN 版本 ---
    local version_file=""
    if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        # 尝试多个可能的 version.info 路径
        for candidate in \
            "$ASCEND_HOME_PATH/opp/version.info" \
            "$ASCEND_HOME_PATH/version.info"; do
            if [ -f "$candidate" ]; then
                version_file="$candidate"
                break
            fi
        done
    fi

    if [ -n "$version_file" ] && [ -f "$version_file" ]; then
        CANN_VERSION=$(grep -m1 -iE "Version|version" "$version_file" | awk -F'=' '{print $NF}' | tr -d ' "' || echo "UNKNOWN")
        # 如果格式不对，尝试直接读取
        if [ "$CANN_VERSION" = "UNKNOWN" ] || [ -z "$CANN_VERSION" ]; then
            CANN_VERSION=$(head -5 "$version_file" | grep -m1 -oP '(\d+\.\d+[^\s]*)' || echo "UNKNOWN")
        fi
    fi
    # 兜底：从 version.cfg 提取 toolkit_upgrade_version
    if [ "$CANN_VERSION" = "UNKNOWN" ] && [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/version.cfg" ]; then
        CANN_VERSION=$(grep -i "toolkit_upgrade_version" "${ASCEND_HOME_PATH}/version.cfg" | grep -m1 -oP '\[\K[0-9.]+' || echo "UNKNOWN")
    fi
    if [ -z "$CANN_VERSION" ]; then
        CANN_VERSION="UNKNOWN"
    fi
    log_info "CANN 版本：$CANN_VERSION"

    # 版本基线判定
    if [ "$CANN_VERSION" != "UNKNOWN" ]; then
        if version_gte "$CANN_VERSION" "8.2.0"; then
            CANN_VERSION_CHECK="OK (>= 8.2.0)"
        else
            CANN_VERSION_CHECK="LOW (< 8.2.0，建议升级)"
        fi
    else
        CANN_VERSION_CHECK="UNKNOWN（无法获取版本）"
    fi
    log_info "CANN 版本基线判定：$CANN_VERSION_CHECK"

    if [ "$HDK_VERSION" != "UNKNOWN" ]; then
        HDK_VERSION_CHECK="已获取"
    else
        HDK_VERSION_CHECK="UNKNOWN（无法获取版本）"
    fi
    log_info "HDK 版本基线判定：$HDK_VERSION_CHECK"

    return 0
}

# ============================================================
# 第三步：检测拓扑链路
# ============================================================
step3_check_topology() {
    log_step "第三步：检测拓扑链路"

    if ! command -v npu-smi &>/dev/null; then
        log_error "npu-smi 命令不可用，无法检测拓扑"
        TOPOLOGY="UNKNOWN"
        MTE_SUPPORT="UNKNOWN"
        return 1
    fi

    local topo_output
    topo_output=$(npu-smi info -t topo 2>&1 | sed $'s/\033\[[0-9;]*[a-zA-Z]//g' | tr -d '\r') || true

    if [ -z "$topo_output" ]; then
        log_warn "npu-smi info -t topo 无输出"
        TOPOLOGY="无输出"
    else
        log_info "拓扑输出已获取"
        # 保存摘要用于报告
        TOPOLOGY=$(echo "$topo_output" | head -30)
    fi

    # 判定 MTE 支持
    if echo "$CHIP" | grep -qi "950"; then
        # Ascend950 默认支持 MTE
        MTE_SUPPORT="是（Ascend950 默认支持）"
        log_info "MTE 支持：是（Ascend950 默认支持）"
    elif echo "$CHIP" | grep -qi "910"; then
        if [ -n "$topo_output" ]; then
            if echo "$topo_output" | grep -qiE "HCCS|SIO"; then
                MTE_SUPPORT="是（含 HCCS/SIO 链路）"
                log_info "MTE 支持：是（检测到 HCCS/SIO 链路）"
            elif echo "$topo_output" | grep -qiE "SYS|PHB|PIX|PXB"; then
                # 仅 PCIe 链路
                MTE_SUPPORT="否（仅 PCIe 链路：SYS/PHB/PIX/PXB）"
                log_warn "MTE 支持：否（仅检测到 PCIe 链路，不支持 MTE）"
            else
                MTE_SUPPORT="未知（无法识别的链路类型）"
                log_warn "MTE 支持：未知（无法识别链路类型）"
            fi
        else
            MTE_SUPPORT="未知（无拓扑数据）"
        fi
    else
        MTE_SUPPORT="未知（芯片型号未识别）"
        log_warn "MTE 支持：未知（需先确认芯片型号）"
    fi

    return 0
}

# ============================================================
# 第四步：检测 SDMA（仅 910B/C）
# ============================================================
step4_check_sdma() {
    log_step "第四步：检测 SDMA（适用平台：910B/C）"

    if ! echo "$CHIP" | grep -qi "910"; then
        SDMA_SUPPORT="N/A（非 910B/C 平台）"
        log_info "SDMA 支持：N/A（当前平台为 $CHIP，SDMA 仅适用于 910B/C）"
        return 0
    fi

    # 前提1：芯片确认为 910B 或 910C
    log_info "芯片确认为 910B/C 系列 ✓"

    # 前提2：CANN 版本不低于 9.0.0.beta.2
    # 注意：这里检查 CANN >= 9.0.0
    if [ "$CANN_VERSION" != "UNKNOWN" ]; then
        if version_gte "$CANN_VERSION" "9.0.0.beta.2"; then
            log_info "CANN 版本满足 SDMA 要求 ($CANN_VERSION >= 9.0.0.beta.2) ✓"
        else
            SDMA_SUPPORT="否（CANN 版本 $CANN_VERSION 低于 9.0.0.beta.2 要求）"
            log_warn "CANN 版本 $CANN_VERSION 不满足 SDMA 最低要求 (>= 9.0.0.beta.2)"
            return 0
        fi
    else
        log_warn "无法获取 CANN 版本，跳过版本检查"
    fi

    # 前提3：已安装 ops 包（检查 opp 目录下是否有 version 文件）
    local ops_found=false
    local asc_home="${ASCEND_HOME_PATH:-}"
    if [ -n "$asc_home" ] && [ -d "$asc_home/opp" ]; then
        if find "$asc_home/opp" -name "version.info" 2>/dev/null | grep -q .; then
            ops_found=true
        fi
    fi
    if $ops_found; then
        log_info "ops 包已安装（opp/version.info 存在） ✓"
    else
        log_warn "ops 包可能未安装（未在 ${asc_home:-<ASCEND_HOME_PATH>}/opp 下找到 version.info）"
    fi

    # 判定：若 MTE 不支持 → SDMA 不支持
    if echo "$MTE_SUPPORT" | grep -qi "否"; then
        SDMA_SUPPORT="否（MTE 不支持）"
        log_warn "SDMA 支持：否（MTE 不支持，SDMA 依赖 MTE 链路）"
    elif echo "$MTE_SUPPORT" | grep -qi "未知"; then
        SDMA_SUPPORT="不确定（MTE 状态未知）"
        log_warn "SDMA 支持：不确定（MTE 状态未知）"
    else
        SDMA_SUPPORT="是（MTE 支持 ✓，CANN 版本满足 ✓）"
        log_info "SDMA 支持：是"
    fi

    return 0
}

# ============================================================
# 第五步：检测 UDMA（仅 950）
# ============================================================
step5_check_udma() {
    log_step "第五步：检测 UDMA（适用平台：Ascend950）"

    if ! echo "$CHIP" | grep -qi "950"; then
        UDMA_SUPPORT="N/A（非 Ascend950 平台）"
        log_info "UDMA 支持：N/A（当前平台为 $CHIP，UDMA 仅适用于 Ascend950）"
        return 0
    fi

    log_info "芯片确认为 Ascend950 ✓"

    # 条件1：root_info_generate 工具存在
    local rootinfo_bin=""
    if [ -n "${SHMEM_HOME_PATH:-}" ] && [ -f "$SHMEM_HOME_PATH/bin/root_info_generate" ]; then
        rootinfo_bin="$SHMEM_HOME_PATH/bin/root_info_generate"
    elif command -v root_info_generate &>/dev/null; then
        rootinfo_bin="root_info_generate"
    elif [ -n "$PKG_DIR" ] && [ -f "$PKG_DIR/bin/root_info_generate" ]; then
        rootinfo_bin="$PKG_DIR/bin/root_info_generate"
    fi

    if [ -n "$rootinfo_bin" ]; then
        log_info "root_info_generate 工具已找到 ✓"
    else
        log_warn "root_info_generate 工具未找到（需确认出包是否包含此工具）"
    fi

    # 使用 root_info_generate 输出检测 UDMA
    # 层级0（net_layer: 0）若包含 UB（Unified Bus），说明支持 UDMA
    local rootinfo_udma_ok=false
    if [ -n "$rootinfo_bin" ]; then
        # root_info_generate 依赖 libshmem.so（含 topo_addr_info_get_size）
        # 运行时需要 libshmem.so 在 LD_LIBRARY_PATH 中
        local rootinfo_ld_path=""
        if [ -n "$PKG_DIR" ]; then
            # pip wheel 布局：backends/<soc>/libshmem.so，优先匹配检测到的芯片
            local target_soc=""
            if echo "$CHIP" | grep -qi "950"; then
                target_soc="950"
            elif echo "$CHIP" | grep -qi "910"; then
                target_soc="910"
            fi
            if [ -n "$target_soc" ] && [ -f "$PKG_DIR/backends/${target_soc}/libshmem.so" ]; then
                rootinfo_ld_path="$PKG_DIR/backends/${target_soc}"
            else
                local lib_dir
                lib_dir=$(find "$PKG_DIR/backends" -name "libshmem.so" -type f 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)
                [ -n "$lib_dir" ] && rootinfo_ld_path="$lib_dir"
            fi
        fi
        # .run 包布局：lib/libshmem.so
        local lib_flat="$PKG_DIR/lib/libshmem.so"
        if [ -z "$rootinfo_ld_path" ] && [ -f "$lib_flat" ]; then
            rootinfo_ld_path=$(dirname "$lib_flat")
        fi

        # 遍历所有 NPU，只要任意一张卡输出包含 UB 即可
        local max_id=$CARD_COUNT
        if [ "$max_id" -le 0 ] || [ "$max_id" -gt 64 ]; then
            max_id=16
        fi
        # 使用 mktemp 创建唯一临时文件，避免并发进程互相覆盖 stderr，
        # 同时防止 /tmp 下固定文件名被符号链接攻击利用
        local rootinfo_stderr_file
        rootinfo_stderr_file=$(mktemp 2>/dev/null) || rootinfo_stderr_file="/tmp/rootinfo_stderr_$$.$RANDOM.txt"
        trap 'rm -f "$rootinfo_stderr_file"' RETURN
        for phy_id in $(seq 0 $((max_id - 1))); do
            local rootinfo_output
            local rootinfo_stderr
            if [ -n "$rootinfo_ld_path" ]; then
                rootinfo_output=$(LD_LIBRARY_PATH="$rootinfo_ld_path:${LD_LIBRARY_PATH:-}" "$rootinfo_bin" "$phy_id" 2>"$rootinfo_stderr_file") || true
            else
                rootinfo_output=$("$rootinfo_bin" "$phy_id" 2>"$rootinfo_stderr_file") || true
            fi
            rootinfo_stderr=$(cat "$rootinfo_stderr_file" 2>/dev/null || true)
            : > "$rootinfo_stderr_file"  # 清空而不是删除，供下次迭代复用
            if [ -z "$rootinfo_output" ]; then
                if [ -n "$rootinfo_stderr" ]; then
                    log_warn "root_info_generate phyId=${phy_id}：无 stdout 输出，stderr：$rootinfo_stderr"
                fi
                continue
            fi

            # 提取 Rank info 中的 JSON 行（紧跟 "Rank info:" 的那一行）
            local json_line
            json_line=$(echo "$rootinfo_output" | sed -n '/^Rank info:/{n;s/^[[:space:]]*//;p;q}')
            [ -z "$json_line" ] && json_line=$(echo "$rootinfo_output" | grep -m1 -oP '^{.*}$')
            [ -z "$json_line" ] && continue

            # 提取层级0 片段（"net_layer": 0 到下一个非0的 "net_layer": 之间），检查是否含 EID 地址
            local segment
            segment=$(echo "$json_line" | grep -oP '"net_layer":\s*0.*?(?="net_layer":\s*[^0])' || true)
            if [ -n "$segment" ]; then
                if echo "$segment" | grep -q '"addr_type":\s*"EID"'; then
                    rootinfo_udma_ok=true
                    log_info "root_info_generate phyId=${phy_id} 层级0 检测到 UB 组网（EID 地址）✓"
                    break
                fi
            elif echo "$json_line" | grep -qP '"net_layer":\s*0.*"addr_type":\s*"EID"'; then
                rootinfo_udma_ok=true
                log_info "root_info_generate phyId=${phy_id} 层级0 检测到 UB 组网（EID 地址）✓"
                break
            fi
        done
        rm -f "$rootinfo_stderr_file"
        trap - RETURN

        if $rootinfo_udma_ok; then
            log_info "UDMA UB 组网检测通过 ✓"
        else
            log_info "所有 NPU 均未检测到 UB 组网（EID 地址）"
        fi
    fi

    # 条件2：读取包构建元信息中的 UDMA 支持状态
    # UDMA 能力在编译期由 CMake try_compile 确定，写入 VERSION/version.info，比 nm 更可靠
    local pkg_udma_support="否"
    if [ -n "$PKG_DIR" ]; then
        local pkg_version_file=""
        for candidate in \
            "$PKG_DIR/version.info" \
            "$PKG_DIR/shmem/version.info" \
            "$PKG_DIR/VERSION"; do
            if [ -f "$candidate" ]; then
                pkg_version_file="$candidate"
                break
            fi
        done
        if [ -n "$pkg_version_file" ]; then
            local udma_line
            udma_line=$(grep -iE "udma.support|UDMA Support" "$pkg_version_file" | tail -1 || true)
            if echo "$udma_line" | grep -qi "是"; then
                pkg_udma_support="是"
            fi
        fi
    fi
    PKG_UDMA_SUPPORT="$pkg_udma_support"
    if [ "$PKG_UDMA_SUPPORT" = "是" ]; then
        log_info "包构建元信息：UDMA 已编译支持 ✓"
    else
        log_warn "包构建元信息：UDMA 未编译支持（或无法读取）"
    fi

    # 综合判定 UDMA 支持
    if $rootinfo_udma_ok; then
        if [ "$PKG_UDMA_SUPPORT" = "是" ]; then
            UDMA_SUPPORT="是（层级0 UB 组网 ✓，包内 UDMA 已编译 ✓）"
        else
            UDMA_SUPPORT="否（层级0检测到 UB 组网，但包未编译 UDMA 支持）"
            log_warn "根因信息显示 UB 组网，但 SHMEM 包未编译 UDMA 支持"
        fi
    elif [ -n "$rootinfo_bin" ]; then
        # root_info_generate 已执行但未检测到 UB
        if [ "$PKG_UDMA_SUPPORT" = "是" ]; then
            UDMA_SUPPORT="否（层级0 未检测到 UB 组网）"
        else
            UDMA_SUPPORT="否（层级0 未检测到 UB 组网，且包未编译 UDMA 支持）"
        fi
    else
        # root_info_generate 不存在，仅凭包元信息
        if [ "$PKG_UDMA_SUPPORT" = "是" ]; then
            UDMA_SUPPORT="否（包编译了 UDMA，但 root_info_generate 不可用，无法确认 UB 组网）"
            log_warn "root_info_generate 不可用，无法确认 UB 组网，UDMA 不可用"
        else
            UDMA_SUPPORT="否（包未编译 UDMA 支持）"
        fi
    fi

    log_info "UDMA 支持：$UDMA_SUPPORT"
    return 0
}

# ============================================================
# 第六步：检测 RDMA（所有平台）
# ============================================================
step6_check_rdma() {
    log_step "第六步：检测 RDMA（适用平台：所有平台）"

    if echo "$CHIP" | grep -qi "910"; then
        _check_rdma_910b
    elif echo "$CHIP" | grep -qi "950"; then
        _check_rdma_950
    else
        log_info "芯片型号未知，尝试通用 RDMA 检测..."
        _check_rdma_generic
    fi

    return 0
}

_check_rdma_910b() {
    log_info "910B/C 平台 RDMA 检测"

    # 确保 hccn_tool 所在路径在 PATH 中（用户可能未设置环境变量）
    export PATH=/usr/local/Ascend/driver/tools:${PATH:-}
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}

    # 检查 hccn_tool 可用性
    if ! command -v hccn_tool &>/dev/null; then
        RDMA_HCCN_TOOL_OK="否（hccn_tool 不可用）"
        log_warn "hccn_tool 不可用"
        log_warn "参考文档：https://www.hiascend.com/document/detail/zh/mindcluster/2600/toolbox/toolboxug/toolboxug_0142.html#ZH-CN_TOPIC_0000002581610522__section465544911710"
        RDMA_SUPPORT="否（hccn_tool 不可用，无法检测 RDMA）"
        return 0
    fi
    RDMA_HCCN_TOOL_OK="是"

    # 检查 IP 配置
    local max_card=$CARD_COUNT
    if [ "$max_card" -le 0 ] || [ "$max_card" -gt 16 ]; then
        max_card=8
    fi
    local has_ip=false
    for i in $(seq 0 $((max_card - 1))); do
        local ip_output
        ip_output=$(hccn_tool -i $i -ip -g 2>/dev/null) || true
        if [ -n "$ip_output" ] && echo "$ip_output" | grep -qi "ip\|addr"; then
            has_ip=true
            break
        fi
    done

    # 检查网络健康状态
    # hccn_tool -net_health 返回值含义：
    # 0:Success 1:Socket fail 2:Receive timeout 3:Unreachable 4:Time exceeded
    # 5:Fault 6:Init 7:Thread error 8:Detect ip set other:Unknown
    local net_healthy=true
    local net_fault_info=""
    for i in $(seq 0 $((max_card - 1))); do
        local net_output
        net_output=$(hccn_tool -i $i -net_health -g 2>/dev/null) || true
        if [ -n "$net_output" ]; then
            local status
            status=$(echo "$net_output" | grep -oP 'net health status:\s*\K\S.*' | sed 's/[[:space:]]*$//')
            case "${status:-}" in
                Success|0) ;;
                "Socket fail"|1)  net_healthy=false; net_fault_info="NPU$i: Socket fail" ;;
                "Receive timeout"|2) net_healthy=false; net_fault_info="NPU$i: Receive timeout" ;;
                Unreachable|3)  net_healthy=false; net_fault_info="NPU$i: Unreachable" ;;
                "Time exceeded"|4) net_healthy=false; net_fault_info="NPU$i: Time exceeded" ;;
                Fault|5)        net_healthy=false; net_fault_info="NPU$i: Fault" ;;
                Init|6)         net_healthy=false; net_fault_info="NPU$i: Init" ;;
                "Thread error"|7) net_healthy=false; net_fault_info="NPU$i: Thread error" ;;
                "Detect ip set"|8) net_healthy=false; net_fault_info="NPU$i: Detect ip set" ;;
                *)              net_healthy=false; net_fault_info="NPU$i: Unknown ($status)" ;;
            esac
        else
            net_healthy=false
            net_fault_info="NPU$i: 无输出（hccn_tool 可能不可用）"
        fi
        if ! $net_healthy; then
            break
        fi
    done

    if $has_ip && $net_healthy; then
        RDMA_SUPPORT="是（hccn_tool 检测通过）"
        RDMA_NIC_TYPE="DEFAULT"
        log_info "RDMA 检测：IP 配置正常，网络健康 ✓"
    elif $has_ip && ! $net_healthy; then
        RDMA_SUPPORT="部分（IP 配置正常，但 $net_fault_info）"
        RDMA_NIC_TYPE="DEFAULT"
        log_warn "RDMA 检测：IP 配置正常，但 ${net_fault_info}"
        log_warn "请联系网络管理员排查 NPU 间 RoCE 网络连通性"
    else
        RDMA_SUPPORT="否（未检测到 IP 配置）"
        RDMA_NIC_TYPE="DEFAULT"
        log_warn "RDMA 检测：未检测到 IP 配置"
    fi

    # RDMA QoS 环境变量提示
    log_info "请手动执行以下命令设置 RDMA QoS 环境变量："
    echo "  export HCCL_RDMA_TC=132"
    echo "  export HCCL_RDMA_SL=4"
}

_auto_set_ibv_extend_drivers() {
    # $1: so filename, $2: NIC 描述
    local so_name="$1"
    local nic_desc="$2"

    if [ -n "${IBV_EXTEND_DRIVERS:-}" ]; then
        log_info "IBV_EXTEND_DRIVERS 已设置（${nic_desc}）：${IBV_EXTEND_DRIVERS}"
        return
    fi

    local so_path
    so_path=$(ldconfig -p 2>/dev/null | grep -m1 "$so_name" | awk '{print $NF}')
    if [ -z "$so_path" ] || [ ! -f "$so_path" ]; then
        so_path=$(find /usr/lib /usr/lib64 /usr/local/lib /opt -maxdepth 5 -name "$so_name" -type f 2>/dev/null | head -1)
    fi
    if [ -n "$so_path" ] && [ -f "$so_path" ]; then
        log_info "请手动执行以下命令设置 IBV_EXTEND_DRIVERS（${nic_desc}）："
        echo "  export IBV_EXTEND_DRIVERS=${so_path}"
    else
        log_warn "未找到 ${so_name}，请手动设置 IBV_EXTEND_DRIVERS（${nic_desc}）"
        log_warn "提示：使用 ldconfig -p | grep ${so_name} 或 find /usr/lib /usr/lib64 /opt -name ${so_name} 定位后执行 export IBV_EXTEND_DRIVERS=<路径>"
    fi
}

_check_rdma_950() {
    log_info "Ascend950 平台 RDMA 检测"

    if ! command -v ibv_devinfo &>/dev/null; then
        log_warn "ibv_devinfo 不可用，请安装 libibverbs 相关包"
        RDMA_SUPPORT="否（ibv_devinfo 不可用，无法检测 RDMA 网卡）"
        return 0
    fi

    local ibv_output
    ibv_output=$(ibv_devinfo 2>/dev/null) || true

    # 检测云脉网卡（XSCALE）
    if echo "$ibv_output" | grep -qi "xscale"; then
        RDMA_NIC_TYPE="云脉网卡（XSCALE）"
        RDMA_SUPPORT="是（检测到 XSCALE 网卡 ✓）"
        log_info "检测到云脉网卡（XSCALE）"
        _auto_set_ibv_extend_drivers "libxscale_nda.so" "云脉网卡"
    # 检测 1825 网卡
    elif echo "$ibv_output" | grep -qi "hrn"; then
        RDMA_NIC_TYPE="1825 网卡（HRN）"
        RDMA_SUPPORT="是（检测到 1825 网卡 ✓）"
        log_info "检测到 1825 网卡（HRN）"
        _auto_set_ibv_extend_drivers "libhrn5-rdmav34.so" "1825 HRN 网卡"
    else
        RDMA_NIC_TYPE="未识别"
        RDMA_SUPPORT="否（未检测到已知 RDMA 网卡）"
        log_warn "未检测到已知 RDMA 网卡"
    fi

    # RDMA QoS 环境变量提示
    log_info "请手动执行以下命令设置 RDMA QoS 环境变量："
    echo "  export HCCL_RDMA_TC=132"
    echo "  export HCCL_RDMA_SL=4"

    # RDMA 后端 so 自动切换：检测到 1825 时切换到 HNS_1825 内核
    # 检查返回值，cp 失败时更新 RDMA_SUPPORT 反映后端切换失败
    _switch_950_rdma_backend || {
        RDMA_SUPPORT="否（后端切换失败）"
    }
}

_switch_950_rdma_backend() {
    # 仅在指定包目录时执行（import shmem 时由 __init__.py 传入 --package）
    if [ -z "${PKG_DIR:-}" ] || [ ! -d "$PKG_DIR" ]; then
        return 0
    fi

    # 定位 backends 目录（兼容直接安装和 wheel 两种布局）
    local backends_dir=""
    for candidate in \
        "$PKG_DIR/backends" \
        "$PKG_DIR/shmem/backends"; do
        if [ -d "$candidate" ]; then
            backends_dir="$candidate"
            break
        fi
    done

    if [ -z "$backends_dir" ]; then
        return 0
    fi

    local target_950="${backends_dir}/950"
    local src_dir=""

    # 根据 RDMA_NIC_TYPE（由 _check_rdma_950 设置）选择源目录
    case "$RDMA_NIC_TYPE" in
        "1825 网卡（HRN）")
            src_dir="${backends_dir}/950_hns1825"
            ;;
        "云脉网卡（XSCALE）")
            src_dir="${backends_dir}/950_xscale"
            ;;
        *)
            # 未识别的 NIC 类型，保持默认（950 目录下已为 XSCALE）
            return 0
            ;;
    esac

    if [ ! -d "$src_dir" ]; then
        log_warn "RDMA 后端目录不存在：$src_dir（wheel 可能未包含此后端）"
        return 0
    fi

    if [ ! -d "$target_950" ]; then
        log_warn "目标后端目录不存在：$target_950"
        return 0
    fi

    # 检查源目录中是否有 .so 文件
    local so_count
    so_count=$(find "$src_dir" -maxdepth 1 -name "*.so" -type f 2>/dev/null | wc -l)
    if [ "$so_count" -eq 0 ]; then
        log_warn "RDMA 后端目录为空：$src_dir"
        return 0
    fi

    log_info "检测到 ${RDMA_NIC_TYPE}，切换到对应 RDMA 内核后端"
    cp "$src_dir"/*.so "$target_950/" || {
        log_error "RDMA 后端 so 切换失败：cp $src_dir/*.so -> $target_950/"
        return 1
    }
    log_info "RDMA 后端 so 切换完成：$(basename "$src_dir") -> 950/"
}

_detect_950_active_rdma_backend() {
    # 通过对比 backends/950/libshmem.so 与已知变体的 md5，判断当前激活的 RDMA 后端
    local backends_dir=""
    for candidate in \
        "$PKG_DIR/backends" \
        "$PKG_DIR/shmem/backends"; do
        if [ -d "$candidate" ]; then
            backends_dir="$candidate"
            break
        fi
    done

    if [ -z "$backends_dir" ]; then
        PKG_RDMA_BACKEND="未知（未找到 backends 目录）"
        return 0
    fi

    local active_so="${backends_dir}/950/libshmem.so"
    if [ ! -f "$active_so" ]; then
        PKG_RDMA_BACKEND="未知（未找到 libshmem.so）"
        return 0
    fi

    local md5_active
    md5_active=$(md5sum "$active_so" 2>/dev/null | awk '{print $1}') || true

    local md5_xscale
    md5_xscale=$(md5sum "${backends_dir}/950_xscale/libshmem.so" 2>/dev/null | awk '{print $1}') || true

    local md5_hns1825
    md5_hns1825=$(md5sum "${backends_dir}/950_hns1825/libshmem.so" 2>/dev/null | awk '{print $1}') || true

    if [ -n "$md5_active" ] && [ -n "$md5_hns1825" ] && [ "$md5_active" = "$md5_hns1825" ]; then
        PKG_RDMA_BACKEND="HNS_1825"
    elif [ -n "$md5_active" ] && [ -n "$md5_xscale" ] && [ "$md5_active" = "$md5_xscale" ]; then
        PKG_RDMA_BACKEND="XSCALE"
    else
        PKG_RDMA_BACKEND="未知（md5 不匹配）"
    fi
}

_check_rdma_generic() {
    # 确保驱动工具在 PATH 中（用户可能未设置环境变量）
    export PATH=/usr/local/Ascend/driver/tools:${PATH:-}
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}

    if command -v hccn_tool &>/dev/null; then
        _check_rdma_910b
    elif command -v ibv_devinfo &>/dev/null; then
        _check_rdma_950
    else
        RDMA_SUPPORT="否（无可用的 RDMA 检测工具）"
        log_warn "无可用的 RDMA 检测工具（hccn_tool 和 ibv_devinfo 均不可用）"
    fi
}

# ============================================================
# 第七步：包级别检查
# ============================================================
step7_check_package() {
    log_step "第七步：包级别检查"

    if [ -z "$PKG_DIR" ]; then
        log_info "未指定包目录，跳过包级别检查"
        return 0
    fi

    if [ ! -d "$PKG_DIR" ]; then
        log_error "指定的包目录不存在：$PKG_DIR"
        return 1
    fi

    # 读取包的版本/元信息（先读取，用于选择正确的 backend）
    local pkg_version_file=""
    for candidate in \
        "$PKG_DIR/version.info" \
        "$PKG_DIR/shmem/version.info" \
        "$PKG_DIR/VERSION"; do
        if [ -f "$candidate" ]; then
            pkg_version_file="$candidate"
            break
        fi
    done

    if [ -n "$pkg_version_file" ] && [ -f "$pkg_version_file" ]; then
        PKG_CANN_VERSION=$(grep -iE "cann_version|CANN Version" "$pkg_version_file" | awk -F'[=:]' '{print $NF}' | tr -d ' ' || echo "UNKNOWN")
        PKG_RDMA_BACKEND=$(grep -iE "rdma_backend|RDMA Backend" "$pkg_version_file" | awk -F'[=:]' '{print $NF}' | tr -d ' ' || echo "UNKNOWN")
        PKG_SOC_TYPE=$(grep -iE "soc_type|SOC Type" "$pkg_version_file" | awk -F'[=:]' '{print $NF}' | tr -d ' ' || echo "UNKNOWN")
        PKG_UDMA_SUPPORT=$(grep -iE "udma.support|UDMA Support" "$pkg_version_file" | tail -1 | awk -F'[=:]' '{print $NF}' | tr -d ' ' || echo "UNKNOWN")
        if [ -z "$PKG_UDMA_SUPPORT" ] || [ "$PKG_UDMA_SUPPORT" = "UNKNOWN" ]; then
            PKG_UDMA_SUPPORT="否"
        fi
        log_info "包 CANN 版本：$PKG_CANN_VERSION"
        log_info "包 UDMA 支持：$PKG_UDMA_SUPPORT"
    else
        log_warn "未找到包版本信息文件"
    fi

    # 根据检测到的芯片选择 backend 目录（确保多 SOC 时选对 .so）
    local target_backend=""
    if echo "$CHIP" | grep -qi "950"; then
        target_backend="950"
    elif echo "$CHIP" | grep -qi "910"; then
        target_backend="910"
    elif echo "$PKG_SOC_TYPE" | grep -qi "950"; then
        target_backend="950"
    elif echo "$PKG_SOC_TYPE" | grep -qi "910"; then
        target_backend="910"
    else
        target_backend="910"  # 兜底
    fi
    log_info "目标 backend：${target_backend}（检测芯片：$CHIP）"

    # 查找 libshmem.so（优先匹配 backend 目录）
    local libshmem_path=""
    for candidate in \
        "$PKG_DIR/shmem/lib/libshmem.so" \
        "$PKG_DIR/lib/libshmem.so" \
        "$PKG_DIR/backends/${target_backend}/libshmem.so" \
        "$PKG_DIR/backends/910/libshmem.so" \
        "$PKG_DIR/backends/950/libshmem.so"; do
        if [ -f "$candidate" ] && [ "${candidate%.so}" != "$candidate" ]; then
            libshmem_path="$candidate"
            break
        fi
    done
    # 全局查找（仍使用 head -1，但已优先匹配目标 backend）
    if [ -z "$libshmem_path" ]; then
        libshmem_path=$(find "$PKG_DIR" -path "*/backends/${target_backend}/libshmem.so" -type f 2>/dev/null | head -1)
        if [ -z "$libshmem_path" ]; then
            libshmem_path=$(find "$PKG_DIR" -name "libshmem.so" -type f 2>/dev/null | head -1)
        fi
    fi

    if [ -z "$libshmem_path" ] || [ ! -f "$libshmem_path" ]; then
        log_warn "在包目录中未找到 libshmem.so"
    else
        log_info "找到 libshmem.so（${target_backend} backend）：$libshmem_path"

        # 检查传输管理器符号（符号名在 C++ mangling 中为大驼峰）
        local symbols
        symbols=$(nm -D "$libshmem_path" 2>/dev/null) || true

        if grep -q "SdmaTransportManager" <<< "$symbols"; then
            PKG_HAS_SDM_TRANSPORT="是"
            log_info "SdmaTransportManager 符号：已找到 ✓"
        else
            PKG_HAS_SDM_TRANSPORT="否"
            log_warn "SdmaTransportManager 符号：未找到 ✗"
        fi

        if grep -q "RdmaTransportManager" <<< "$symbols"; then
            PKG_HAS_RDM_TRANSPORT="是"
            log_info "RdmaTransportManager 符号：已找到 ✓"
        else
            PKG_HAS_RDM_TRANSPORT="否"
            log_warn "RdmaTransportManager 符号：未找到 ✗"
        fi

        if grep -q "UdmaTransportManager" <<< "$symbols"; then
            PKG_HAS_UDM_TRANSPORT="是"
            log_info "UdmaTransportManager 符号：已找到 ✓"
        elif [ "$target_backend" = "950" ]; then
            PKG_HAS_UDM_TRANSPORT="否"
            log_warn "UdmaTransportManager 符号：未找到 ✗"
        else
            PKG_HAS_UDM_TRANSPORT="N/A（910 backend 不含 UDMA）"
            log_info "UdmaTransportManager 符号：N/A（UDMA 仅 Ascend950 支持）"
        fi
    fi

    # 包 RDMA 后端：910 有 RdmaTransportManager 则为默认，950 通过对比实际 .so 判断
    if [ "$target_backend" = "910" ] && [ "$PKG_HAS_RDM_TRANSPORT" = "是" ]; then
        PKG_RDMA_BACKEND="默认"
    elif [ "$target_backend" = "950" ]; then
        _detect_950_active_rdma_backend
    fi
    log_info "包 ${target_backend} RDMA 后端：${PKG_RDMA_BACKEND}"

    # 检查 root_info_generate 工具
    if [ -f "$PKG_DIR/bin/root_info_generate" ] || find "$PKG_DIR" -name "root_info_generate" -type f 2>/dev/null | grep -q .; then
        PKG_HAS_ROOT_INFO="是"
        log_info "root_info_generate 工具：已包含 ✓"
    else
        PKG_HAS_ROOT_INFO="否"
        log_warn "root_info_generate 工具：未包含 ✗"
    fi

    return 0
}

# ============================================================
# 第八步：结果汇总
# ============================================================
step8_print_summary() {
    # 最终结论在 main() 中输出
    :
}

# ============================================================
# 参数解析
# ============================================================
print_usage() {
    cat <<EOF
用法：$0

通过 shmem-config check 或 import shmem 自动调用。
手动执行将仅输出环境检测结果（不含包级别检查）。
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --package)
                PKG_DIR="$2"
                shift 2
                ;;
            --package=*)
                PKG_DIR="${1#*=}"
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                echo "未知参数：$1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# ============================================================
# 主流程
# ============================================================
main() {
    parse_args "$@"

    echo "======================================================================"
    echo "  SHMEM 安装前环境检测"
    echo "  开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================================"

    # 按序执行检测步骤
    step1_check_chip_platform   || true
    step2_check_version_baseline || true
    step3_check_topology        || true
    step4_check_sdma            || true
    step5_check_udma            || true
    step6_check_rdma            || true
    step7_check_package         || true
    step8_print_summary

    echo ""
    echo -e "\033[34m============================================================\033[0m"
    echo -e "\033[34m  环境检测总结\033[0m"
    echo -e "\033[34m============================================================\033[0m"
    echo "  芯片：${CHIP}，${CARD_COUNT} 卡"
    echo "  MTE  ：${MTE_SUPPORT}"
    echo "  SDMA ：${SDMA_SUPPORT}"
    echo "  UDMA ：${UDMA_SUPPORT}"
    echo "  RDMA ：${RDMA_SUPPORT}"
    echo "  RDMA 后端：${RDMA_NIC_TYPE:-UNKNOWN}"
    echo "  包 RDMA 后端：${PKG_RDMA_BACKEND:-UNKNOWN}"
    echo -e "\033[34m============================================================\033[0m"
}

main "$@"
