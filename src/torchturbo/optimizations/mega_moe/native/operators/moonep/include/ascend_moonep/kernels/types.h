#pragma once

#include <cstdint>

namespace ascend_moonep {

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MOONEP_DEVICE_INLINE __forceinline__ __aicore__
#else
#define MOONEP_DEVICE_INLINE static inline constexpr
#endif

// Default host/device launch width for the other operators. PUT UDMA combine
// uses its own 64-AIV schedule in moonep_combine_udma.cpp; do not change this
// shared default when tuning combine.
inline constexpr uint32_t kMoonEPAivNum = 32;
// Server dedup groups 8 globally consecutive ranks into one board.
inline constexpr uint32_t kMoonEPDispatchRanksPerServer = 8;
inline constexpr uint32_t kMoonEPUdmaCombineQpCount = 2;

// Server-dedup planning keeps the public dup_groups tensor at [NvS, 3].  The
// third word packs the source-rank-local route index above the duplicate count.
// K is capped at 16 by combine, so five low bits are sufficient for dup_n and
// leave 26 bits for a route index (well above the advertised 8M-token limit).
inline constexpr uint32_t kMoonEPServerDupCountBits = 5;
inline constexpr uint32_t kMoonEPServerDupCountMask =
    (1U << kMoonEPServerDupCountBits) - 1U;
inline constexpr uint32_t kMoonEPServerDupRouteLimit =
    1U << (31U - kMoonEPServerDupCountBits);

MOONEP_DEVICE_INLINE int32_t PackMoonEPServerDupRoute(
    uint32_t source_route, uint32_t duplicate_count) {
    return static_cast<int32_t>(
        (source_route << kMoonEPServerDupCountBits) |
        (duplicate_count & kMoonEPServerDupCountMask));
}

MOONEP_DEVICE_INLINE uint32_t UnpackMoonEPServerDupCount(int32_t packed) {
    return static_cast<uint32_t>(packed) & kMoonEPServerDupCountMask;
}

MOONEP_DEVICE_INLINE uint32_t UnpackMoonEPServerDupSourceRoute(int32_t packed) {
    return static_cast<uint32_t>(packed) >> kMoonEPServerDupCountBits;
}

// dst uses three states in server mode.  Existing positive values are normal
// canonical routes, ~raw remains a secondary duplicate, and 0x80000000|raw is
// a canonical route whose reduced payload is pushed into source-rank staging.
// The latter two are disjoint while R*NvS < 2^30.
inline constexpr uint32_t kMoonEPServerForwardedRouteTag = 0x80000000U;
inline constexpr uint32_t kMoonEPServerRoutePayloadMask = 0x3fffffffU;

MOONEP_DEVICE_INLINE int32_t EncodeMoonEPServerForwardedRoute(uint32_t raw) {
    return static_cast<int32_t>(kMoonEPServerForwardedRouteTag | raw);
}

MOONEP_DEVICE_INLINE bool IsMoonEPServerForwardedRoute(int32_t encoded) {
    return (static_cast<uint32_t>(encoded) & 0xc0000000U) ==
        kMoonEPServerForwardedRouteTag;
}

MOONEP_DEVICE_INLINE uint32_t DecodeMoonEPServerForwardedRoute(int32_t encoded) {
    return static_cast<uint32_t>(encoded) & kMoonEPServerRoutePayloadMask;
}

// Legacy GET scratch constants retained for compatibility. The current PUT
// workspace ABI is defined by MakeCombinePutLayout in types.h.
inline constexpr uint32_t kMoonEPUdmaCombineScratchTileElements = 8192;
inline constexpr uint32_t kMoonEPUdmaCombineScratchTilesPerAiv = 2;

#undef MOONEP_DEVICE_INLINE

struct alignas(8) MoonEPCombineTilingData {
    uint32_t rank;              // current rank
    uint32_t world_size;        // EP size
    uint32_t num_tokens;        // S
    uint32_t hidden;            // H
    uint32_t num_topk;          // K
    uint32_t block_dim;         // fixed 32 AIV (host always writes 32)
    uint32_t with_weight;       // 1: gather weights → [S,K]; 0: skip
    uint32_t reserved;
};

struct alignas(8) MoonEPUdmaCombineTilingData {
    uint32_t S;
    uint32_t H;
    uint32_t K;
    uint32_t R;
    uint32_t rank;
    uint32_t with_weights;
    uint32_t server_dedup;
    uint32_t reserved; // Preserve the 32-byte host/device tiling structure size.
};

struct alignas(8) MoonEPDispatchTilingData {
    uint32_t S;
    uint32_t H;
    uint32_t K;
    uint32_t R;
    uint32_t rank;
    uint32_t with_weights;
    uint32_t server_dedup;
};

// Udma dispatch profile ABI (mirrors the UBMEM template-<bool Profile> pattern
// from the timing branch): moonep_dispatch_udma_profile_kernel writes each AIV's
// metrics as an int64 row into a [kMoonEPAivNum, kMoonEPDispatchUdmaProfileFields]
// GM buffer (profile_cycles_gm); the host reads the buffer back instead of
// kernel printf. Plain (non-profile) kernel passes a dummy pointer that is
// never dereferenced when Profile=false.
inline constexpr uint32_t kMoonEPDispatchUdmaProfileFields = 9;

enum MoonEPDispatchUdmaProfileField : uint32_t {
    kDispatchUdmaProfileTotal = 0,      // full operator wall cycles (incl. barriers)
    kDispatchUdmaProfileTokenScatter,   // token scatter stage cycles
    kDispatchUdmaProfileTokenWqeSubmit, // accumulated WQE fill/submit cycles
    kDispatchUdmaProfileTokenSync,      // token scatter completion sync cycles
    kDispatchUdmaProfileTokenBytes,     // bytes actually handled by this AIV (token)
    kDispatchUdmaProfileWqePackets,     // WQE packets issued by this AIV
    kDispatchUdmaProfileWeightScatter,  // weight scatter stage cycles
    kDispatchUdmaProfileWeightBytes,    // weight bytes scattered by this AIV
    kDispatchUdmaProfileDuplicate,      // duplicate expansion stage cycles
};

struct alignas(8) PlanningTilingData {
    uint32_t rank;
    uint32_t world_size;
    uint32_t num_tokens;
    uint32_t num_topk;
    uint32_t num_experts;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_prefetch_slots;
    uint32_t block_dim;
    uint32_t symmetric_tpe_offset;
    uint32_t symmetric_dup_rank_counts_offset;
    uint32_t source_dup_tokens_per_tile;
    uint32_t owner_reduce_tile_elems;
    uint32_t server_dedup;
    uint32_t combine_mode; // 0: base only; 1: fused tail; 2: precomputed-dst only.
    uint32_t combine_h;
    uint32_t combine_reserved;
    uint64_t combine_workspace;
    uint64_t combine_plan;
};

static_assert(sizeof(PlanningTilingData) == 80,
              "PlanningTilingData fused ABI must remain 80 bytes");

}  // namespace ascend_moonep

// Shared combine ABI: launch/batch sizes, workspace/plan layouts and trace capacity.
// Used by host launchers, planning.cpp and moonep_combine_udma.cpp.

// Shared launch/workspace capacities, also consumed by host and planning.
namespace ascend_moonep {
inline constexpr uint32_t kCombinePutCores = 64;
inline constexpr uint32_t kCombinePutBatchRows = 64;
} // namespace ascend_moonep

// ---- Communication workspace layout ----
namespace ascend_moonep {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MOONEP_PUT_INLINE __forceinline__ __aicore__
#else
#define MOONEP_PUT_INLINE inline constexpr
#endif

// Aligned PUT workspace ABI. All uint64 members except bytes are byte offsets
// relative to symmetric ws, NOT pointers or lengths. Host and Python mirror it.
// N=S*K, H=hidden, R=ranks, B=ceil(N/64), A(x)=ceil(x/512)*512.
// Meta(base,i) uses independent 64-byte records; Flag(base,i) uses 512-byte slots.
// Only the first uint64 of a flag carries magic; reducer MTE3 writes 32 bytes.
//
// Source=original token owner/final destination; producer=expert-output holder;
// ingress=server group reducer. The same rank can have all three roles.
// Earlier payload/metadata offsets remain stable, but flag/extension offsets
// differ from checkpoint f5fd3b0: host, Python and kernel MUST be installed together.
struct CombinePutLayout {
    uint64_t final_hidden; // BF16[N,H], compact producer segments at each source.
                           // Local-server routes include suppressed members too.
    uint64_t send_hidden;  // BF16[N,H], legacy payload arena. Aligned path reuses
                           // its first N 64-byte slots as direct_send:
                           // {tx loff, tx ordinal}; H>=32 keeps ABI unchanged.
    uint64_t relay_hidden; // BF16[8*N,H], source-major contiguous contributor groups:
                           // [representative,duplicate0,...]. Reducer overwrites
                           // each group head with its BF16 sum; sender reads there.
    uint64_t weights;      // float[N], every original source route, no weight dedup.

    uint64_t reverse;     // Retired reverse-route records; offset reserved for ABI stability.
    uint64_t reverse_send;// Reserved N*64B arena; planner publishes contiguous dst[N]
                           // in its first N*4B. Peers read it, never PUT reverse requests.
    uint64_t final_map;    // Retired scalar final mapping arena, offset reserved.
                           // Planner now builds final descriptors by partition prefixes.
    uint64_t group_of_loff;// Retired group lookup arena; offset reserved.
    uint64_t tx_rows;      // N records in source/ordinal order:
                           // {loff,group,scratchHead(-1 for DIRECT),contributorCount}.
                           // DIRECT count=1, no reducer/copy to send_hidden.
    uint64_t source_counts;// R records:
                           // [0]=active result count, [1]=remote final segment base,
                           // [2]=local tx base, [3]=DIRECT count,
                           // [4]=scratch range start, [5]=scratch slot count,
                           // [6]=direct_send compact range start.
                           // Scratch slots INCLUDE contributor holes; <=N/source.
    uint64_t counts_send;  // Planner-local R records: {groupCount,groupPrefix} per source.
    uint64_t contributor; // 8*N records: word0 is {count:high32,txIndex:low32},
                           // word1 is original global contributor address.
                           // Only a group head has count>0; interior slots are zero.
                           // 64-byte spacing prevents concurrent planner cacheline
                           // writeback races. Reducer gathers 64 entries into UB.
    uint64_t relay_send;   // 8*N requests at producer: [ingressLocal*N+ordinal]
                           // -> {input loff,remote scratch slot}. Includes ingress
                           // self contributors for planning; payload skips them because
                           // Reduce reads their original input directly.
    uint64_t relay_publish;// Retired relay request PUT arena; offset reserved.
    uint64_t relay_counts, relay_counts_send; // Legacy reserved 8-record tables.

    uint64_t relay_ready; // 8*R slots, producerLocal*R+source, source-level SO ready.
                           // Empty producer streams MUST publish as well.
    uint64_t reduce_ready;// R*B slots, source*B+scratchBatch; MTE3-ordered after sum.
                           // B uses N, NOT N/R: nonuniform input must fit safely.
    uint64_t final_ready; // 2*R slots, producer*2+QP lane. Both remote lanes close
                           // DIRECT+RELAY even when empty. Local Phase A uses lane0.
    uint64_t weight_ready;// R slots, producer -> complete all original route weights.
    uint64_t magic;       // 64 stable per-core SHMEM PUT-source flag slots.
    uint64_t relay_ranges;// 8*R records at producer: ingressLocal*R+source
                           // -> {requestBegin,requestCount}.
    uint64_t relay_ranges_send; // Planner-local 8*R stats records:
                           // {scratchCount,requestCount,scratchPrefix,requestPrefix}.
                           // After metadata completion, reused as R*512B final
                           // histograms: 64 partition counts/prefixes + peer total.
    uint64_t source_input_ready;// R slots, shard1 waits all producers, wakes shard0.
    uint64_t final_gate;        // One local slot: core0 publishes after it has
                                // observed every final_ready; all other cores wait here.
    uint64_t final_routes;      // N independent 128-byte source-final descriptors.
                                // Each output token owns {count, route[16]}; route is either
                                // local hidden loff (bit31 set) or final_hidden row. N records
                                // reserve for K=1 too; the active prefix is N/K records.
    uint64_t bytes;       // Total bytes including all alignment/reserved arenas.
    uint32_t batches;     // ceil(N/64), worst per-source scratch batch capacity.
};

MOONEP_PUT_INLINE uint64_t CombinePutAlign(uint64_t bytes) {
    return (bytes + 511U) & ~uint64_t(511U);
}
MOONEP_PUT_INLINE CombinePutLayout MakeCombinePutLayout(
    uint32_t n, uint32_t h, uint32_t ranks) {
    CombinePutLayout p{};
    const uint64_t row_bytes = uint64_t(n) * h * 2U; // all N rows, not one row
    const uint64_t meta = uint64_t(n) * 64U;
    p.batches = (uint64_t(n) + 63U) / 64U;
    p.final_hidden = 0;
    p.send_hidden = CombinePutAlign(row_bytes);
    p.relay_hidden = p.send_hidden + CombinePutAlign(row_bytes);
    p.weights = p.relay_hidden + CombinePutAlign(8U * row_bytes);
    p.reverse = p.weights + CombinePutAlign(uint64_t(n) * 4U);
    p.reverse_send = p.reverse + CombinePutAlign(meta);
    p.final_map = p.reverse_send + CombinePutAlign(meta);
    p.group_of_loff = p.final_map + CombinePutAlign(meta);
    p.tx_rows = p.group_of_loff + CombinePutAlign(meta);
    p.source_counts = p.tx_rows + CombinePutAlign(meta);
    p.counts_send = p.source_counts + CombinePutAlign(uint64_t(ranks) * 64U);
    p.contributor = p.counts_send + CombinePutAlign(uint64_t(ranks) * 64U);
    p.relay_send = p.contributor + CombinePutAlign(8U * meta);
    p.relay_publish = p.relay_send + CombinePutAlign(8U * meta);
    p.relay_counts = p.relay_publish + CombinePutAlign(8U * meta);
    p.relay_counts_send = p.relay_counts + 512U;
    p.relay_ready = p.relay_counts_send + 512U;
    p.reduce_ready = p.relay_ready + uint64_t(8U) * ranks * 512U;
    p.final_ready = p.reduce_ready + uint64_t(ranks) * p.batches * 512U;
    p.weight_ready = p.final_ready + uint64_t(ranks) * 2U * 512U;
    p.magic = p.weight_ready + uint64_t(ranks) * 512U;
    p.relay_ranges = p.magic + uint64_t(kCombinePutCores) * 512U;
    p.relay_ranges_send = p.relay_ranges + CombinePutAlign(uint64_t(8U) * ranks * 64U);
    p.source_input_ready = p.relay_ranges_send + CombinePutAlign(uint64_t(8U) * ranks * 64U);
    p.final_gate = p.source_input_ready + uint64_t(ranks) * 512U;
    p.final_routes = p.final_gate + 512U;
    p.bytes = p.final_routes + CombinePutAlign(uint64_t(n) * 128U);
    return p;
}
#undef MOONEP_PUT_INLINE
} // namespace ascend_moonep

// ---- Immutable execution-plan layout ----
namespace ascend_moonep {
#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define MOONEP_PLAN_INLINE __forceinline__ __aicore__
#else
#define MOONEP_PLAN_INLINE inline constexpr
#endif
// Rank-local immutable snapshot. No payload, ready flags, magic or QP state.
// Offsets are relative to this tensor, not the symmetric communication workspace.
// v5: planner writes directly here. TX/contributor/relay records are packed;
// scalar-written counts/ranges and final descriptors retain cacheline isolation.
// Regenerate plans and deploy host/device together; v4 snapshots are incompatible.
inline constexpr uint64_t kCombineExecutionPlanVersion = 0x4d45504300000005ULL;
inline constexpr uint32_t kCombinePlanTxBytes = 16U;
inline constexpr uint32_t kCombinePlanContributorBytes = 16U;
inline constexpr uint32_t kCombinePlanRelayBytes = 8U;
struct CombineExecutionPlanLayout {
    uint64_t tx_rows, source_counts, contributor, relay_send, relay_ranges;
    uint64_t final_routes, bytes;
};
MOONEP_PLAN_INLINE CombineExecutionPlanLayout MakeCombineExecutionPlanLayout(
    uint32_t n, uint32_t ranks) {
    CombineExecutionPlanLayout q{};
    q.tx_rows = 512U; // Header: version, N,H,K,R,rank,server_dedup,reserved.
    q.source_counts = q.tx_rows + CombinePutAlign(uint64_t(n) * kCombinePlanTxBytes);
    q.contributor = q.source_counts + CombinePutAlign(uint64_t(ranks) * 64U);
    q.relay_send = q.contributor + CombinePutAlign(uint64_t(n) * 8U * kCombinePlanContributorBytes);
    q.relay_ranges = q.relay_send + CombinePutAlign(uint64_t(n) * 8U * kCombinePlanRelayBytes);
    q.final_routes = q.relay_ranges + CombinePutAlign(uint64_t(ranks) * 8U * 64U);
    // Weights read the existing dst directly; no dedicated weights table.
    q.bytes = q.final_routes + CombinePutAlign(uint64_t(n) * 128U);
    return q;
}
#undef MOONEP_PLAN_INLINE
} // namespace ascend_moonep
