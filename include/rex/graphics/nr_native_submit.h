#pragma once
#include <cstdint>

// R2 native submission seam (FULLY-NATIVE-PLAN.md, rung R2, class static_lit).
//
// The guest-side owner of a render object (the game's hook on the retail
// type-5 material callback 82105648) does NOT run the retail body for an owned
// object: no XDK D3D9 calls, no PM4 recording. Instead it fills one
// NrNativeDrawRecord in a ring shared with the command processor and writes a
// three-dword PM4 marker at the exact stream position the retail DRAW_INDX
// group would have occupied:
//
//   dword 0: 0xC0011000 | predicate      type 3, count 2, opcode 0x10 (NOP)
//   dword 1: kNrNativeMarkerMagic
//   dword 2: slot index (low 16 bits) | sequence (high 16 bits)
//
// The command processor executes the record when it reaches the marker, so the
// draw keeps its pass and loop position relative to unowned neighbours and is
// replayed by every band of the predicated tiling exactly as the retail draw
// would be. A marker whose (slot, sequence) does not match the ring is refused
// and counted; it never draws stale data.
//
// Values in the record are HOST byte order. Guest addresses are guest physical
// addresses as IM_LOAD / VGT_DMA_BASE carry them (bits 0..28 valid).
// Producer protocol: fill every field, then store `sequence` last with
// release semantics; the consumer loads it with acquire semantics.
// Reusing a slot: store 0 to `sequence` first, then a release fence, then
// refill. The consumer copies the record and re-reads `sequence` after an
// acquire fence; a changed value is refused as torn (refused_torn).

namespace rex::graphics::nr_native {

constexpr uint32_t kNrNativeMarkerMagic = 0x4A444E52u;  // 'JDNR'
constexpr uint32_t kNrNativeEndMarkerMagic = 0x4A444E45u;  // 'JDNE', closes a retail oracle scope
constexpr uint32_t kNrNativeMarkerHeader = 0xC0011000u;  // NOP, two payload dwords

// One register write in the 0x2000..0x4FFF space (render state, fetch
// constants 0x4800.., bool/loop 0x4900..). Never below 0x2000.
struct NrNativeRegister {
  uint16_t index;
  uint16_t pad;
  uint32_t value;
};

// One float4 ALU constant. `reg` 0..255 = vertex c0..c255, 256..511 = pixel
// c0..c255 (register 0x4000 + 4 * reg).
struct NrNativeConstant {
  uint16_t reg;
  uint16_t pad;
  float v[4];
};

constexpr uint32_t kNrNativeMaxRegisters = 96;
constexpr uint32_t kNrNativeMaxConstants = 512;

enum NrNativeRecordFlags : uint32_t {
  kNrNativeIndexed = 1u << 0,        // DRAW_INDX with a DMA index buffer; else auto-index
  kNrNativeIndex32 = 1u << 1,        // 32-bit indices (else 16-bit)
  kNrNativeIndexSwap16in32 = 1u << 2,  // VGT_DMA_SIZE swap mode bits, see below
  kNrNativeVerifyOnly = 1u << 8,     // producer also ran the retail body: consumer must NOT draw
  kNrNativeRetailFollows = 1u << 9,  // immutable: retail group + matching JDNE follow this marker
};

struct NrNativeDrawRecord {
  uint32_t sequence;        // written last (release); 0 = empty slot
  uint32_t flags;           // NrNativeRecordFlags
  uint32_t object_token;    // producer's census token for correlation (0 = none)
  uint32_t frame;           // producer frame counter when written

  // Shaders, IM_LOAD pointer form: guest physical address and dword count.
  uint32_t vs_address, vs_dwords;
  uint32_t ps_address, ps_dwords;

  // Draw arguments as VGT_DRAW_INITIATOR / VGT_DMA_BASE / VGT_DMA_SIZE carry them.
  uint32_t prim_type;         // xenos::PrimitiveType value
  uint32_t index_count;       // num_indices
  uint32_t index_base;        // guest physical address of the first index (indexed only)
  uint32_t index_endian;      // xenos::Endian value for the index fetch (indexed only)
  uint32_t index_length;      // bytes readable at index_base (indexed only)
  uint32_t reserved0[3];

  uint32_t register_count;
  uint32_t constant_count;
  NrNativeRegister registers[kNrNativeMaxRegisters];
  NrNativeConstant constants[kNrNativeMaxConstants];
};
// The complete VS+PS constant file fits one bounded record: 72-byte header
// + 96 * 8 + 512 * 20 = 11,080 bytes, in one 16 KB ring slot.
constexpr uint32_t kNrNativeSlotSize = 16384;
constexpr uint32_t kNrNativeAbiVersion = 2;
static_assert(sizeof(NrNativeDrawRecord) <= kNrNativeSlotSize, "record must fit one ring slot");
constexpr bool NrNativeLayoutValid(uint32_t abi, uint32_t record_bytes, uint32_t slot_bytes) {
  return abi == kNrNativeAbiVersion && record_bytes == sizeof(NrNativeDrawRecord) &&
         slot_bytes == kNrNativeSlotSize;
}

// Consumer-side outcome counters, one per executed marker. Reported at 1 Hz
// by the command processor as [jade-native-cp] and readable by the producer.
struct NrNativeCounters {
  uint64_t markers;            // markers executed
  uint64_t drawn;              // records that reached the D3D12 draw
  uint64_t verified;           // kNrNativeVerifyOnly records seen (no draw)
  uint64_t refused_sequence;   // marker (slot, sequence) did not match the ring
  uint64_t refused_no_ring;    // marker before the ring was installed
  uint64_t refused_shader;     // LoadShader failed for VS or PS
  uint64_t refused_draw;       // the draw path returned false
  uint64_t refused_predicated; // marker skipped by the bin predicate
  uint64_t registers_applied;
  uint64_t constants_applied;
  // Appended by the SDK consumer (WP1). Every refusal has its own counter.
  uint64_t refused_slot;           // slot index >= installed slot count
  uint64_t refused_torn;           // sequence changed while the record was copied
  uint64_t refused_record;         // register_count / constant_count above the maximum
  uint64_t skipped_band;           // de-tile repeat band: the retail draw would drop too
  uint64_t skipped_viz;            // PA_SC_VIZ_QUERY kill, as the retail draw tail
  uint64_t skipped_register_low;   // register index below 0x2000: never written
  uint64_t skipped_register_high;  // register index at or above 0x5000
  uint64_t skipped_constant_range; // constant reg >= 512
  uint64_t refused_precord;        // precord capture on (also counted in refused_draw)
  // Appended by the v2 fast executor (gpu_native_exec 1 fast, 2 compare).
  // A fast-mode marker is counted in `markers` and in exactly one of the v1
  // outcomes above; fast_draws is the subset of `drawn` the fast path drew.
  uint64_t fast_draws;         // records the fast path drew itself
  uint64_t fast_fallback;      // fast-mode markers run by the v1 body (reasons: [jade-native-fast])
  uint64_t entry_hit;          // object entry reused (no derivation)
  uint64_t entry_miss;         // object entry derived and recorded
  uint64_t pso_hit;            // pipeline without ConfigurePipeline (entry or PSO table)
  uint64_t pso_miss;           // ConfigurePipeline ran
  uint64_t shader_hit;         // shader pair from the table (no LoadShader)
  uint64_t shader_miss;        // LoadShader ran for the pair
  uint64_t tex_hit;            // used texture slots served from a cache
  uint64_t tex_miss;           // used texture slots derived by RequestTextures
  uint64_t rt_update_skipped;  // render target update entry guard held (no body run)
  uint64_t cbv_bytes;          // constant buffer bytes the fast draws uploaded
  uint64_t fast_ns;            // CP nanoseconds inside the fast executor
  uint64_t v1_ns;              // CP nanoseconds inside the v1 executor (timed markers)
  uint64_t v1_draws;           // draws made by the timed v1 markers
  uint64_t cmp_markers;        // compare-mode markers compared
  uint64_t cmp_eq;             // compare classes equal (all classes; per class: [jade-native-cmp])
  uint64_t cmp_ne;             // compare classes different
};

// ---- Pure parse rules (header-only, no D3D12; tools/test-nr-native-marker.py) ----

constexpr uint32_t kNrNativeRegisterFloor = 0x2000;   // never write below
constexpr uint32_t kNrNativeRegisterCeiling = 0x5000; // exclusive
constexpr uint32_t kNrNativeConstantBase = 0x4000;    // XE_GPU_REG_SHADER_CONSTANT_000_X
constexpr uint32_t kNrNativeConstantCount = 512;

// Packet header of a type-3 NOP with exactly two payload dwords, any predicate.
constexpr bool NrNativeIsMarkerHeader(uint32_t header) {
  return (header & ~1u) == kNrNativeMarkerHeader;
}
// Full marker test: header shape plus the first payload dword.
constexpr bool NrNativeIsMarker(uint32_t header, uint32_t dword1) {
  return NrNativeIsMarkerHeader(header) && dword1 == kNrNativeMarkerMagic;
}
constexpr bool NrNativeIsEndMarker(uint32_t header, uint32_t dword1) {
  return NrNativeIsMarkerHeader(header) && dword1 == kNrNativeEndMarkerMagic;
}
constexpr bool NrNativeIsScopeMarker(uint32_t header, uint32_t dword1) {
  return NrNativeIsMarker(header, dword1) || NrNativeIsEndMarker(header, dword1);
}
enum class NrNativeRetailEndVerdict { kWrongEnd, kNoRetail, kMultiple, kUnavailable, kPaired };
// The exact comparison gate, shared with the standalone protocol tests.
constexpr NrNativeRetailEndVerdict NrNativeCheckRetailEnd(
    uint32_t expected_ref, uint32_t actual_ref, uint32_t retail_draws,
    bool native_observed, bool retail_observed) {
  if (expected_ref != actual_ref) return NrNativeRetailEndVerdict::kWrongEnd;
  if (!retail_draws) return NrNativeRetailEndVerdict::kNoRetail;
  if (retail_draws != 1) return NrNativeRetailEndVerdict::kMultiple;
  if (!native_observed || !retail_observed) return NrNativeRetailEndVerdict::kUnavailable;
  return NrNativeRetailEndVerdict::kPaired;
}
// The executor's view: type-3 opcode 0x10 with count 2 and payload dword 0.
constexpr bool NrNativeIsMarkerPayload(uint32_t count, uint32_t dword0) {
  return count == 2 && dword0 == kNrNativeMarkerMagic;
}
constexpr uint32_t NrNativeMarkerSlot(uint32_t slot_sequence) { return slot_sequence & 0xFFFFu; }
constexpr uint32_t NrNativeMarkerSequence(uint32_t slot_sequence) { return slot_sequence >> 16; }
constexpr uint32_t NrNativeMarkerRef(uint32_t slot, uint32_t sequence) {
  return (slot & 0xFFFFu) | ((sequence & 0xFFFFu) << 16);
}

enum class NrNativeVerdict : uint32_t {
  kOk = 0,
  kNoRing,     // no ring installed (or zero slots)
  kSlotRange,  // slot >= slots
  kSequence,   // record sequence 0 (empty) or low 16 bits differ from the marker
};

// `record_sequence` is the acquire-loaded full 32-bit sequence of the slot.
// An empty slot (0) never matches, even for a marker sequence of 0.
constexpr NrNativeVerdict NrNativeCheck(bool ring_installed, uint32_t slots,
                                        uint32_t slot_sequence, uint32_t record_sequence) {
  if (!ring_installed || !slots) return NrNativeVerdict::kNoRing;
  if (NrNativeMarkerSlot(slot_sequence) >= slots) return NrNativeVerdict::kSlotRange;
  if (!record_sequence ||
      (record_sequence & 0xFFFFu) != NrNativeMarkerSequence(slot_sequence)) {
    return NrNativeVerdict::kSequence;
  }
  return NrNativeVerdict::kOk;
}

constexpr bool NrNativeRecordBoundsOk(uint32_t register_count, uint32_t constant_count) {
  return register_count <= kNrNativeMaxRegisters && constant_count <= kNrNativeMaxConstants;
}
constexpr bool NrNativeRegisterWritable(uint32_t index) {
  return index >= kNrNativeRegisterFloor && index < kNrNativeRegisterCeiling;
}
// First of the four dwords of float constant `reg` (0..511), or 0 when out of range.
constexpr uint32_t NrNativeConstantRegister(uint32_t reg) {
  return reg < kNrNativeConstantCount ? kNrNativeConstantBase + 4 * reg : 0;
}

// ---- SDK consumer side (src/graphics/nr_native_submit.cpp), CP thread only ----

// Copies the record named by `slot_sequence` into `out` (header, then the used
// register/constant entries), re-checking the sequence after the copy. Out
// arrays beyond the used counts are left untouched. kOk only when the copy is
// consistent; a changed sequence returns kSequence and sets *torn.
NrNativeVerdict NrNativeAcquire(uint32_t slot_sequence, NrNativeDrawRecord* out, bool* torn);
// Executor snapshot: retain the coherent header and sequence check, but leave
// payload arrays untouched for verify-only records or a rejected draw band.
// The band predicate must be pure; draw-dispatch effects still run at the
// executor's original gate after record validation and precord refusal.
NrNativeVerdict NrNativeAcquireForDraw(uint32_t slot_sequence, NrNativeDrawRecord* out,
                                     bool* torn, bool band_allowed);
// The consumer counters (incremented on the CP thread through NrNativeBump).
NrNativeCounters& NrNativeCpCounters();
void NrNativeBump(uint64_t NrNativeCounters::*field, uint64_t n = 1);
// Installed phase for the swap marker, -1 none.
int NrNativePhase();
// 1 Hz [jade-native-cp] line, CP thread; logs only when markers changed.
void NrNativeReport1Hz();

}  // namespace rex::graphics::nr_native

// C ABI for the producer (game project). Install once before guest startup.
// `ring` holds `slots` records at kNrNativeSlotSize byte stride.
// A mismatched producer must keep the retail hook active. Invalid installs
// clear the SDK consumer so an incompatible allocation is never read.
extern "C" uint32_t rex_nr_native_install_checked(void* ring, uint32_t slots, uint32_t abi,
                                                 uint32_t record_bytes, uint32_t slot_bytes);
// Legacy ABI cannot communicate its record layout and now refuses non-null
// installs. Retained to fail explicitly instead of reading old 8 KB slots.
extern "C" void rex_nr_native_install(void* ring, uint32_t slots);
// Phase shown at the top left of the game window at every swap:
// -1 none, 0 red (off), 1 green (verify), 2 blue (native), 3 yellow.
extern "C" void rex_nr_native_phase(int phase);
// Snapshot of the consumer counters (host thread safe).
extern "C" void rex_nr_native_counters(rex::graphics::nr_native::NrNativeCounters* out);
