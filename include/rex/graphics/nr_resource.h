/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_RESOURCE_H_
#define REX_GRAPHICS_NR_RESOURCE_H_

#include <cstdint>

// [NR-RES] The RESOURCE context: native-renderer build-out increment 4b-3.
//
// 4b-1 closed the INPUT model: the depth-1 buffer stream plus the bin state
// determines every recovery register, diverge 0 at full city load. 4b-2
// closed the SHADER question. What a draw still needs before it can be issued
// natively is its RESOURCES: which vertex buffer, which textures, which
// constants. None of that has ever been measured here, and it is the half
// where the next surprise is expected, so it gets the same treatment the
// recovery set got rather than being assumed.
//
// The resources live in four register files the recovery mirror deliberately
// ignored:
//
//   0x4000 + 2048   ALU constants   512 float4: transforms, material params
//   0x4800 +  192   FETCH constants  32 slots x 6 dwords: vertex AND texture
//                                    fetch (address, size, stride, format)
//   0x4900 +    8   BOOL constants
//   0x4908 +   32   LOOP constants
//
// The fetch file is the interesting one: a Xenos fetch constant carries the
// guest ADDRESS of the buffer, so it is simultaneously the vertex-buffer
// binding, the texture binding and the proof that a replayed draw reads guest
// memory the guest may since have rewritten.
//
// What this unit measures (consumer cvar gpu_nr_res):
//
//   1. GROUND TRUTH -- at buffer entry, compare every defined mirrored
//      resource register against the live RegisterFile, which at that moment
//      holds exactly the state a ring-order replay would have. Same gate that
//      settled 4a and 4b-1: diverge ~0 means the buffer stream alone
//      determines the resource state too. Divergence names the file and the
//      register.
//
//   2. WHERE THE STATE COMES FROM -- inline in the packet (SET_CONSTANT) vs
//      by reference to guest memory (LOAD_ALU_CONSTANT). This is a design
//      input, not bookkeeping: buffers outlive the frame that recorded them,
//      so a by-reference constant must be re-read at REPLAY time, while an
//      inline one is frozen when the buffer is recorded. A high by-reference
//      share means the replay cannot pre-bake constant buffers per buffer.
//
//   3. FETCH-SLOT OCCUPANCY -- how many of the 32 fetch slots are fully
//      defined when a draw executes, and how many distinct buffer addresses
//      they point at. Sizes the binding work per draw and says whether the
//      slots are stable across draws or rewritten constantly.
//
//   4. CARRY -- as in 4a: is the state a draw uses established by its own
//      buffer or inherited from an earlier one? The recovery set turned out
//      to be 78-90% carried in the city; if the resource state is too, the
//      replay's constant handling has to be a running mirror rather than a
//      per-buffer snapshot.
//
// Pure functions over a caller-owned struct: no globals, no threads, no SDK
// dependencies, so tools/nr-resource-test.cpp builds this file bare.

namespace rex {
namespace graphics {
namespace nr {

// The four mirrored files, packed into one dense slot space in this order.
enum ResFile : uint8_t {
  kResFileAlu,
  kResFileFetch,
  kResFileBool,
  kResFileLoop,
  kResFileCount
};

constexpr uint32_t kResAluFirst = 0x4000;
constexpr uint32_t kResAluCount = 2048;  // 512 float4
constexpr uint32_t kResFetchFirst = 0x4800;
constexpr uint32_t kResFetchCount = 192;  // 32 slots x 6 dwords
constexpr uint32_t kResBoolFirst = 0x4900;
constexpr uint32_t kResBoolCount = 8;
constexpr uint32_t kResLoopFirst = 0x4908;
constexpr uint32_t kResLoopCount = 32;
constexpr uint32_t kResRegCount =
    kResAluCount + kResFetchCount + kResBoolCount + kResLoopCount;

// ★ The fetch file is ALIASED (xe_gpu_fetch_group_t): the same 192 dwords are
// either 96 VERTEX fetch constants of 2 dwords or 32 TEXTURE fetch constants
// of 6, and the low 2 bits of a constant's first dword (FetchConstantType,
// 2 = texture, 3 = vertex) say which one a given constant currently is. A
// census that picks one stride and decodes the other's fields reads a
// plausible wrong number, so both views are kept and the type decides.
constexpr uint32_t kResFetchVertexDwords = 2;
constexpr uint32_t kResFetchVertexSlots =
    kResFetchCount / kResFetchVertexDwords;  // 96
constexpr uint32_t kResFetchTextureDwords = 6;
constexpr uint32_t kResFetchTextureSlots =
    kResFetchCount / kResFetchTextureDwords;  // 32

enum ResFetchType : uint8_t {
  kResFetchInvalidTexture = 0,
  kResFetchInvalidVertex = 1,
  kResFetchTexture = 2,
  kResFetchVertex = 3,
};

// reg -> dense slot (-1 outside every file), and the inverses for reporting.
int32_t ResSlot(uint32_t reg);
uint32_t ResSlotReg(uint32_t slot);
ResFile ResSlotFile(uint32_t slot);
const char* ResFileName(ResFile file);

// The persistent resource mirror. Zero-initialize once, then feed it every
// register write the context walk decodes. in_buffer[] means "written by the
// buffer currently being walked" and is cleared by ResBeginBuffer.
struct ResourceContext {
  uint32_t values[kResRegCount];
  uint8_t defined[kResRegCount];
  uint8_t in_buffer[kResRegCount];
  // Per slot: was the value last obtained by reference to guest memory?
  uint8_t from_memory[kResRegCount];

  // Summaries maintained incrementally by ResApplyWrite. A draw observation
  // has to be O(1): the city issues ~200k draws/s, so a per-draw scan of the
  // 2,048-entry ALU file would cost more than the thing being measured and
  // would change the interleaving it is supposed to report.
  uint32_t alu_defined_count;  // ALU slots ever written
  uint8_t alu_any_in_buffer;   // this buffer wrote at least one ALU constant
  // Bit per constant, in each of the two views. "live" means every dword of
  // that constant is defined and its type field claims that view.
  uint32_t vfetch_live_mask[3];      // 96 bits
  uint32_t vfetch_in_buffer_mask[3];
  uint32_t tfetch_live_mask;         // 32 bits
  uint32_t tfetch_in_buffer_mask;
};

struct ResStats {
  uint64_t writes;             // decoded writes landing in a mirrored file
  uint64_t writes_from_memory;  // of those, obtained via LOAD_ALU_CONSTANT
  uint64_t writes_by_file[kResFileCount];
  uint64_t writes_from_memory_by_file[kResFileCount];
  uint64_t draws;               // draws observed by ResObserveDraw
  // Per draw, summed: vertex fetch constants whose 2 dwords are defined AND
  // whose type says vertex, and the same for 6-dword texture constants. The
  // two views overlap in the register file but not in what they claim.
  uint64_t vfetch_live;
  uint64_t vfetch_carried;  // live but not written by this buffer
  uint64_t tfetch_live;
  uint64_t tfetch_carried;
  uint64_t draws_all_alu_carried;  // no ALU constant written by this buffer
  uint64_t alu_defined_at_draw;    // summed over draws
  // Ground truth (ResCompareLive).
  uint64_t checks;
  uint64_t diverge;
  uint64_t diverge_by_file[kResFileCount];
};

// Clears the in-buffer marks. Call once per executed buffer, before walking.
void ResBeginBuffer(ResourceContext* ctx);

// Applies one decoded register write. Ignores registers outside the files.
void ResApplyWrite(ResourceContext* ctx, ResStats* stats, uint32_t reg,
                   uint32_t value, bool from_memory);

// Records the resource state a draw would see. Call per executed draw.
void ResObserveDraw(const ResourceContext* ctx, ResStats* stats);

// One vertex fetch constant, decoded. `valid` is false unless both dwords are
// defined and the type field says vertex.
struct ResVertexFetch {
  bool valid;
  uint32_t base;   // byte address (the stored field counts dwords)
  uint32_t words;  // size field, in words
};
ResVertexFetch ResDecodeVertexFetch(const ResourceContext* ctx, uint32_t slot);

// The raw type field of the constant beginning at a vertex-view slot.
ResFetchType ResFetchTypeAt(const ResourceContext* ctx, uint32_t slot);

// ---- Coverage: does the mirror hold what a SHADER actually reads? -----------
//
// Everything above measures what the buffer stream POPULATES. The question a
// native draw needs answered is narrower and stricter: of the fetch constants
// the draw's own shaders reference (Shader::vertex_bindings /
// texture_bindings, indices [0-95] and [0-31] in the two views), is each one
// recoverable from the stream at the moment the draw runs? A populated file
// with the wrong constants live would satisfy the occupancy counters and
// still bind the wrong buffer.
//
// The failure is split rather than pooled, because the two mean different
// things: kUndefined says the stream never wrote that constant (a hole in the
// recovery model), while kWrongType says it was written but currently claims
// the other view (either the shader reads a stale slot, or the type-field
// reading is wrong -- and confusing that for a hole would send the next
// investigation in the wrong direction).
enum ResCoverage : uint8_t {
  kResCoverLive,
  kResCoverUndefined,
  kResCoverWrongType,
};

ResCoverage ResVertexFetchCoverage(const ResourceContext* ctx, uint32_t slot);
ResCoverage ResTextureFetchCoverage(const ResourceContext* ctx, uint32_t slot);

// The constant's first dword, VERBATIM, in whichever view is asked for. A
// coverage failure has to be reported with the bytes behind it: this project
// has already lost a run to a field read at the wrong stride, and a decoded
// summary would have hidden that exactly as well the second time.
uint32_t ResFetchDword0(const ResourceContext* ctx, uint32_t slot,
                        bool texture_view);

// Reads a live register value (the consumer wraps its RegisterFile).
using ResLiveReadFn = uint32_t (*)(void* user, uint32_t reg);

// One diverging register, for reporting.
struct ResDivergence {
  uint32_t reg;
  uint32_t ours;
  uint32_t live;
};

// Compares every DEFINED mirrored register against the live register file,
// counting divergences into `stats` and writing up to `max_samples` of them.
// Returns the number of samples written.
uint32_t ResCompareLive(const ResourceContext* ctx, ResStats* stats,
                        ResLiveReadFn read, void* user,
                        ResDivergence* samples, uint32_t max_samples);

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_RESOURCE_H_
