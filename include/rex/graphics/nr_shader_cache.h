/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_SHADER_CACHE_H_
#define REX_GRAPHICS_NR_SHADER_CACHE_H_

#include <cstdint>
#include <vector>

// [NR-SC] THE SHADER CACHE: native-renderer phase 5, increment 5-2.
//
// A native draw needs two host shader blobs. Increment 5-1 mirrored the state
// half of a pipeline (topology, cull, depth, blend, masks) and proved it byte
// for byte at city load. This is the other half: given the ucode a draw's
// registers point at, produce the DXBC the PSO is built from, and produce it
// once.
//
// What is OURS here is the CACHE, not the compiler. The translation itself is
// the SDK's DxbcShaderTranslator -- the same one increment 4b-2's offline
// census ran over all 3,320 shaders in the retail xeshader.sdb with 0 failures
// -- reached through a caller-supplied callback. What this unit owns is
// everything the emulator's PipelineCache does around it and that a native
// renderer must do for itself:
//
//   - the KEY. A shader is not identified by its ucode alone. The DXBC depends
//     on the translator's `modification`, which is per-draw register state
//     (interpolator mask, param-gen, user clip planes, point size, the
//     depth-stencil mode). The cache is therefore keyed on the triple
//     {stage, ucode hash, modification}, and the number of modifications a
//     single ucode is seen under is counted, because that number is what
//     decides whether an ahead-of-time corpus translation is possible at all.
//   - the LIFETIME. Entries never move (the storage is reserved once), so a
//     pointer handed to a draw stays valid, and a full cache refuses cleanly
//     instead of evicting something a draw is holding.
//   - the NEGATIVE result. A ucode that fails to translate is remembered as
//     failed, so a draw that cannot be issued natively costs one lookup, not
//     one translation attempt per frame forever.
//
// THE GATE: for each distinct key, our DXBC is byte-compared once against the
// binary the emulated PipelineCache produced for the same translation, and any
// disagreement is named (offset and both bytes). Our Shader object is built
// from the ucode dwords alone -- none of the emulator's shader bookkeeping --
// so byte equality is the proof that the native renderer can translate from
// what the walk recovers and nothing else. `bytes_ne` must be 0.
//
// The other deliverable is the hit rate at city load: how often a draw's key
// is already translated. That is the number that says whether a runtime cache
// is enough (4b-2 measured 366 of 3,320 ucodes touched in two minutes of city,
// so it should be), and the translate latency counters say how big the first
// -sighting hitch is.
//
// No SDK dependencies and no threads: the translator arrives as a callback, so
// tools/nr-shader-cache-test.cpp builds this file bare with a fake one. Single
// -threaded by contract, like every other nr probe -- the caller is the
// command-processor thread.

namespace rex {
namespace graphics {
namespace nr {

enum : uint32_t {
  kNrShaderStageVertex = 0,
  kNrShaderStagePixel = 1,
  kNrShaderStageCount = 2,
};

// One translated shader. Owned by the cache; the pointer is stable for the
// lifetime of the cache (until NrShaderCacheReset).
struct NrShaderEntry {
  uint64_t ucode_hash = 0;
  uint64_t modification = 0;
  uint32_t stage = kNrShaderStageVertex;
  // False when the translator refused this ucode. The entry still exists, so
  // the refusal is remembered rather than retried every draw; a renderer must
  // fall back to the emulated path for it.
  bool valid = false;
  // Whether this key has been compared against the emulated pipeline cache's
  // binary yet, and the result. Checked once per key, not once per draw: the
  // comparison is a memcmp over kilobytes and the draw path is not the place
  // for it.
  bool verified = false;
  bool agreed = false;
  std::vector<uint8_t> dxbc;
};

// Translate one ucode blob. `ucode_dwords` are in HOST byte order and
// `ucode_hash` is the runtime's own key for them (XXH3-64 over the raw
// big-endian guest bytes). Returns false if the shader cannot be translated.
using NrShaderTranslateFn = bool (*)(void* ctx, uint32_t stage, uint64_t ucode_hash,
                                     const uint32_t* ucode_dwords, uint32_t ucode_dword_count,
                                     uint64_t modification, std::vector<uint8_t>* dxbc_out);

struct NrShaderCacheStats {
  // ---- window counters (cleared by NrShaderCacheEndWindow) ----
  uint64_t lookups = 0;
  uint64_t hits = 0;
  uint64_t misses = 0;    // ... which each cost one translation attempt
  uint64_t refused = 0;   // no room left: neither served nor translated
  uint64_t pending = 0;   // key not comparable yet (their translation deferred)
  uint64_t invalid_hits = 0;  // served a known-untranslatable key

  // ---- cumulative: the sets and the verdicts ----
  uint64_t translations = 0;
  uint64_t translate_fail = 0;
  uint64_t translate_ns_total = 0;
  uint64_t translate_ns_max = 0;
  uint64_t translate_ns_max_hash = 0;
  uint64_t dxbc_bytes = 0;

  uint32_t entries = 0;  // distinct {stage, ucode, modification} keys
  uint32_t ucodes = 0;   // distinct ucode hashes
  uint32_t vs_entries = 0, ps_entries = 0;
  uint32_t vs_ucodes = 0, ps_ucodes = 0;
  // The AOT question, measured: the most modifications any single ucode has
  // been seen under. If this is 1 for pixel shaders, an offline corpus
  // translation can be keyed on ucode; if it is not, the modification is
  // per-draw state and an offline translation would be translating the wrong
  // variant.
  uint32_t mods_max = 0;
  uint64_t mods_max_hash = 0;

  // ---- the gate ----
  uint64_t verified = 0;
  uint64_t agreed = 0;
  uint64_t size_ne = 0;
  uint64_t bytes_ne = 0;
  uint64_t valid_ne = 0;  // they translated it, we did not
  // First disagreement, kept for the whole session so it is still named at the
  // end of a run rather than scrolling away with its window.
  bool have_first_ne = false;
  uint64_t first_ne_hash = 0;
  uint64_t first_ne_modification = 0;
  uint32_t first_ne_stage = 0;
  uint64_t first_ne_offset = 0;
  uint32_t first_ne_ours = 0;
  uint32_t first_ne_theirs = 0;
  uint64_t first_ne_ours_size = 0;
  uint64_t first_ne_theirs_size = 0;

  // ---- instrument health ----
  uint64_t probe_ovf = 0;   // a key found no free slot within the probe limit
  uint64_t byte_refused = 0;  // a translation that would exceed the byte budget
};

// Installs the translator and sizes the cache. Safe to call repeatedly with
// the same arguments; changing the limits resets the cache.
void NrShaderCacheConfigure(NrShaderTranslateFn fn, void* ctx, uint32_t max_entries,
                            uint64_t max_bytes);
bool NrShaderCacheConfigured();

// Drops every entry and zeroes every counter, keeping the configuration.
void NrShaderCacheReset();

// The renderer's whole shader interface: give me the DXBC for this draw's
// shader. Translates on first sighting, serves the cached blob afterwards.
// Returns nullptr only when the cache is full (counted as `refused`); an entry
// with valid == false means the translator refused this ucode.
NrShaderEntry* NrShaderCacheLookup(uint32_t stage, uint64_t ucode_hash,
                                   const uint32_t* ucode_dwords, uint32_t ucode_dword_count,
                                   uint64_t modification);

enum : uint32_t {
  kNrShaderVerifyAlready = 0,  // this key was already checked
  kNrShaderVerifyAgreed = 1,
  kNrShaderVerifySizeNe = 2,
  kNrShaderVerifyBytesNe = 3,
  kNrShaderVerifyValidNe = 4,  // theirs exists, ours failed to translate
  kNrShaderVerifyPending = 5,  // theirs not translated yet, ask again later
};

// Compares one key against the emulated pipeline cache's binary, once. Pass
// theirs == nullptr when their translation is not available yet (async
// compilation); that counts as pending and leaves the key unverified so a
// later draw can settle it.
uint32_t NrShaderCacheVerify(NrShaderEntry* entry, const uint8_t* theirs, uint64_t theirs_size);

const NrShaderCacheStats& NrShaderCacheGetStats();

// Clears the per-second counters. The sets, the verdicts and the latency
// extremes are cumulative and survive.
void NrShaderCacheEndWindow();

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_SHADER_CACHE_H_
