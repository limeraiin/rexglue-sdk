/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_NATIVE_PSO_H_
#define REX_GRAPHICS_NR_NATIVE_PSO_H_

#include <cstdint>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <d3d12.h>

#include <rex/graphics/nr_pipeline_state.h>

// [NR-NPSO] THE NATIVE PIPELINE OBJECT: native-renderer phase 5, increment
// 5-3a -- the first peel.
//
// What the ladder has established, and what this adds:
//
//   5-1  register file -> the packed pipeline description   (mismatch=0, city)
//   5-2  ucode + modification -> DXBC                       (agreed=679/679)
//   5-3a description + DXBC -> ID3D12PipelineState          <- here
//
// So this unit owns exactly one mapping: from the 64-byte description (whose
// derivation 5-1 already gates, and whose bytes are therefore taken as given
// here) plus our own shader bytes (5-2's) to a filled
// D3D12_GRAPHICS_PIPELINE_STATE_DESC and the object created from it. Nothing
// upstream is re-derived and nothing downstream is assumed.
//
// The mapping is a transcription surface of its own -- the blend-factor table,
// the "comparison functions are the same plus one" rule, the RTV/DSV format
// switches, the sample-mask trick for unsupported 2x MSAA, the
// disable-rasterization collapse -- and none of it is covered by 5-1 or 5-2.
// It is therefore written here independently of pipeline_cache.cpp's
// CreateD3D12Pipeline and then MEASURED against it: the consumer builds theirs
// from the same runtime description at the same moment on the same thread and
// compares the two structures byte for byte, naming the first differing field.
// Both sides are fully zeroed first, so padding compares equal and a memcmp is
// the whole gate.
//
// Three host objects are DELEGATED rather than mirrored, because they are not
// derivations of guest state and a later increment owns each:
//   - the root signature (a function of the shaders' binding layouts; the
//     binding peel of 5-3 owns it),
//   - the geometry shader for primitive types Direct3D lacks (synthesized host
//     DXBC, not guest ucode),
//   - the prebuilt depth-conversion and tessellation shaders.
// They enter through NrNpsoBlobs as caller-owned pointers.
//
// Everything here is pure over caller-owned structs except the cache, which is
// a process global holding created objects. Creation and release go through
// callbacks, so tools/nr-native-pso-test.cpp builds this bare with no device.

namespace rex {
namespace graphics {
namespace nr {

// ---------------------------------------------------------------------------
// Why a draw could not be given one of our pipelines. Never silent: the
// consumer falls back to the emulated pipeline and counts the reason.
// ---------------------------------------------------------------------------
enum NrNpsoStatus : uint32_t {
  kNrNpsoOk = 0,
  // The description asks for a host vertex shader type this increment does not
  // build (domain shaders: the tessellation HS/DS pairs are prebuilt host
  // blobs, not something the walk recovers).
  kNrNpsoTessellation,
  // Pixel-shader-interlock path. A different rasterization model altogether
  // (depth/stencil and blending move into the shader); not this game's path.
  kNrNpsoRov,
  kNrNpsoBadTopology,
  kNrNpsoBadColorFormat,
  kNrNpsoBadDepthFormat,
  kNrNpsoBadMsaa,
  // Our own DXBC was not available for one of the stages.
  kNrNpsoNoShaderBytes,
  // The cache is full. Counted, never evicted: an entry pointer handed to a
  // draw has to stay valid, and a created pipeline that is still bound must
  // not be released.
  kNrNpsoCacheFull,
  // CreateGraphicsPipelineState failed. Remembered on the entry so a hopeless
  // description costs one attempt, not one per draw.
  kNrNpsoCreateFailed,
  kNrNpsoStatusCount,
};
const char* NrNpsoStatusName(uint32_t status);

// ---------------------------------------------------------------------------
// Host facts that are constant for the life of a device but that the mapping
// needs. Not part of the cache key: a device reset resets the cache.
// ---------------------------------------------------------------------------
struct NrNpsoEnv {
  bool edram_rov_used;
  bool msaa_2x_supported;
  bool gamma_render_target_as_unorm16;
  bool depth_float24_convert_in_pixel_shader;
  bool depth_float24_round;
  uint32_t draw_resolution_scale_x;
  uint32_t draw_resolution_scale_y;
};

// The byte sources. All caller-owned and all required to outlive the created
// pipeline object only until creation returns (Direct3D copies the bytecode).
struct NrNpsoBlobs {
  ID3D12RootSignature* root_signature;
  // Ours, from the 5-2 cache.
  const void* vs;
  size_t vs_size;
  // Ours; null when drawing without a pixel shader.
  const void* ps;
  size_t ps_size;
  // Delegated host blobs.
  const void* gs;
  size_t gs_size;
  const void* depth_only_ps;
  size_t depth_only_ps_size;
  const void* float24_round_ps;
  size_t float24_round_ps_size;
  const void* float24_truncate_ps;
  size_t float24_truncate_ps_size;
};

// ---------------------------------------------------------------------------
// The mapping. Individually exposed where it is individually wrong-able.
// ---------------------------------------------------------------------------

// Format of a colour render target as bound for drawing. Transcribed from the
// Xenos colour format set; the gamma case depends on a device capability.
DXGI_FORMAT NrNpsoColorDrawFormat(uint32_t color_format, bool gamma_as_unorm16);

// Format of the depth/stencil view. D24FS8 is stored as float depth on the
// host, so it is NOT D24_UNORM_S8_UINT.
DXGI_FORMAT NrNpsoDepthDsvFormat(uint32_t depth_format);

// Fills `out` completely (zeroed first, including padding, so two builds are
// memcmp-able). Returns kNrNpsoOk, or the reason a native pipeline cannot be
// described. `out` is left zeroed on failure.
uint32_t NrNpsoBuildStateDesc(const NrPsoDesc& desc, const NrNpsoEnv& env,
                              const NrNpsoBlobs& blobs,
                              D3D12_GRAPHICS_PIPELINE_STATE_DESC* out);

// ---------------------------------------------------------------------------
// The gate: our filled description against theirs.
// ---------------------------------------------------------------------------
enum : uint32_t {
  kNrNpsoCmpEqual = 0,
  // A stage's bytecode length differs.
  kNrNpsoCmpBytecodeSize,
  // Lengths agree, bytes do not. For VS and PS this would contradict 5-2.
  kNrNpsoCmpBytecodeBytes,
  // One side has bytecode for a stage and the other does not.
  kNrNpsoCmpBytecodePresence,
  // Any other member. `offset_out` receives the first differing byte offset.
  kNrNpsoCmpField,
};
// Compares everything: the five bytecode fields by length and content, and
// every other member byte for byte. `stage_out` receives 0=VS 1=PS 2=DS 3=HS
// 4=GS for the bytecode verdicts, `offset_out` the byte offset for
// kNrNpsoCmpField (and the first differing bytecode byte otherwise).
uint32_t NrNpsoCompareStateDesc(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& ours,
                                const D3D12_GRAPHICS_PIPELINE_STATE_DESC& theirs,
                                uint32_t* stage_out, uint32_t* offset_out);

// Names the member a byte offset falls in, including "BlendState.RenderTarget
// [2].SrcBlend" style names. Writes into a caller buffer.
void NrNpsoFieldName(uint32_t byte_offset, char* out, uint32_t out_size);

// ---------------------------------------------------------------------------
// The cache. Keyed on the description bytes alone -- which is legitimate for
// the same reason the emulated cache does it: the shaders' identity (hash and
// modification) and the tessellation flag are IN the description, so the root
// signature, the geometry shader and the DXBC are all functions of the key.
// ---------------------------------------------------------------------------
using NrNpsoCreateFn = ID3D12PipelineState* (*)(void* ctx,
                                                const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc);
using NrNpsoReleaseFn = void (*)(void* ctx, ID3D12PipelineState* state);

struct NrNpsoEntry {
  NrPsoDesc desc;  // the key, stored so a probe collision cannot alias
  ID3D12PipelineState* state;
  // kNrNpsoOk, or why this description will never get a native pipeline.
  // Sticky: a failure is remembered, not retried per draw.
  uint32_t status;
  bool verified;
  bool agreed;
  // Our filled description, kept for the verification a later draw settles.
  D3D12_GRAPHICS_PIPELINE_STATE_DESC state_desc;
};

struct NrNpsoStats {
  // Per window.
  uint64_t requests;
  uint64_t hits;
  uint64_t misses;
  uint64_t fallbacks;
  uint64_t fallback_reason[kNrNpsoStatusCount];

  // Cumulative.
  uint32_t entries;
  uint32_t entries_ok;
  uint64_t created;
  uint64_t create_fail;
  uint64_t create_ns_total;
  uint64_t create_ns_max;
  uint64_t refused;
  uint32_t probe_ovf;

  // The gate.
  uint64_t verified;
  uint64_t agreed;
  uint64_t desc_ne;
  bool have_first_ne;
  uint32_t first_ne_verdict;
  uint32_t first_ne_stage;
  uint32_t first_ne_offset;
  uint64_t first_ne_vs_hash;
  uint64_t first_ne_ps_hash;
};

void NrNpsoCacheConfigure(NrNpsoCreateFn create_fn, NrNpsoReleaseFn release_fn, void* ctx,
                          uint32_t max_entries);
bool NrNpsoCacheConfigured();
// Releases every created pipeline and empties the cache. Cumulative stats are
// cleared too: a reset means a new device, so nothing before it is comparable.
void NrNpsoCacheReset();

// Finds or creates our pipeline for this description. Returns null only when
// the cache is unconfigured or full (both counted). The returned pointer is
// stable for the life of the cache; `state` is null when `status` is not
// kNrNpsoOk.
NrNpsoEntry* NrNpsoCacheLookup(const NrPsoDesc& desc, const NrNpsoEnv& env,
                               const NrNpsoBlobs& blobs);

// Settles the comparison for an entry that has not been verified yet. A no-op
// on an already-verified entry, so the memcmp runs once per pipeline rather
// than once per draw.
uint32_t NrNpsoVerify(NrNpsoEntry* entry, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& theirs);

// Counts a draw that was, or was not, given our pipeline to bind.
void NrNpsoCountBind(bool ours);
// Window counters for the bind decision.
uint64_t NrNpsoBoundOurs();
uint64_t NrNpsoBoundTheirs();

const NrNpsoStats& NrNpsoGetStats();
void NrNpsoEndWindow();

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_NATIVE_PSO_H_
