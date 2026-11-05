// [NR-NATIVE-R2] v2: the fast native record executor.
//
// gpu_native_exec selects, per marker, how a native ring record becomes a
// D3D12 draw:
//   0  v1, the delegate body (command_processor.cpp): every register and
//      constant of the record goes through the virtual WriteRegister, then
//      IssueDrawImpl derives the whole draw, then every write is undone
//      through WriteRegister again.
//   1  fast (this file): the record is stored RAW into the live register
//      file with a save list, and only the D3D12 invalidations the changed
//      values need are applied (WriteRegister's tail, by change class, not
//      per dword); the draw is derived ONCE per object identity and then
//      replayed from a cache entry: pipeline handle + root signature + native
//      PSO, primitive processing result, host viewport, texture bindings with
//      their SRV indices, sampler params and heap indices, descriptor-indices
//      packs. What stays per draw is what changes per draw or what another
//      subsystem owns: the render target cache update (its entry guard),
//      texture reload checks and barriers, the pipeline bind (tracker), the
//      fixed-function state (latched), the system constants, the float
//      constant packs, root parameters, vertex residency, the IA views, the
//      index buffer view and the draw. Then the raw restore.
//   2  compare: the fast derivation runs WITHOUT drawing, the raw overlay is
//      undone, v1 draws, and the pipeline, root signature, texture bindings,
//      descriptor packs, constant buffer bytes and viewport v1 used are
//      compared with the fast ones (eq/ne per class, first 16 mismatches
//      logged). The gate for mode 1: run 2 until every class reads ne 0.
//
// Caches (all CP thread, all invalidated by the fast generation, which
// ClearCaches and InitializeShaderStorage bump):
//   shader table   (vs addr, vs dwords, ps addr, ps dwords) -> Shader pair,
//                  four sampled ucode dwords per stage, full LoadShader
//                  recheck every 256th hit of an entry.
//   object table   128-bit XXH3 over FastComposeObjectKey (the [dcache]
//                  identity words plus the record's draw identity) -> entry.
//                  Also refused: a texture destroy generation or a bindless
//                  heap that moved since the record.
//   PSO table      FastComposePsoKey words (every ConfigurePipeline input)
//                  -> handle, root signature, native PSO; consulted on an
//                  object miss before ConfigurePipeline.
//   texture table  the 6 fetch dwords -> a texture snapshot (binding + SRV
//                  indices), under the destroy generation; consulted per
//                  used slot on an object miss before RequestTextures.

#include <rex/cvar.h>
#include <rex/graphics/d3d12/command_processor.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/nr_bindings.h>
#include <rex/graphics/nr_native_fast.h>
#include <rex/graphics/nr_detile.h>
#include <rex/graphics/nr_direct_env.h>
#include <rex/graphics/frame_trace_scope.h>
#include <rex/graphics/nr_float_diff.h>
#include <rex/graphics/nr_sys_constants.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/hash.h>

REXCVAR_DEFINE_BOOL(gpu_native_retail_cmp, false, "GPU",
                    "Compare emitted native and bracketed retail draws; requires RetailFollows/JDNE.");
#include <rex/logging.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <optional>
#include <vector>

REXCVAR_DEFINE_INT32(gpu_native_exec, 1, "GPU/D3D12",
                     "[jade-native] native record executor: 0 v1 delegate, 1 fast (default) "
                     "(cached derivation, raw overlay), 2 compare (fast derivation vs the v1 "
                     "draw, eq/ne per class).");
REXCVAR_DEFINE_INT32(gpu_native_fast_lean, 1, "GPU/D3D12",
                     "[jade-native] fast executor work cuts (mode 1 only; runs 974/975 shipped "
                     "1, 0 is the kill switch until the Intel read): 0 full overlay, 1 lean: "
                     "record constants composed straight into the float upload (no overlay "
                     "or restore of them); the sort census and the occlusion eligibility "
                     "decode skipped while their consumers are off.");
REXCVAR_DEFINE_INT32(gpu_native_direct, 1, "GPU/D3D12",
                     "[jade-native] direct emit head for owned object hits (N4 step 1): 0 off, "
                     "1 direct (default; the entry's baked viewport / scissor / stencil / Hi-Z mask / "
                     "pool key / system constants through the SDK's per-list latches), 2 "
                     "compare (prepare the direct state, draw through the full path, diff "
                     "the classes: [jade-native-direct]).");

namespace rex::graphics::d3d12 {

using namespace nr_native;

namespace {

using Clock = std::chrono::steady_clock;
inline uint64_t NsSince(Clock::time_point t0) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
}

constexpr uint32_t kFastTexMax = 16;       // used texture slots an entry holds
constexpr uint32_t kFastObjSetBits = 12;   // 4096 sets x 2 ways (run 973: 2,953 live, 310k evictions/s at 11)
constexpr uint32_t kFastPsoBits = 12;      // 4096 direct-mapped
constexpr uint32_t kFastTexBits = 12;      // 4096 direct-mapped
constexpr uint32_t kFastIaMax = 8;

enum : uint8_t { kFastKindDraw = 0, kFastKindNoDraw = 1, kFastKindDelegate = 2 };

struct FastEntry {
  uint64_t h1 = 0, h2 = 0;
  uint32_t stamp = 0;
  bool valid = false;
  uint32_t gen = 0, tex_gen = 0;
  uint64_t smp_heap = 0, view_heap = 0;
  const D3D12Shader* vs = nullptr;  // the record's vertex shader
  D3D12Shader* ps = nullptr;        // after the pixel-shader-needed rule (null: none)
  uint8_t kind = kFastKindDraw, reason = 0;
  uint8_t ia = 0, polygonal = 0, blob_ok = 0, tex_count = 0;
  uint32_t ndc = 0, color_mask = 0;
  D3D_PRIMITIVE_TOPOLOGY topology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
  PrimitiveProcessor::ProcessingResult prim = {};
  uint32_t ib_req_base = 0, ib_req_size = 0;
  void* pso = nullptr;
  ID3D12RootSignature* rootsig = nullptr;
  ID3D12PipelineState* npso = nullptr;
  uint32_t rt_bits = 0;
  uint32_t rt_formats[5] = {};
  draw_util::ViewportInfo viewport = {};
  uint32_t tex_mask = 0;
  TextureCache::NrTexSnap tex[kFastTexMax] = {};
  alignas(16) uint8_t blob[kFastBlobBytes] = {};
  // Direct submission head (gpu_native_direct): baked by NrFastRecordEntry
  // from the miss emission's values; direct_ok 0 when an input was not
  // available (no host render targets, a compare-mode record).
  uint8_t direct_ok = 0;
  uint8_t hiz_mono_mask = 0;  // FastHizMonoMask(ndc)
  uint8_t pool_cls = 0;       // PoolClassOpaque at record
  // Step 4: the environment stamp of the bake (nr_direct_env.h) and the
  // fields behind it (mode 2 class env).
  uint32_t env = 0;
  FastEnvFields env_fields = {};
  uint64_t pool_key = 0;      // the pool pass key (the inst_mix chain)
  uint32_t stencil_ref = 0;
  uint64_t sys_hash = 0;      // XXH3_64 of sys
  D3D12_VIEWPORT vp = {};
  draw_util::Scissor sc_guest = {};  // GetScissor, guest pixels, before the live widen
  nr::NrSysConstants sys = {};
  // Step 2 latches and bakes.
  uint64_t rt_serial = UINT64_MAX;  // RT cache nr_update_serial at the entry's last Update
  uint8_t xfer_used = 0, xfer_reads = 0;  // FastXferVerdict at record
  uint8_t xfer_reason[5] = {};
  uint8_t ib_latched = 0;      // a successful index RequestRange under shm_gen
  uint32_t shm_gen = 0;        // the executor's shared memory invalidation generation
  uint32_t indx_offset = 0;    // VGT_INDX_OFFSET.indx_offset (keyed)
  uint32_t dt_regs[6] = {};    // the de-tile widen's six registers (keyed)
  // Step 3: the host scissor after the widen and the scale (baked unless a
  // widen segment limit makes the widen per draw), the index buffer view of
  // an indexed draw whose base is static (not an IA guest-DMA view).
  uint8_t sc_baked = 0;
  FastRect sc_host = {};
  uint8_t ibv_ok = 0;
  FastIbv ibv = {};
  // The texture binding generation latch (nr_binding_generation): taken
  // after a successful restore + ensure-loaded on a hit.
  uint8_t tex_latched = 0;
  uint32_t tex_bgen = 0;
};

struct FastPsoSlot {
  bool valid = false;
  uint32_t gen = 0;
  uint32_t words[kFastPsoKeyWords] = {};
  void* pso = nullptr;
  ID3D12RootSignature* rootsig = nullptr;
  ID3D12PipelineState* npso = nullptr;
  bool npso_done = false;
};

struct FastTexSlot {
  bool valid = false;
  uint32_t gen = 0, tex_gen = 0;
  uint64_t view_heap = 0;
  TextureCache::NrTexSnap snap = {};
};

// Tables (allocated on first use) and the per-marker scratch.
std::vector<FastEntry> g_entries;
FastObjectTable<FastEntry, kFastObjSetBits> g_objects;
std::vector<FastPsoSlot> g_psos;
std::vector<FastTexSlot> g_texs;
FastShaderTable<Shader> g_shaders;
uint32_t g_stamp = 1;
FastSaved g_saved[kFastMaxSaved];
uint32_t g_key_words[kFastKeyMaxWords];
FastConstMap g_lean_map;  // lean: record reg -> constant index
// Step 2: the record register map (direct lookups and consumers without the
// overlay), the per-marker overlay state (applied on demand) and the shared
// memory invalidation generation (a global watch, released at shutdown).
FastRegMap g_map;
FastDrawRegs g_dr;
FastOverlayStats g_ost;
std::atomic<uint32_t> g_shm_gen{1};
void* g_shm_watch = nullptr;
void ShmWatchCallback(const std::unique_lock<std::recursive_mutex>&, void*, uint32_t, uint32_t,
                      bool) {
  g_shm_gen.fetch_add(1, std::memory_order_relaxed);
}
// Per-marker scratch the plan points at (never re-initialized per draw).
TextureCache::NrTexSnap g_plan_tex[kFastTexMax];

void EnsureTables() {
  if (!g_entries.empty()) return;
  g_entries.resize(size_t(2) << kFastObjSetBits);
  g_objects.ways = g_entries.data();
  g_psos.resize(size_t(1) << kFastPsoBits);
  g_texs.resize(size_t(1) << kFastTexBits);
}

// Diagnostics beyond the contract counters ([jade-native-fast] 1 Hz).
struct FastStats {
  uint64_t fb[kFastFbReasons] = {};
  uint64_t delegate_hits = 0;    // entry says delegate class (no derivation)
  uint64_t nodraw = 0;           // register-determined no-draw
  uint64_t nodraw_async = 0;     // pipeline still compiling (not recorded)
  uint64_t rt_rederive = 0;      // hit re-derived: bound render target formats differ
  uint64_t tex_restore_fail = 0; // entry snapshots refused (a watch fired): full request
  uint64_t blob_refused = 0;     // bindings not recordable (restore off for the entry)
  uint64_t shader_recheck = 0, shader_recheck_ne = 0;
  uint64_t pso_table_hit = 0;    // object miss served by the PSO table
  uint64_t evictions = 0;
  uint64_t entries = 0;
};
FastStats g_stats;

// Direct submission (gpu_native_direct): counters and the mode 2 classes.
struct FastDirectStats {
  uint64_t draws = 0;        // owned hits emitted through the direct head
  uint64_t refused_env = 0;  // entry baked under another RT path / scale / depth convert
  uint64_t nodirect = 0;     // hit whose entry has no baked head
  uint64_t sys_ucp_stale = 0;  // mode 2: sys equal once the unread clip-plane dwords are zeroed
  uint64_t sys_ucp_live_ne = 0;  // mode 2: only the enabled (patched) clip-plane dwords differ
  uint64_t rt_skip = 0;        // (a) Update replaced by the direct tail
  uint64_t ib_skip = 0;        // (c) index RequestRange skipped by the latch
  uint64_t overlay_skip = 0;   // (d) neither overlay apply nor restore
  uint64_t overlay_live_rt = 0, overlay_live_blob = 0, overlay_live_other = 0;  // (d) refusals
  // Step 3: (b) the texture generation latch; the di and sys latches; the
  // IA entries whose index view stays live; step 4: entries re-baked after
  // an environment stamp change.
  uint64_t tex_skip = 0;       // restore + ensure-loaded skipped (barriers live)
  uint64_t tex_live = 0;       // restore ran on a direct hit
  uint64_t tex_gen_miss = 0;   // of tex_live: the entry was latched, the generation moved
  uint64_t tex_outdated = 0;   // of tex_live: a texture watch was pending
  uint64_t di_skip = 0;        // descriptor-indices stage uploads skipped by the latch
  uint64_t sys_skip = 0;       // system constants uploads skipped by the latch
  uint64_t ia_live = 0;        // direct draws whose index view is the live IA view
  uint64_t env_refresh = 0;    // entries dropped for re-bake on a stamp mismatch
  uint64_t eq[kFastDirectClasses] = {};
  uint64_t ne[kFastDirectClasses] = {};
  uint32_t logged[kFastDirectClasses] = {};
};
FastDirectStats g_direct;

// [jade-native-derive] sub-stage probe inside the derive bracket: host ticks
// per sub-stage (the trace's clock), only while the split brackets are on
// (ftrace::g_exec_split); reset by the 1 Hz report. The bracket opens per
// marker for the overlay apply (ovl), the NrFastDerive tail (sub: submission
// open, ib: index residency, ia: the IA view acquire, rt: render target
// cache Update, tex: texture restore + ensure loaded + barriers; a miss
// derivation is one lump, miss), the NrFastEmit head (hiz .. sys) and the
// restore (ovr, its own derive bracket in NrNativeExecuteFast). Each span
// charges its unlapped remainder to rest. n counts emissions.
struct FastDeriveProbe {
  uint64_t n = 0;
  uint64_t hiz = 0, pool = 0, pso = 0, vport = 0, scis = 0, ffs = 0, sys = 0, rest = 0;
  uint64_t ovl = 0, ovr = 0, sub = 0, ib = 0, ia = 0, rt = 0, tex = 0, miss = 0;
  uint64_t map = 0, txb = 0;  // the record map build; the texture barriers (split from tex)
};
FastDeriveProbe g_derive;
// One derive bracket span: laps accumulate since the previous lap, the
// destructor charges what no lap took to *tail (rest, or miss).
struct FastProbeSpan {
  bool on;
  uint64_t t0 = 0, start = 0, probe_laps = 0;
  uint64_t* tail = &g_derive.rest;
  explicit FastProbeSpan(bool enabled) : on(enabled) {
    if (on) t0 = start = rex::chrono::Clock::QueryHostTickCount();
  }
  FastProbeSpan(const FastProbeSpan&) = delete;
  FastProbeSpan& operator=(const FastProbeSpan&) = delete;
  void Lap(uint64_t& acc) {
    if (!on) return;
    const uint64_t now = rex::chrono::Clock::QueryHostTickCount();
    const uint64_t d = now - t0;
    acc += d;
    probe_laps += d;
    t0 = now;
  }
  ~FastProbeSpan() {
    if (!on) return;
    const uint64_t total = rex::chrono::Clock::QueryHostTickCount() - start;
    *tail += total > probe_laps ? total - probe_laps : 0;
  }
};

// Compare mode (gpu_native_exec 2).
struct FastCmpStats {
  uint64_t eq[kFastCmpClasses] = {};
  uint64_t ne[kFastCmpClasses] = {};
  uint64_t skip_emit = 0;  // v1 emitted no draw command (pool hit / no-draw): binding classes skipped
  uint64_t no_packs = 0;   // object miss: descriptor packs not compared (no entry yet)
  uint64_t fallback = 0;   // fast derivation refused: v1 only
  uint64_t pso_desc_eq = 0, pso_desc_ne = 0, pso_desc_unavailable = 0;
  uint64_t excluded_pool = 0, excluded_instanced = 0, excluded_indirect = 0;
  uint32_t logged = 0;
  uint32_t logged_class[kFastCmpClasses] = {};
  uint32_t logged_pso_description = 0;
};
FastCmpStats g_cmp;

struct FastCmpCapture {
  bool valid = false;
  bool drew = false;
  bool emitted = false;
  const D3D12Shader* vs = nullptr;
  const D3D12Shader* ps = nullptr;
  uint32_t tex_mask = 0;
  void* guest_pso = nullptr;
  ID3D12PipelineState* ext_pso = nullptr;
  ID3D12RootSignature* rootsig = nullptr;
  bool pso_desc_valid = false;
  uint32_t pso_desc[16] = {};
  void* pso_description_handle = nullptr;
  draw_util::ViewportInfo vp = {};
  uint32_t tex_n = 0;
  TextureCache::NrTexSnap tex[kFastTexMax] = {};
  uint32_t div[64] = {}, dip[64] = {};
  uint32_t nv = 0, np = 0;
  bool bv = false, bp = false;
  alignas(16) uint8_t fv[nr::kBindFloatMaxBytes] = {};
  alignas(16) uint8_t fp[nr::kBindFloatMaxBytes] = {};
  uint32_t fv_n = 0, fp_n = 0;
  uint8_t bl[nr::kBindBoolLoopBytes] = {};
  uint8_t ft[nr::kBindFetchBytes] = {};
};
FastCmpCapture g_cap;
struct RetailDrawCapture {
  FastCmpCapture state;
  DxbcShaderTranslator::SystemConstants system = {};
  D3D12_RECT scissor = {};
  PrimitiveProcessor::ProcessingResult prim = {};
  uintptr_t targets[1 + xenos::kMaxColorRenderTargets] = {};
  bool targets_valid = false;
  bool direct = false;
  bool indexed = false;
  uint32_t args[5] = {};
  D3D12_INDEX_BUFFER_VIEW ibv = {};
};
struct RetailScope {
  uint32_t ref = 0;
  NrNativeDrawRecord record = {};
  RetailDrawCapture native, retail;
  uint32_t retail_draws = 0;
  uint32_t retail_prim = 0, retail_count = 0, retail_base = 0, retail_endian = 0, retail_format = 0;
  bool input_match = false;
};
RetailScope g_retail;
constexpr const char* kRetailClasses[] = {
    "identity", "shader_vs", "shader_ps", "pso", "root", "float_sizes", "fv", "fp",
    "bl", "ft", "system", "viewport", "scissor", "targets", "packs_meta", "div", "dip",
    "texture", "primitive", "indexed", "draw_args", "ib_view"};
constexpr uint32_t kRetailClassCount = sizeof(kRetailClasses) / sizeof(kRetailClasses[0]);
struct RetailStats {
  uint64_t begun = 0, paired = 0, superseded = 0, wrong_end = 0, no_retail = 0,
           multiple = 0, identity_ne = 0, unavailable = 0, classes_eq = 0, classes_ne = 0;
  uint32_t logged = 0;
  uint64_t eq[kRetailClassCount] = {}, ne[kRetailClassCount] = {};
  uint8_t class_logged[kRetailClassCount] = {};
  FloatDifferenceStats float_words[2];
  uint64_t float_size_different_draws[2] = {};
};
RetailStats g_retail_stats;
// The fast side of one compare.
alignas(16) uint8_t g_fast_fv[nr::kBindFloatMaxBytes];
alignas(16) uint8_t g_fast_fp[nr::kBindFloatMaxBytes];
uint8_t g_fast_bl[nr::kBindBoolLoopBytes];
uint8_t g_fast_ft[nr::kBindFetchBytes];

bool TexSnapEq(const TextureCache::NrTexSnap& a, const TextureCache::NrTexSnap& b) {
  return a.slot == b.slot && std::memcmp(a.fetch, b.fetch, sizeof(a.fetch)) == 0 &&
         !(a.binding.key != b.binding.key) && a.binding.host_swizzle == b.binding.host_swizzle &&
         a.binding.swizzled_signs == b.binding.swizzled_signs &&
         a.binding.texture == b.binding.texture &&
         a.binding.texture_signed == b.binding.texture_signed &&
         a.descriptor_index == b.descriptor_index &&
         a.descriptor_index_signed == b.descriptor_index_signed;
}

// [vtl] / [psf] sign gate (command_processor.cpp VtlSignsOk, same rule).
bool FastSignsOk(const xenos::xe_gpu_texture_fetch_t& tf) {
  const uint8_t signs = texture_util::SwizzleSigns(tf);
  if (signs == 0) return true;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t sg = (signs >> (i * 2)) & 3;
    if (sg != uint32_t(xenos::TextureSign::kUnsigned) &&
        sg != uint32_t(xenos::TextureSign::kSigned)) {
      return false;
    }
  }
  switch (tf.format) {
    case xenos::TextureFormat::k_16_FLOAT:
    case xenos::TextureFormat::k_16_16_FLOAT:
    case xenos::TextureFormat::k_16_16_16_16_FLOAT:
    case xenos::TextureFormat::k_32_FLOAT:
    case xenos::TextureFormat::k_32_32_FLOAT:
    case xenos::TextureFormat::k_32_32_32_32_FLOAT:
      return true;
    default:
      return false;
  }
}

bool HostTopology(xenos::PrimitiveType t, D3D_PRIMITIVE_TOPOLOGY* out) {
  switch (t) {
    case xenos::PrimitiveType::kPointList:
      *out = D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
      return true;
    case xenos::PrimitiveType::kLineList:
      *out = D3D_PRIMITIVE_TOPOLOGY_LINELIST;
      return true;
    case xenos::PrimitiveType::kLineStrip:
      *out = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
      return true;
    case xenos::PrimitiveType::kTriangleList:
    case xenos::PrimitiveType::kRectangleList:
      *out = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
      return true;
    case xenos::PrimitiveType::kTriangleStrip:
      *out = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
      return true;
    case xenos::PrimitiveType::kQuadList:
      *out = D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ;
      return true;
    default:
      return false;
  }
}

void CmpCount(uint32_t cls, bool eq, const NrNativeDrawRecord& rec, uint32_t a, uint32_t b,
              uint32_t detail) {
  if (eq) {
    ++g_cmp.eq[cls];
    NrNativeBump(&NrNativeCounters::cmp_eq);
    return;
  }
  ++g_cmp.ne[cls];
  NrNativeBump(&NrNativeCounters::cmp_ne);
  if (g_cmp.logged_class[cls] < 2) {
    ++g_cmp.logged_class[cls];
    ++g_cmp.logged;
    REXGPU_WARN(
        "[jade-native-cmp] MISMATCH #{} class={} token={:08X} frame={} vs={:08X}/{} ps={:08X}/{} "
        "fast={:08X} v1={:08X} detail={}",
        g_cmp.logged, FastCmpName(cls), rec.object_token, rec.frame, rec.vs_address,
        rec.vs_dwords, rec.ps_address, rec.ps_dwords, a, b, detail);
  }
}

// The de-tile widen (NrDetileWidenDrawScissor's rule without its probe) over
// an entry's six baked registers: the answer stays live, it reads the
// frame's band-1 tap progress and the widen segment limit.
void FastWidenBaked(const uint32_t dt_regs[6], draw_util::Scissor& scissor) {
  const nr::DetileRegs dt{dt_regs[0], dt_regs[1], dt_regs[2], dt_regs[3], dt_regs[4], dt_regs[5]};
  const uint32_t full_height = nr::DetileWidenFullHeight(dt, true);
  if (!full_height || scissor.offset[1] + scissor.extent[1] >= full_height) return;
  scissor.extent[1] = full_height - scissor.offset[1];
  nr::DetileCountScissorWidened();
}

// VGT_INDX_OFFSET.indx_offset from the map (record over live) or the file.
inline uint32_t g_map_indx_offset(bool mapped, const uint32_t* rf, const RegisterFile& regs) {
  if (mapped) return g_map.Read(rf, XE_GPU_REG_VGT_INDX_OFFSET) & 0xFFFFFFu;
  return regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
}

// gpu_native_direct 2: one class verdict, the first 4 mismatches per class logged.
void DirectCount(uint32_t cls, bool eq, const NrNativeDrawRecord& rec, uint64_t direct,
                 uint64_t full, uint32_t detail) {
  if (eq) {
    ++g_direct.eq[cls];
    return;
  }
  ++g_direct.ne[cls];
  if (g_direct.logged[cls] < 4) {
    ++g_direct.logged[cls];
    REXGPU_WARN("[jade-native-direct] MISMATCH class={} #{} token={:08X} frame={} vs={:08X} "
                "ps={:08X} ib={:08X} n={} direct={:016X} full={:016X} detail={}",
                FastDirectName(cls), g_direct.logged[cls], rec.object_token, rec.frame,
                rec.vs_address, rec.ps_address, rec.index_base, rec.index_count, direct, full,
                detail);
  }
}

// First differing dword of two byte runs (UINT32_MAX when equal), for logs.
uint32_t FirstDiffDword(const void* a, const void* b, uint32_t bytes) {
  const uint32_t* x = static_cast<const uint32_t*>(a);
  const uint32_t* y = static_cast<const uint32_t*>(b);
  for (uint32_t i = 0; i < bytes / 4; ++i) {
    if (x[i] != y[i]) return i;
  }
  return UINT32_MAX;
}

uint32_t PackedFloatRegister(const uint64_t bitmap[4], uint32_t packed_dword) {
  uint32_t packed_register = packed_dword / 4;
  for (uint32_t word = 0; word < 4; ++word) {
    uint64_t bits = bitmap[word];
    while (bits) {
      const uint32_t bit = uint32_t(__builtin_ctzll(bits));
      bits &= bits - 1;
      if (!packed_register--) return word * 64 + bit;
    }
  }
  return UINT32_MAX;
}

}  // namespace

// Everything one fast marker derives; on an object hit, copied from the entry.
struct D3D12CommandProcessor::NrFastPlan {
  const NrNativeDrawRecord* rec = nullptr;
  int mode = 1;
  bool lean = false;       // gpu_native_fast_lean 1 (mode 1 only)
  uint8_t lean_floats = 0; // bit 0 vs, bit 1 ps: upload holds record constants
  D3D12Shader* vs = nullptr;
  D3D12Shader* ps_record = nullptr;
  // The object key and the entry it found (hit) or will fill (miss).
  FastKey128 key;
  uint32_t key_tex_mask = 0;
  FastKeyInputs key_inputs;  // the composition inputs (mode 2 recomposes over the file)
  FastEntry* entry = nullptr;
  bool hit = false;
  // Derived.
  uint8_t kind = kFastKindDraw;
  uint8_t reason = 0;          // delegate entries: the fallback reason
  bool record_nodraw = false;  // a no-draw that is a pure function of the key
  D3D12Shader* ps = nullptr;
  bool polygonal = false;
  reg::RB_DEPTHCONTROL ndc = {};
  uint32_t color_mask = 0;
  PrimitiveProcessor::ProcessingResult prim = {};
  uint32_t ib_req_base = 0, ib_req_size = 0;
  D3D_PRIMITIVE_TOPOLOGY topology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
  bool ia = false;
  D3D12_GPU_VIRTUAL_ADDRESS ia_vb_va[kFastIaMax] = {};
  D3D12_GPU_VIRTUAL_ADDRESS ia_ib_va = 0;
  void* pso = nullptr;
  ID3D12RootSignature* rootsig = nullptr;
  ID3D12PipelineState* npso = nullptr;
  uint32_t rt_bits = 0;
  uint32_t rt_formats[5] = {};
  uint32_t tex_mask = 0;
  uint32_t tex_count = 0;
  TextureCache::NrTexSnap* tex = g_plan_tex;  // scratch, tex_count entries live
  draw_util::ViewportInfo viewport = {};
  bool bind_ok = false;
  bool failed = false;  // a stage failed where IssueDrawImpl returns false (refused draw)
  // Direct submission (gpu_native_direct): direct = this marker emitted
  // through the direct head; direct_ready = the miss emission baked the head
  // below for the record.
  bool direct = false;
  bool direct_ready = false;
  bool pool_cls = false;
  uint64_t pool_key = 0;
  uint32_t stencil_ref = 0;
  uint64_t sys_hash = 0;
  D3D12_VIEWPORT vp = {};
  draw_util::Scissor sc_guest = {};
  nr::NrSysConstants sys = {};
  // Step 3 bakes (a miss) and the direct bind flag (a direct hit's bindings
  // went through the descriptor-indices latch: the post-draw dirty set).
  bool sc_baked = false;
  FastRect sc_host = {};
  bool ibv_ok = false;
  FastIbv ibv = {};
  bool direct_bind = false;
  // Step 2.
  int32_t dmode = 0;          // gpu_native_direct, when applicable (exec 1 and lean)
  bool mapped = false;        // g_map holds this record: register reads go through it
  bool overlay = false;       // the raw overlay is applied (a restore is owed)
  bool overlay_skip = false;  // (d) direct hit: the overlay was never applied
  bool direct_env_ok = false; // the entry's head is bakeable here (env stamp match)
  bool rt_skip = false, ib_skip = false;
  uint64_t sys_hash_draw = 0; // the patched block's hash (the latch key)
  FastChanges dry;            // (d) the dirty set a skipped overlay owes, pre and post draw
};

// ── Dispatch ────────────────────────────────────────────────────────────────
void D3D12CommandProcessor::NrNativeExecute(uint32_t slot_sequence) {
  if (REXCVAR_GET(gpu_native_retail_cmp)) NrRetailCmpBegin(slot_sequence);
  nr_retail_cmp_native_ = nr_retail_cmp_active_;
  struct NativeScopeExit {
    bool& native;
    ~NativeScopeExit() { native = false; }
  } scope_exit{nr_retail_cmp_native_};
  const int32_t mode = REXCVAR_GET(gpu_native_exec);
  if (mode == 1 || mode == 2) {
    NrNativeExecuteFast(slot_sequence, mode);
    return;
  }
  const bool timed = ftrace::g_exec_split;
  const auto t0 = timed ? Clock::now() : Clock::time_point{};
  const uint64_t drawn0 = NrNativeCpCounters().drawn;
  NrNativeExecuteV1(slot_sequence);
  if (timed) NrNativeBump(&NrNativeCounters::v1_ns, NsSince(t0));
  NrNativeBump(&NrNativeCounters::v1_draws, NrNativeCpCounters().drawn - drawn0);
}

// WriteRegister's D3D12 tail for a set of changed registers, by class.
void D3D12CommandProcessor::NrFastInvalidate(const FastChanges& ch) {
  if (!ch.Any()) return;
  if (frame_open_) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (ch.float_bits[i] & current_float_constant_map_vertex_[i]) {
        cbuffer_binding_float_vertex_.up_to_date = false;
      }
      if (ch.float_bits[4 + i] & current_float_constant_map_pixel_[i]) {
        cbuffer_binding_float_pixel_.up_to_date = false;
      }
    }
  }
  if (ch.bool_loop) cbuffer_binding_bool_loop_.up_to_date = false;
  if (ch.tex_fetch || ch.vtx_fetch[0] || ch.vtx_fetch[1]) {
    cbuffer_binding_fetch_.up_to_date = false;
    uint32_t t = ch.tex_fetch;
    while (t) {
      const uint32_t i = uint32_t(__builtin_ctz(t));
      t &= t - 1;
      if (texture_cache_) texture_cache_->TextureFetchConstantWritten(i);
    }
    for (uint32_t w = 0; w < 2; ++w) {
      uint64_t v = ch.vtx_fetch[w];
      while (v) {
        const uint32_t i = uint32_t(__builtin_ctzll(v));
        v &= v - 1;
        InvalidateVertexBufferResidency(w * 64 + i);
      }
    }
  }
}

// ── The fast executor ─────────────────────────────────────────────────────
void D3D12CommandProcessor::NrNativeExecuteFast(uint32_t slot_sequence, int mode) {
  // Per-marker wall clocks are diagnostic work, just like the split scopes.
  // In normal play this also avoids timing the discarded repeat-band markers.
  const bool timed = ftrace::g_exec_split;
  const auto t0 = timed ? Clock::now() : Clock::time_point{};
  uint64_t v1_ns = 0;  // the part of this marker spent in the v1 body
  struct FastTimer {
    bool timed;
    Clock::time_point t0;
    uint64_t* v1;
    ~FastTimer() {
      if (!timed) return;
      const uint64_t all = NsSince(t0);
      NrNativeBump(&NrNativeCounters::fast_ns, all > *v1 ? all - *v1 : 0);
    }
  } timer{timed, t0, &v1_ns};
  // The v1 body, timed as v1.
  auto run_delegate = [&](Shader* dvs, Shader* dps) {
    const auto d0 = timed ? Clock::now() : Clock::time_point{};
    const uint64_t drawn0 = NrNativeCpCounters().drawn;
    NrNativeDelegateBody(dvs, dps);
    if (timed) {
      const uint64_t ns = NsSince(d0);
      v1_ns += ns;
      NrNativeBump(&NrNativeCounters::v1_ns, ns);
    }
    NrNativeBump(&NrNativeCounters::v1_draws, NrNativeCpCounters().drawn - drawn0);
  };

  // Prologue: the v1 prologue's checks and counters, in its order.
  NrNativeBump(&NrNativeCounters::markers);
  NrNativeDrawRecord& rec = nr_native_rec_;
  bool torn = false;
  switch (NrNativeAcquireForDraw(slot_sequence, &rec, &torn, NrNativeDrawBandAllowed())) {
    case NrNativeVerdict::kOk:
      break;
    case NrNativeVerdict::kNoRing:
      NrNativeBump(&NrNativeCounters::refused_no_ring);
      return;
    case NrNativeVerdict::kSlotRange:
      NrNativeBump(&NrNativeCounters::refused_slot);
      return;
    case NrNativeVerdict::kSequence:
    default:
      NrNativeBump(torn ? &NrNativeCounters::refused_torn : &NrNativeCounters::refused_sequence);
      return;
  }
  if (!NrNativeRecordBoundsOk(rec.register_count, rec.constant_count)) {
    NrNativeBump(&NrNativeCounters::refused_record);
    return;
  }
  if (rec.flags & kNrNativeVerifyOnly) {
    NrNativeBump(&NrNativeCounters::verified);
    return;
  }
  if (NrFastPrecordCapture()) {
    NrNativeBump(&NrNativeCounters::refused_draw);
    NrNativeBump(&NrNativeCounters::refused_precord);
    if (!nr_native_precord_logged_) {
      nr_native_precord_logged_ = true;
      REXGPU_WARN(
          "[jade-native-cp] REFUSED: precord capture is on, native records are not drawn "
          "(counted as refused draw)");
    }
    return;
  }
  if (!NrNativeDrawAllowed()) {
    NrNativeBump(&NrNativeCounters::skipped_band);
    return;
  }
  EnsureTables();
  const bool lean = mode == 1 && REXCVAR_GET(gpu_native_fast_lean) != 0;

  // Shaders: the table, else LoadShader (which hashes the ucode).
  Shader* vs = nullptr;
  Shader* ps = nullptr;
  {
    ftrace::SplitScope _ft_lookup(&ftrace::SplitTotals::nr_lookup_ticks);
    const uint32_t vs_addr = rec.vs_address & ~3u;
    const uint32_t ps_addr = rec.ps_address & ~3u;
    const uint32_t* vs_host =
        rec.vs_dwords ? memory_->TranslatePhysical<const uint32_t*>(vs_addr) : nullptr;
    const uint32_t* ps_host =
        rec.ps_dwords ? memory_->TranslatePhysical<const uint32_t*>(ps_addr) : nullptr;
    using ShaderResult = FastShaderTable<Shader>::Result;
    const ShaderResult sr = g_shaders.Lookup(vs_addr, rec.vs_dwords, ps_addr, rec.ps_dwords, vs_host,
                                             ps_host, nr_fast_gen_, &vs, &ps);
    if (sr == ShaderResult::kMiss || sr == ShaderResult::kHitRecheck || mode == 2) {
      Shader* lvs = rec.vs_dwords ? LoadShader(xenos::ShaderType::kVertex, vs_addr,
                                               const_cast<uint32_t*>(vs_host), rec.vs_dwords)
                                  : nullptr;
      Shader* lps = rec.ps_dwords ? LoadShader(xenos::ShaderType::kPixel, ps_addr,
                                               const_cast<uint32_t*>(ps_host), rec.ps_dwords)
                                  : nullptr;
      if (sr != ShaderResult::kMiss) {
        const bool eq = lvs == vs && lps == ps;
        if (sr == ShaderResult::kHitRecheck) {
          ++g_stats.shader_recheck;
          if (!eq) ++g_stats.shader_recheck_ne;
        }
        if (mode == 2) {
          // Stage bits make a PS-only mismatch visible; log both shader pairs
          // before replacing the stale table result with freshly loaded ones.
          const uint32_t stages = (lvs != vs ? 1u : 0u) | (lps != ps ? 2u : 0u);
          const bool sample = !eq && g_cmp.logged_class[kFastCmpShader] < 2;
          CmpCount(kFastCmpShader, eq, rec, uint32_t(uintptr_t(vs)), uint32_t(uintptr_t(lvs)), stages);
          if (sample) {
            REXGPU_WARN("[jade-native-cmp] shader stages={} cached_vs={:#x} loaded_vs={:#x} "
                        "cached_ps={:#x} loaded_ps={:#x} cached_hash_vs/ps={:016X}/{:016X} "
                        "loaded_hash_vs/ps={:016X}/{:016X}",
                        stages, uintptr_t(vs), uintptr_t(lvs), uintptr_t(ps), uintptr_t(lps),
                        vs ? vs->ucode_data_hash() : 0, ps ? ps->ucode_data_hash() : 0,
                        lvs ? lvs->ucode_data_hash() : 0, lps ? lps->ucode_data_hash() : 0);
          }
        }
        if (!eq) {
          vs = lvs;
          ps = lps;
          if (vs && (!rec.ps_dwords || ps)) {
            g_shaders.Insert(vs_addr, rec.vs_dwords, ps_addr, rec.ps_dwords, vs_host, ps_host,
                             nr_fast_gen_, vs, ps);
          }
        }
        NrNativeBump(&NrNativeCounters::shader_hit);
      } else {
        vs = lvs;
        ps = lps;
        if (vs && (!rec.ps_dwords || ps)) {
          g_shaders.Insert(vs_addr, rec.vs_dwords, ps_addr, rec.ps_dwords, vs_host, ps_host,
                           nr_fast_gen_, vs, ps);
        }
        NrNativeBump(&NrNativeCounters::shader_miss);
      }
    } else {
      NrNativeBump(&NrNativeCounters::shader_hit);
    }
  }
  if (!vs || (rec.ps_dwords && !ps)) {
    NrNativeBump(&NrNativeCounters::refused_shader);
    return;
  }

  // Environment refusals: the v1 body, before any raw store.
  uint32_t env = NrFastEnvRefusal();
  if (!env && !FastDrawCountOk(rec.index_count)) env = kFastFbCount16;
  if (env) {
    ++g_stats.fb[env];
    NrNativeBump(&NrNativeCounters::fast_fallback);
    if (mode == 2) ++g_cmp.fallback;
    run_delegate(vs, ps);
    return;
  }

  // The three draw registers the replaced retail DRAW_INDX writes.
  g_dr = FastSynthDrawRegs(rec.prim_type, rec.index_count, rec.flags, rec.index_base,
                           rec.index_endian);
  g_ost = FastOverlayStats{};
  NrFastPlan plan;
  plan.rec = &rec;
  plan.mode = mode;
  plan.lean = lean;
  plan.vs = static_cast<D3D12Shader*>(vs);
  plan.ps_record = static_cast<D3D12Shader*>(ps);
  // Direct submission needs the lean float path (the record's constants never
  // enter the live file) and the fast emission.
  plan.dmode = mode == 1 && lean ? REXCVAR_GET(gpu_native_direct) : 0;
  if (plan.dmode) {
    // The record map: the lookup and every direct consumer read the record
    // through it, so a direct hit never needs the overlay.
    ftrace::SplitScope _ft_derive(&ftrace::SplitTotals::nr_derive_ticks);
    FastProbeSpan span(ftrace::g_exec_split);
    if (!g_shm_watch) g_shm_watch = shared_memory_->RegisterGlobalWatch(ShmWatchCallback, nullptr);
    if (g_map.rf_count != uint32_t(RegisterFile::kRegisterCount)) {
      g_map.Init(uint32_t(RegisterFile::kRegisterCount));
    }
    g_map.Build(rec, &g_dr);
    plan.mapped = true;
    span.Lap(g_derive.map);
    NrFastLookup(plan);
  }
  // (d) the overlay is skipped for a direct hit whose every register consumer
  // reads the map or the entry: the bindings restore from the blob (else
  // NrUpdateBindings derives sampler params from the live fetch constants),
  // the render target latch holds (else Update reads sixteen live registers),
  // no occlusion submission (OccSetElig reads the live file), no sorter
  // segment and no retail comparator (both capture the live file).
  bool overlay_skip = false;
  if (plan.dmode == 1 && plan.hit && plan.direct_env_ok && plan.entry->direct_ok &&
      plan.entry->kind == kFastKindDraw) {
    const FastEntry* e = plan.entry;
    if (!e->blob_ok) {
      ++g_direct.overlay_live_blob;
    } else if (e->rt_serial != render_target_cache_->nr_update_serial()) {
      ++g_direct.overlay_live_rt;
    } else if (occ_sub_active_ || NrFastSortOn() || nr_retail_cmp_active_) {
      ++g_direct.overlay_live_other;
    } else {
      overlay_skip = true;
    }
  }
  if (overlay_skip) {
    // The dirty set the overlay apply and its restore would have raised
    // (fetch, bool/loop, texture slots, vertex residency), raised now so the
    // draw's own bindings re-derive from the record, and again after the
    // draw so the next retail draw re-derives from the live file.
    ftrace::SplitScope _ft_derive(&ftrace::SplitTotals::nr_derive_ticks);
    FastProbeSpan span(ftrace::g_exec_split);
    plan.overlay_skip = true;
    ++g_direct.overlay_skip;
    FastOverlayClassify(register_file_->values, uint32_t(RegisterFile::kRegisterCount), rec, &g_dr,
                        plan.dry, /*constants=*/false);
    NrFastInvalidate(plan.dry);
    span.Lap(g_derive.ovl);
  } else {
    NrFastApplyOverlay(plan);
  }
  if (!plan.dmode) NrFastLookup(plan);  // from the overlaid file
  auto restore = [&]() {
    if (!plan.overlay) return;
    ftrace::SplitScope _ft_derive(&ftrace::SplitTotals::nr_derive_ticks);
    FastProbeSpan span(ftrace::g_exec_split);
    FastChanges back;
    FastOverlayRestore(register_file_->values, g_saved, g_ost.n_saved, back);
    NrFastInvalidate(back);
    plan.overlay = false;
    span.Lap(g_derive.ovr);
  };
  auto bump_overlay_counters = [&]() {
    const FastOverlayStats& ost = g_ost;
    NrNativeBump(&NrNativeCounters::registers_applied, ost.regs_applied);
    NrNativeBump(&NrNativeCounters::constants_applied, ost.consts_applied);
    if (ost.skipped_low) NrNativeBump(&NrNativeCounters::skipped_register_low, ost.skipped_low);
    if (ost.skipped_high) NrNativeBump(&NrNativeCounters::skipped_register_high, ost.skipped_high);
    if (ost.skipped_const) {
      NrNativeBump(&NrNativeCounters::skipped_constant_range, ost.skipped_const);
    }
  };
  if (NrNativeVizKilled()) {
    bump_overlay_counters();
    NrNativeBump(&NrNativeCounters::skipped_viz);
    restore();
    return;
  }
  if (plan.dmode == 2 && plan.hit && plan.direct_env_ok && plan.entry->direct_ok) {
    // (d) compare: with the overlay applied, the map must read what the file
    // holds at every direct consumer, and the entry's bakes must match.
    const uint32_t* rf = register_file_->values;
    const FastEntry* e = plan.entry;
    uint32_t bad = UINT32_MAX;
    auto check = [&](uint32_t index) {
      if (bad == UINT32_MAX && g_map.Read(rf, index) != rf[index]) bad = index;
    };
    for (uint32_t i = 0; i < 4; ++i) check(XE_GPU_REG_RB_BLEND_RED + i);
    for (uint32_t i = 0; i < 24; ++i) check(kFastRegPaClUcp0X + i);
    for (uint32_t i = kFastRegFetchBase; i < kFastRegFetchEnd; ++i) check(i);
    for (uint32_t i = kFastRegBoolBase; i < kFastRegLoopEnd; ++i) check(i);
    check(XE_GPU_REG_VGT_INDX_OFFSET);
    check(XE_GPU_REG_PA_CL_CLIP_CNTL);
    const uint32_t dt_idx[6] = {XE_GPU_REG_PA_SC_WINDOW_OFFSET, XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL,
                                XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR, XE_GPU_REG_RB_SURFACE_INFO,
                                XE_GPU_REG_RB_COLOR_INFO, XE_GPU_REG_RB_DEPTH_INFO};
    bool bake_ok = e->indx_offset == (rf[XE_GPU_REG_VGT_INDX_OFFSET] & 0xFFFFFFu);
    for (uint32_t i = 0; i < 6; ++i) bake_ok = bake_ok && e->dt_regs[i] == rf[dt_idx[i]];
    // The key over the file equals the key the map produced.
    FastKeyInputs ki = plan.key_inputs;
    ki.rf = rf;
    const uint32_t n = std::min(FastComposeObjectKey(ki, g_key_words, kFastKeyMaxWords),
                                kFastKeyMaxWords);
    const XXH128_hash_t h = XXH3_128bits(g_key_words, size_t(n) * sizeof(uint32_t));
    const bool key_ok = h.low64 == plan.key.h1 && h.high64 == plan.key.h2;
    DirectCount(kFastDirectOvl, bad == UINT32_MAX && bake_ok && key_ok, rec,
                bad == UINT32_MAX ? 0 : g_map.Read(rf, bad), bad == UINT32_MAX ? 0 : rf[bad],
                bad != UINT32_MAX ? bad : (!key_ok ? 0xFFFF0001u : 0xFFFF0002u));
  }
  const uint32_t fb = NrFastDerive(plan);

  if (fb || plan.kind == kFastKindDelegate) {
    // A class the fast path does not emit: undo the raw overlay, v1 draws.
    const uint32_t why = fb ? fb : plan.entry ? plan.entry->reason : uint32_t(kFastFbProcess);
    ++g_stats.fb[why < kFastFbReasons ? why : 0];
    NrNativeBump(&NrNativeCounters::fast_fallback);
    if (mode == 2) ++g_cmp.fallback;
    // A class verdict that is a pure function of the key is remembered, so
    // the next marker of the object goes to v1 without deriving again.
    // Transient refusals (processing, IA acquisition) are not.
    if (!plan.hit && plan.kind != kFastKindDelegate &&
        (why == kFastFbMemexport || why == kFastFbHostVs || why == kFastFbConverted ||
         why == kFastFbLineLoop || why == kFastFbTexCount || why == kFastFbEdram)) {
      plan.kind = kFastKindDelegate;
      plan.reason = uint8_t(why);
      NrFastRecordEntry(plan, false);
    }
    restore();
    run_delegate(vs, ps);
    return;
  }
  bump_overlay_counters();
  if (plan.failed) {
    NrNativeBump(&NrNativeCounters::refused_draw);
    restore();
    return;
  }

  if (mode == 2) {
    // ── Compare: the fast side from the raw-overlaid file, then v1 draws. ──
    NrNativeBump(&NrNativeCounters::cmp_markers);
    const uint32_t* rf = register_file_->values;
    const uint64_t zero_map[4] = {0, 0, 0, 0};
    const uint32_t fast_fv_n = FastComposeFloats(
        rf, kFastRegFloatBase, plan.vs->constant_register_map().float_bitmap, g_fast_fv,
        sizeof(g_fast_fv));
    const uint32_t fast_fp_n = FastComposeFloats(
        rf, kFastRegFloatBase + 0x400,
        plan.ps ? plan.ps->constant_register_map().float_bitmap : zero_map, g_fast_fp,
        sizeof(g_fast_fp));
    std::memcpy(g_fast_bl, &rf[kFastRegBoolBase], sizeof(g_fast_bl));
    std::memcpy(g_fast_ft, &rf[kFastRegFetchBase], sizeof(g_fast_ft));
    const bool drawable = plan.kind == kFastKindDraw;
    // v1 must derive its own texture bindings, or the compare is vacuous.
    if (plan.tex_mask) texture_cache_->NrTexDesync(plan.tex_mask);
    restore();
    g_cap = FastCmpCapture{};
    g_cap.vs = plan.vs;
    g_cap.ps = plan.ps;
    g_cap.tex_mask = plan.tex_mask;
    nr_fast_cmp_armed_ = true;
    run_delegate(vs, ps);
    nr_fast_cmp_armed_ = false;
    if (!g_cap.valid) return;  // v1 refused before its draw (viz, draw failure)
    bool all_eq = true;
    auto cmp = [&](uint32_t cls, bool eq, uint32_t a, uint32_t b, uint32_t detail) {
      CmpCount(cls, eq, rec, a, b, detail);
      all_eq = all_eq && eq;
    };
    // Constant buffer bytes: the overlay (raw vs WriteRegister) is compared
    // whether or not v1 emitted.
    cmp(kFastCmpFloatV,
        fast_fv_n == g_cap.fv_n && !std::memcmp(g_fast_fv, g_cap.fv, fast_fv_n), fast_fv_n,
        g_cap.fv_n, FirstDiffDword(g_fast_fv, g_cap.fv, std::min(fast_fv_n, g_cap.fv_n)));
    cmp(kFastCmpFloatP,
        fast_fp_n == g_cap.fp_n && !std::memcmp(g_fast_fp, g_cap.fp, fast_fp_n), fast_fp_n,
        g_cap.fp_n, FirstDiffDword(g_fast_fp, g_cap.fp, std::min(fast_fp_n, g_cap.fp_n)));
    cmp(kFastCmpBoolLoop, !std::memcmp(g_fast_bl, g_cap.bl, sizeof(g_fast_bl)), 0, 0,
        FirstDiffDword(g_fast_bl, g_cap.bl, sizeof(g_fast_bl)));
    cmp(kFastCmpFetch, !std::memcmp(g_fast_ft, g_cap.ft, sizeof(g_fast_ft)), 0, 0,
        FirstDiffDword(g_fast_ft, g_cap.ft, sizeof(g_fast_ft)));
    if (!drawable || !g_cap.emitted) {
      ++g_cmp.skip_emit;
      return;
    }
    bool npso_build = false, npso_bind = false;
    NrFastNativePsoFlags(&npso_build, &npso_bind);
    uint32_t fast_desc[16];
    const bool fast_desc_valid = pipeline_cache_->NrCopyPipelineDescription(plan.pso, fast_desc);
    if (fast_desc_valid && g_cap.pso_desc_valid) {
      const uint32_t diff = FirstDiffDword(fast_desc, g_cap.pso_desc, sizeof(fast_desc));
      if (diff == UINT32_MAX) ++g_cmp.pso_desc_eq;
      else ++g_cmp.pso_desc_ne;
      if (diff != UINT32_MAX && g_cmp.logged_pso_description < 2) {
        ++g_cmp.logged_pso_description;
        uint64_t fvs, fps, vvs, vps;
        std::memcpy(&fvs, fast_desc + 2, 8);
        std::memcpy(&fps, fast_desc + 6, 8);
        std::memcpy(&vvs, g_cap.pso_desc + 2, 8);
        std::memcpy(&vps, g_cap.pso_desc + 6, 8);
        REXGPU_WARN("[jade-native-cmp] pso_description first_dword={} fast={:08X} v1={:08X} "
                    "mod_vs={:016X}/{:016X} mod_ps={:016X}/{:016X} handles={:#x}/{:#x}",
                    diff, fast_desc[diff], g_cap.pso_desc[diff], fvs, vvs, fps, vps,
                    uintptr_t(plan.pso), uintptr_t(g_cap.pso_description_handle));
      }
    } else {
      ++g_cmp.pso_desc_unavailable;
    }
    if (plan.npso && npso_bind) {
      cmp(kFastCmpPso, g_cap.ext_pso == plan.npso, uint32_t(uintptr_t(plan.npso)),
          uint32_t(uintptr_t(g_cap.ext_pso)), 1);
    } else {
      cmp(kFastCmpPso, g_cap.guest_pso == plan.pso && !g_cap.ext_pso,
          uint32_t(uintptr_t(plan.pso)), uint32_t(uintptr_t(g_cap.guest_pso)), 0);
    }
    cmp(kFastCmpRootSig, g_cap.rootsig == plan.rootsig, uint32_t(uintptr_t(plan.rootsig)),
        uint32_t(uintptr_t(g_cap.rootsig)), 0);
    {
      bool teq = g_cap.tex_n == plan.tex_count;
      uint32_t bad = UINT32_MAX;
      for (uint32_t i = 0; teq && i < plan.tex_count; ++i) {
        if (!TexSnapEq(plan.tex[i], g_cap.tex[i])) {
          teq = false;
          bad = plan.tex[i].slot;
        }
      }
      cmp(kFastCmpTex, teq, plan.tex_count, g_cap.tex_n, bad);
    }
    cmp(kFastCmpViewport, !std::memcmp(&plan.viewport, &g_cap.vp, sizeof(plan.viewport)), 0, 0,
        FirstDiffDword(&plan.viewport, &g_cap.vp, sizeof(plan.viewport)));
    if (plan.hit && plan.entry->blob_ok) {
      const uint32_t *v = nullptr, *p = nullptr;
      uint32_t nv = 0, np = 0;
      bool bv = false, bp = false;
      NrFastBoundPacks(plan.entry->blob, &v, &nv, &p, &np, &bv, &bp);
      if (bv) {
        cmp(kFastCmpDiV,
            g_cap.bv && nv == g_cap.nv && !std::memcmp(v, g_cap.div, nv * sizeof(uint32_t)), nv,
            g_cap.nv, FirstDiffDword(v, g_cap.div, std::min(nv, g_cap.nv) * 4));
      }
      if (bp) {
        cmp(kFastCmpDiP,
            g_cap.bp && np == g_cap.np && !std::memcmp(p, g_cap.dip, np * sizeof(uint32_t)), np,
            g_cap.np, FirstDiffDword(p, g_cap.dip, std::min(np, g_cap.np) * 4));
      }
    } else {
      ++g_cmp.no_packs;
    }
    // A miss becomes an entry from this derivation, with v1's bindings (the
    // state NrUpdateBindings left for exactly this draw), so the next marker
    // of the object exercises the hit path under the compare.
    if (!plan.hit && all_eq) NrFastRecordEntry(plan, /*bindings_from_v1=*/true);
    return;
  }

  // ── Fast: emit. ──
  if (plan.kind == kFastKindNoDraw) {
    ++g_stats.nodraw;
    NrNativeBump(&NrNativeCounters::drawn);  // IssueDrawImpl returns true here
    if (!plan.hit && plan.record_nodraw) NrFastRecordEntry(plan, false);
    restore();
    return;
  }
  const bool ok = NrFastEmit(plan);
  if (ok) {
    NrNativeBump(&NrNativeCounters::drawn);
    NrNativeBump(&NrNativeCounters::fast_draws);
    if (plan.direct) ++g_direct.draws;
    if (!plan.hit) NrFastRecordEntry(plan, false);
  } else {
    NrNativeBump(&NrNativeCounters::refused_draw);
  }
  // Lean: a float upload composed with record constants holds values the live
  // file never had; the next draw composes its own.
  if (plan.lean_floats & 1u) cbuffer_binding_float_vertex_.up_to_date = false;
  if (plan.lean_floats & 2u) cbuffer_binding_float_pixel_.up_to_date = false;
  // Direct: the system constants upload holds the entry's block (the mirror
  // was written from it, not derived); a following retail IssueDrawImpl
  // derives and uploads its own, a following direct hit goes by its latch.
  if (plan.direct) cbuffer_binding_system_.up_to_date = false;
  // Step 2 bind: the descriptor-indices stages the blob bound are marked
  // stale and the bound-pack mirror unknown (the sampler members, layout
  // uids and SRV keys were not restored); the next direct hit goes by its
  // latch, the next retail IssueDrawImpl derives and rewrites them.
  if (plan.direct_bind && plan.entry) {
    const uint32_t *dv = nullptr, *dp = nullptr;
    uint32_t nv = 0, np = 0;
    bool bv = false, bp = false;
    NrFastBoundPacks(plan.entry->blob, &dv, &nv, &dp, &np, &bv, &bp);
    NrFastDiPostDraw(bv, bp);
  }
  // (d) post-draw dirty set of a skipped overlay, exactly what the restore's
  // invalidation would have raised: the fetch constant buffer when a record
  // fetch constant differed from the live file, the bool/loop buffer when a
  // record bool or loop constant differed, TextureFetchConstantWritten for
  // every differing texture slot (the next request re-derives from the live
  // file), InvalidateVertexBufferResidency for every differing vertex fetch
  // slot (the next draw re-requests its own range). The float stages follow
  // the lean rule above, the system block the direct rule above.
  if (plan.overlay_skip) NrFastInvalidate(plan.dry);
  restore();
}

// ── Derivation ──────────────────────────────────────────────────────────────
// NrFastDerive returns a fallback reason (the v1 body then draws the marker)
// or 0. On 0 the plan is either drawable, a no-draw, or failed (a refused draw).
// The raw overlay, applied once per marker on demand: before a miss
// derivation, for every non-direct draw, and for a direct hit whose render
// target latch missed (Update reads the live file). Idempotent.
void D3D12CommandProcessor::NrFastApplyOverlay(NrFastPlan& plan) {
  if (plan.overlay) return;
  ftrace::SplitScope _ft_derive(&ftrace::SplitTotals::nr_derive_ticks);
  FastProbeSpan span(ftrace::g_exec_split);
  FastChanges ch;
  // Lean: the float constants stay out of the live file (composed from the
  // record at the const stage), so the save list holds registers only.
  FastOverlayApply(register_file_->values, uint32_t(RegisterFile::kRegisterCount), *plan.rec, &g_dr,
                   g_saved, g_ost, ch, /*constants=*/!plan.lean);
  NrFastInvalidate(ch);
  plan.overlay = true;
  if (plan.overlay_skip) {
    // A late overlay on a draw that had skipped it: the pre-draw dirty set
    // was already raised and the post-draw one is still owed, both idempotent.
    ++g_direct.overlay_live_rt;
    --g_direct.overlay_skip;
    plan.overlay_skip = false;
  }
  span.Lap(g_derive.ovl);
}

void D3D12CommandProcessor::NrFastShutdown() {
  if (g_shm_watch && shared_memory_) shared_memory_->UnregisterGlobalWatch(g_shm_watch);
  g_shm_watch = nullptr;
}

// The object lookup: the key over the overlaid file, or over the record map
// (plan.mapped: the live file with the record's registers read through
// FastRegMap, the same words the overlay would have produced).
void D3D12CommandProcessor::NrFastLookup(NrFastPlan& plan) {
  const RegisterFile& regs = *register_file_;
  const NrNativeDrawRecord& rec = *plan.rec;
  D3D12Shader* vs = plan.vs;
  ftrace::SplitScope _ft_lookup(&ftrace::SplitTotals::nr_lookup_ticks);
  FastKeyInputs ki;
  ki.rf = regs.values;
  NrFastDcRanges(&ki.ranges, &ki.range_count);
  ki.vs_hash = vs->ucode_data_hash();
  ki.ps_hash = plan.ps_record ? plan.ps_record->ucode_data_hash() : 0;
  ki.prim_type = rec.prim_type;
  ki.index_count = rec.index_count;
  ki.flags = rec.flags;
  ki.index_base = rec.index_base;
  ki.index_endian = rec.index_endian;
  ki.index_length = rec.index_length;
  ki.tex_mask = vs->GetUsedTextureMaskAfterTranslation() |
                (plan.ps_record ? plan.ps_record->GetUsedTextureMaskAfterTranslation() : 0u);
  ki.variant = ia_phase_ | (uint32_t(bindless_resources_used_) << 1);
  const bool mapped = plan.mapped;
  const uint32_t* rf = regs.values;
  auto hash_key = [&](const FastKeyInputs& in) {
    const uint32_t n = std::min(
        mapped ? FastComposeObjectKeyRead(
                     in, [rf](uint32_t i) { return g_map.Read(rf, i); }, g_key_words,
                     kFastKeyMaxWords)
               : FastComposeObjectKey(in, g_key_words, kFastKeyMaxWords),
        kFastKeyMaxWords);
    const XXH128_hash_t h = XXH3_128bits(g_key_words, size_t(n) * sizeof(uint32_t));
    FastKey128 k;
    k.h1 = h.low64;
    k.h2 = h.high64;
    return k;
  };
  plan.key = hash_key(ki);
  plan.key_tex_mask = ki.tex_mask;
  plan.key_inputs = ki;
  FastEntry* e = g_objects.Find(plan.key, g_stamp);
  bool valid = e && e->gen == nr_fast_gen_ &&
               e->tex_gen == texture_cache_->nr_texture_destroy_generation() &&
               e->smp_heap == sampler_bindless_heap_gpu_start_.ptr &&
               e->view_heap == view_bindless_heap_gpu_start_.ptr && e->vs == vs;
  if (valid && plan.dmode && e->direct_ok && e->env != nr::NrDirectEnvStamp()) {
    // Step 4: the head was baked under another environment stamp (a writer
    // in nr_direct_env.h moved it): the entry is dropped so the miss
    // derivation re-bakes it under the current one.
    ++g_direct.env_refresh;
    e->valid = false;
    valid = false;
  }
  plan.entry = valid ? e : nullptr;
  plan.hit = valid;
  if (valid && e->direct_ok) {
    plan.direct_env_ok = true;
    if (plan.dmode == 2) {
      // A matching stamp must mean matching fields; a differing field names
      // an environment writer the stamp does not cover.
      const FastEnvFields live = NrFastEnvFields();
      const uint32_t d = FastEnvFirstDiff(e->env_fields, live);
      DirectCount(kFastDirectEnv, d == UINT32_MAX, rec, e->env, nr::NrDirectEnvStamp(), d);
    }
  }
}

FastEnvFields D3D12CommandProcessor::NrFastEnvFields() const {
  FastEnvFields f;
  const bool host_rt =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
  f.path = uint32_t(render_target_cache_->GetPath());
  f.convert_z = host_rt && render_target_cache_->depth_float24_convert_in_pixel_shader() ? 1u : 0u;
  f.gamma16 = render_target_cache_->gamma_render_target_as_unorm16() ? 1u : 0u;
  f.fixed16 = render_target_cache_->IsFixed16TruncatedToMinus1To1() ? 1u : 0u;
  bool npso_build = false, npso_bind = false;
  NrFastNativePsoFlags(&npso_build, &npso_bind);
  f.pso_bind = npso_bind ? 1u : 0u;
  f.scale_x = texture_cache_->draw_resolution_scale_x();
  f.scale_y = texture_cache_->draw_resolution_scale_y();
  f.widen = nr::DetileWidenStamp();
  f.shm_gpu = uint64_t(shared_memory_->GetGPUAddress());
  return f;
}

uint32_t D3D12CommandProcessor::NrFastDerive(NrFastPlan& plan) {
  const RegisterFile& regs = *register_file_;
  const NrNativeDrawRecord& rec = *plan.rec;
  D3D12Shader* vs = plan.vs;
  const uint32_t* rf = regs.values;
  FastEntry* e = plan.entry;
  const bool valid = plan.hit;
  plan.hit = false;
  if (valid && e->kind == kFastKindDelegate) {
    ++g_stats.delegate_hits;
    plan.kind = kFastKindDelegate;
    return e->reason;
  }
  // Vertex fetch constants: through the map when the record is mapped (the
  // overlay may be skipped), else the overlaid file.
  auto vfetch = [&](uint32_t index) {
    xenos::xe_gpu_vertex_fetch_t vf;
    if (plan.mapped) {
      vf.dword_0 = g_map.Read(rf, kFastRegFetchBase + 2 * index);
      vf.dword_1 = g_map.Read(rf, kFastRegFetchBase + 2 * index + 1);
    } else {
      vf = regs.GetVertexFetch(index);
    }
    return vf;
  };
  const bool dcmp = plan.dmode == 2 && valid && e->direct_ok && plan.direct_env_ok;
  ftrace::SplitScope _ft_derive(&ftrace::SplitTotals::nr_derive_ticks);
  FastProbeSpan span(ftrace::g_exec_split);  // [jade-native-derive] tail laps

  // IssueDrawImpl opens the submission after its no-draw returns and before
  // primitive processing; both paths below do the same.
  auto begin = [&]() {
    if (!BeginSubmission(true)) return false;
    GpuCensusSetClass(gpu_census_draw_class_);
    return true;
  };

  if (valid) {
    // ── Hit ──
    plan.entry = e;
    plan.hit = true;
    plan.kind = e->kind;
    plan.ps = e->ps;
    plan.polygonal = e->polygonal != 0;
    plan.ndc.value = e->ndc;
    plan.color_mask = e->color_mask;
    if (plan.kind == kFastKindNoDraw) {
      NrNativeBump(&NrNativeCounters::entry_hit);
      return 0;
    }
    if (!begin()) {
      plan.failed = true;
      return 0;
    }
    span.Lap(g_derive.sub);
    plan.prim = e->prim;
    plan.ib_req_base = e->ib_req_base;
    plan.ib_req_size = e->ib_req_size;
    plan.topology = e->topology;
    plan.ia = e->ia != 0;
    plan.pso = e->pso;
    plan.rootsig = e->rootsig;
    plan.npso = e->npso;
    plan.viewport = e->viewport;
    plan.tex_mask = e->tex_mask;
    plan.tex_count = e->tex_count;
    for (uint32_t i = 0; i < e->tex_count; ++i) plan.tex[i] = e->tex[i];
    // The index residency the skipped Process would have requested. (c) The
    // entry latches a successful request under the shared memory
    // invalidation generation (the executor's global watch: CPU writes and
    // GPU writes to requested pages, resolves included); residency only
    // moves through an invalidation, so an equal generation means the pages
    // are still valid and allocated.
    if (plan.prim.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA) {
      const uint32_t shm_gen = g_shm_gen.load(std::memory_order_relaxed);
      const bool ib_latched = e->ib_latched != 0 && e->shm_gen == shm_gen;
      if (plan.dmode == 1 && ib_latched && plan.direct_env_ok) {
        plan.ib_skip = true;
        ++g_direct.ib_skip;
      } else {
        const bool ok = shared_memory_->RequestRange(plan.ib_req_base, plan.ib_req_size);
        if (dcmp && ib_latched) DirectCount(kFastDirectIb, ok, rec, plan.ib_req_base, ok, shm_gen);
        if (!ok) return kFastFbProcess;
        e->ib_latched = 1;
        e->shm_gen = shm_gen;
      }
    }
    span.Lap(g_derive.ib);
    // The IA views are live (vertex buffers move); an IA entry that cannot
    // acquire them now cannot draw with its IA pipeline.
    if (plan.ia) {
      const std::vector<Shader::VertexBinding>& vb = vs->vertex_bindings();
      for (size_t b = 0; b < vb.size() && b < kFastIaMax; ++b) {
        const xenos::xe_gpu_vertex_fetch_t vf = vfetch(vb[b].fetch_constant);
        if (vf.type != xenos::FetchConstantType::kVertex || vf.size == 0 ||
            !vb_mirror_->Acquire(vf.address << 2, vf.size << 2, uint32_t(vf.endian),
                                 &plan.ia_vb_va[b])) {
          return kFastFbIa;
        }
      }
      if (plan.prim.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
          !vb_mirror_->Acquire(plan.prim.guest_index_base, plan.ib_req_size,
                               uint32_t(plan.prim.host_shader_index_endian), &plan.ia_ib_va)) {
        return kFastFbIa;
      }
    }
    span.Lap(g_derive.ia);
    // (a) The render target cache: Update reads sixteen registers (all
    // identity-keyed: RB_SURFACE_INFO, RB_DEPTH_INFO, RB_COLOR0..3_INFO,
    // PA_SC_WINDOW_OFFSET / SCISSOR_TL / SCISSOR_BR, PA_SC_SCREEN_SCISSOR_BR,
    // PA_CL_CLIP_CNTL, PA_CL_VTE_CNTL, PA_SU_SC_MODE_CNTL, PA_SU_VTX_CNTL,
    // PA_CL_VPORT_YSCALE / YOFFSET) plus the entry's ndc and color mask, the
    // path (env stamp) and its own ownership generation. The entry latches
    // the cache's nr_update_serial after its Update: an equal serial means no
    // body ran and no ownership moved since, so the entry guard would skip
    // the body; only the per-draw tail runs (xfer note, latched binding).
    const uint64_t rt_serial = render_target_cache_->nr_update_serial();
    const bool rt_latched = e->rt_serial == rt_serial && e->direct_ok != 0;
    if (plan.dmode == 1 && rt_latched && plan.direct_env_ok) {
      plan.rt_skip = true;
      ++g_direct.rt_skip;
      render_target_cache_->NrDirectUpdateTail(e->xfer_reads, e->xfer_reason);
      NrNativeBump(&NrNativeCounters::rt_update_skipped);
    } else {
      NrFastApplyOverlay(plan);  // Update reads the live file (a no-op when applied)
      uintptr_t ids0[1 + xenos::kMaxColorRenderTargets] = {};
      const bool ids0_valid = dcmp && render_target_cache_->NrBoundRenderTargetIdentities(ids0);
      const uint64_t rt_runs0 = render_target_cache_->nr_update_body_runs();
      if (!render_target_cache_->Update(true, plan.ndc, plan.color_mask, *vs)) {
        plan.failed = true;
        return 0;
      }
      const bool body_ran = render_target_cache_->nr_update_body_runs() != rt_runs0;
      if (!body_ran) NrNativeBump(&NrNativeCounters::rt_update_skipped);
      if (dcmp && rt_latched) {
        // The latch said skip: the body must not have run and the bound
        // identities must be what the tail would have rebound.
        uintptr_t ids1[1 + xenos::kMaxColorRenderTargets] = {};
        const bool ids1_valid = render_target_cache_->NrBoundRenderTargetIdentities(ids1);
        const bool same = !ids0_valid || (ids1_valid && !std::memcmp(ids0, ids1, sizeof(ids0)));
        DirectCount(kFastDirectRt, !body_ran && same, rec, body_ran, same, 0);
      }
      if (dcmp) {
        // The baked verdict vs the SDK bitfield derivation over the live file.
        uint32_t blend[4], reads = 0;
        uint8_t reasons[5];
        for (uint32_t i = 0; i < 4; ++i) blend[i] = rf[reg::RB_BLENDCONTROL::rt_register_indices[i]];
        const uint32_t used = FastXferVerdict(plan.ndc.value, plan.color_mask, blend, &reads, reasons);
        uint32_t sdk_used = 0, sdk_reads = 0;
        {
          const reg::RB_DEPTHCONTROL dc = plan.ndc;
          auto op_writes = [](xenos::StencilOp op) {
            return op == xenos::StencilOp::kReplace || op == xenos::StencilOp::kZero;
          };
          if (dc.z_enable || dc.stencil_enable) {
            sdk_used |= 1u;
            const bool z_reads = dc.z_enable && dc.zfunc != xenos::CompareFunction::kAlways;
            bool stencil_reads = false;
            if (dc.stencil_enable) {
              const bool front_writes = dc.stencilfunc == xenos::CompareFunction::kAlways &&
                                        op_writes(dc.stencilzpass) &&
                                        (!z_reads || op_writes(dc.stencilzfail));
              const bool back_writes = !dc.backface_enable ||
                                       (dc.stencilfunc_bf == xenos::CompareFunction::kAlways &&
                                        op_writes(dc.stencilzpass_bf) &&
                                        (!z_reads || op_writes(dc.stencilzfail_bf)));
              stencil_reads = !(front_writes && back_writes);
            }
            if (z_reads || stencil_reads) sdk_reads |= 1u;
          }
          auto factor_reads_dest = [](xenos::BlendFactor f) {
            return f == xenos::BlendFactor::kDstColor || f == xenos::BlendFactor::kOneMinusDstColor ||
                   f == xenos::BlendFactor::kDstAlpha || f == xenos::BlendFactor::kOneMinusDstAlpha ||
                   f == xenos::BlendFactor::kSrcAlphaSaturate;
          };
          for (uint32_t i = 0; i < 4; ++i) {
            if (!((plan.color_mask >> (i * 4)) & 0xF)) continue;
            sdk_used |= 1u << (1 + i);
            reg::RB_BLENDCONTROL bc;
            bc.value = blend[i];
            const bool blend_reads =
                factor_reads_dest(bc.color_srcblend) ||
                bc.color_destblend != xenos::BlendFactor::kZero ||
                bc.color_comb_fcn == xenos::BlendOp::kMin || bc.color_comb_fcn == xenos::BlendOp::kMax ||
                factor_reads_dest(bc.alpha_srcblend) ||
                bc.alpha_destblend != xenos::BlendFactor::kZero ||
                bc.alpha_comb_fcn == xenos::BlendOp::kMin || bc.alpha_comb_fcn == xenos::BlendOp::kMax;
            const bool mask_reads = ((plan.color_mask >> (i * 4)) & 0xF) != 0xF;
            if (blend_reads || mask_reads) sdk_reads |= 1u << (1 + i);
          }
        }
        const bool xfer_ok = used == sdk_used && reads == sdk_reads && used == e->xfer_used &&
                             reads == e->xfer_reads && !std::memcmp(reasons, e->xfer_reason, 5);
        DirectCount(kFastDirectXfer, xfer_ok, rec, (uint64_t(used) << 8) | reads,
                    (uint64_t(sdk_used) << 8) | sdk_reads, (uint32_t(e->xfer_used) << 8) | e->xfer_reads);
      }
      e->rt_serial = render_target_cache_->nr_update_serial();
    }
    span.Lap(g_derive.rt);
    // The pipeline was built for the bound render target formats of the
    // recording; if the RT cache now binds others, re-derive the object.
    bool rt_same = true;
    if (render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets) {
      uint32_t formats[5] = {};
      const uint32_t bits = render_target_cache_->GetLastUpdateBoundRenderTargets(formats);
      rt_same = bits == e->rt_bits;
      for (uint32_t i = 0; rt_same && i < 5; ++i) {
        if ((bits & (1u << i)) && formats[i] != e->rt_formats[i]) rt_same = false;
      }
    }
    // (d) NrTexRestore validates the snapshots against the live fetch
    // constants; with the overlay skipped, the record's used texture slots
    // whose fetch constants differ from the live file are overlaid sparsely
    // for the restore and put back when this function returns (nothing else
    // below reads them). The probe splits the price (tex: restore + ensure
    // loaded, txb: barriers).
    struct SparseGuard {
      FastSparseTexOverlay o;
      uint32_t* rf;
      ~SparseGuard() { o.Restore(rf); }
    } sparse{{}, register_file_->values};
    if (rt_same) {
      const uint32_t tn = uint32_t(__builtin_popcount(plan.tex_mask));
      // (b) The texture binding generation latch (N4 step 3): the entry
      // latched the cache's binding generation after its last successful
      // restore + ensure-loaded, when every used slot held the entry's
      // snapshot and every bound texture was up to date. The generation
      // moves on every binding content write, reset, texture memory watch
      // and scaled-range change (nr_binding_generation's writer list), so
      // an equal generation with no watch pending means the slots still
      // hold the snapshots and their data is current: the restore and the
      // loads are skipped. The barriers stay live by decision:
      // NrTexBarriersOnly marks the textures used (the eviction LRU) and
      // asserts the resource state a resolve or load may have moved.
      const uint32_t bgen = texture_cache_->nr_binding_generation();
      const bool outdated = texture_cache_->NrTexOutdatedPending();
      const bool tex_latch =
          plan.tex_count != 0 && e->tex_latched != 0 && e->tex_bgen == bgen && !outdated;
      if (plan.dmode == 1 && plan.direct_env_ok && tex_latch) {
        ++g_direct.tex_skip;
        NrNativeBump(&NrNativeCounters::entry_hit);
        NrNativeBump(&NrNativeCounters::pso_hit);
        span.Lap(g_derive.tex);
        texture_cache_->NrTexBarriersOnly(plan.tex_mask);
        NrNativeBump(&NrNativeCounters::tex_hit, tn);
        span.Lap(g_derive.txb);
        return 0;
      }
      if (dcmp && tex_latch) {
        // The skip's claim against the live state, before the restore runs:
        // every used slot already equals the snapshot (content and
        // descriptor indices) and its textures are up to date.
        uint32_t bad = UINT32_MAX;
        const bool ok = texture_cache_->NrTexLatchCheck(plan.tex, plan.tex_count, &bad);
        DirectCount(kFastDirectTxl, ok, rec, bgen, e->tex_bgen, bad);
      }
      if (plan.dmode == 1 && plan.tex_count) {
        ++g_direct.tex_live;
        if (e->tex_latched && e->tex_bgen != bgen) ++g_direct.tex_gen_miss;
        if (outdated) ++g_direct.tex_outdated;
      }
      // Texture bindings from the entry (refused while a watch is pending);
      // reload checks and transitions stay live.
      if (plan.overlay_skip) sparse.o.Apply(register_file_->values, g_map, plan.dry.tex_fetch & plan.tex_mask);
      if (plan.tex_count == 0 || texture_cache_->NrTexRestore(plan.tex, plan.tex_count)) {
        NrNativeBump(&NrNativeCounters::entry_hit);
        NrNativeBump(&NrNativeCounters::pso_hit);
        bool loaded = true;
        if (plan.tex_mask) {
          loaded = texture_cache_->NrTexEnsureLoaded(plan.tex_mask);
          span.Lap(g_derive.tex);
          texture_cache_->NrTexBarriersOnly(plan.tex_mask);
        }
        // The latch: the generation after the restore's own writes, only
        // with every texture loaded and no watch pending.
        e->tex_latched = 0;
        if (plan.tex_count && loaded && !texture_cache_->NrTexOutdatedPending()) {
          e->tex_bgen = texture_cache_->nr_binding_generation();
          e->tex_latched = 1;
        }
        NrNativeBump(&NrNativeCounters::tex_hit, tn);
        span.Lap(g_derive.txb);
        return 0;
      } else {
        ++g_stats.tex_restore_fail;
        // A refreshed texture may have a new SRV descriptor index. The
        // binding blob caches those indices, so replacing only the texture
        // snapshot would leave stale GPU bindings. Rebuild both together.
      }
    } else {
      ++g_stats.rt_rederive;
    }
    e->valid = false;  // re-recorded below
    plan.hit = false;
    plan.entry = nullptr;
  }

  // ── Miss: the IssueDrawImpl stages, in its order, on the overlaid file. ──
  span.tail = &g_derive.miss;  // the whole miss derivation is one lump
  NrFastApplyOverlay(plan);
  NrNativeBump(&NrNativeCounters::entry_miss);
  plan.hit = false;
  plan.kind = kFastKindDraw;
  pipeline_cache_->AnalyzeShaderUcode(*vs);
  if (vs->memexport_eM_written()) return kFastFbMemexport;
  const xenos::EdramMode edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (edram_mode == xenos::EdramMode::kCopy) return kFastFbEdram;
  const bool surface_pitch_is_zero = regs.Get<reg::RB_SURFACE_INFO>().surface_pitch == 0;
  plan.polygonal = draw_util::IsPrimitivePolygonal(regs);
  const bool rast = draw_util::IsRasterizationPotentiallyDone(regs, plan.polygonal);
  if (surface_pitch_is_zero && rast) {
    plan.kind = kFastKindNoDraw;
    plan.record_nodraw = true;
    return 0;
  }
  D3D12Shader* ps = nullptr;
  if (rast) {
    if (edram_mode == xenos::EdramMode::kColorDepth) {
      ps = plan.ps_record;
      if (ps) {
        pipeline_cache_->AnalyzeShaderUcode(*ps);
        if (!draw_util::IsPixelShaderNeededWithRasterization(*ps, regs)) ps = nullptr;
      }
    }
  } else {
    plan.kind = kFastKindNoDraw;  // no memexport (refused above): no effect
    plan.record_nodraw = true;
    return 0;
  }
  if (ps && ps->memexport_eM_written()) return kFastFbMemexport;
  plan.ps = ps;
  if (!begin()) {
    plan.failed = true;
    return 0;
  }
  // Primitive processing (reads the overlaid VGT draw registers).
  if (!primitive_processor_->Process(plan.prim)) return kFastFbProcess;
  const PrimitiveProcessor::ProcessingResult& pr = plan.prim;
  if (pr.IsTessellated() || pr.host_vertex_shader_type != Shader::HostVertexShaderType::kVertex) {
    return kFastFbHostVs;
  }
  if (pr.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted) {
    return kFastFbConverted;
  }
  if (pr.line_loop_closing_index != 0 ||
      xenos::PrimitiveType(rec.prim_type) == xenos::PrimitiveType::kLineLoop) {
    return kFastFbLineLoop;
  }
  if (!pr.host_draw_vertex_count) {
    plan.kind = kFastKindNoDraw;
    plan.record_nodraw = true;
    return 0;
  }
  if (!HostTopology(pr.host_primitive_type, &plan.topology)) return kFastFbHostVs;
  plan.ib_req_base = pr.guest_index_base;
  plan.ib_req_size =
      pr.host_draw_vertex_count * (pr.host_index_format == xenos::IndexFormat::kInt16 ? 2u : 4u);
  plan.ndc = draw_util::GetNormalizedDepthControl(regs);

  // [vtl] gate (IssueDrawImpl's, without its census).
  bool vtl_draw = false;
  uint32_t vtl_pack = 0;
  {
    const std::vector<Shader::TextureBinding>& tbs = vs->texture_bindings();
    if (!tbs.empty()) {
      vtl_draw = true;
      uint32_t why = 0;
      for (const Shader::TextureBinding& tb : tbs) {
        if (tb.fetch_instr.dimension != xenos::FetchOpDimension::k1D) continue;
        const xenos::xe_gpu_texture_fetch_t tf = regs.GetTextureFetch(tb.fetch_constant);
        const uint32_t pack = tf.dimension == xenos::DataDimension::k1D            ? 1u
                              : tf.dimension == xenos::DataDimension::k2DOrStacked ? 2u
                                                                                   : 0u;
        if (tf.type != xenos::FetchConstantType::kTexture) {
          why = 1;
        } else if (pack == 0 || (vtl_pack != 0 && vtl_pack != pack)) {
          why = 7;
        } else if (!FastSignsOk(tf)) {
          why = 2;
        } else if (tf.mip_min_level != 0 || tf.mip_max_level != 0) {
          why = 3;
        } else if (tf.exp_adjust != 0) {
          why = 4;
        } else if ((tb.fetch_instr.attributes.mag_filter == xenos::TextureFilter::kUseFetchConst &&
                    tf.mag_filter != xenos::TextureFilter::kPoint) ||
                   (tb.fetch_instr.attributes.min_filter == xenos::TextureFilter::kUseFetchConst &&
                    tf.min_filter != xenos::TextureFilter::kPoint)) {
          why = 5;
        } else if (texture_cache_->IsActiveTextureResolutionScaled(tb.fetch_constant)) {
          why = 6;
        }
        if (why) break;
        vtl_pack = pack;
      }
      if (!why && vtl_pack == 0) why = 7;
      if (why) vtl_draw = false;
    }
  }
  // [ia] decision (the pool / instanced branches never apply: the fast draw
  // opens no batch).
  plan.ia = false;
  plan.ia_ib_va = 0;
  {
    const std::vector<Shader::VertexBinding>& vb = vs->vertex_bindings();
    if (ia_available_ && ia_phase_ != 0 && vs->pos_path().ia_eligible && !vb.empty() &&
        vb.size() <= kFastIaMax) {
      plan.ia = true;
      for (size_t b = 0; b < vb.size(); ++b) {
        const xenos::xe_gpu_vertex_fetch_t vf = regs.GetVertexFetch(vb[b].fetch_constant);
        if (vf.type != xenos::FetchConstantType::kVertex || vf.size == 0 ||
            !vb_mirror_->Acquire(vf.address << 2, vf.size << 2, uint32_t(vf.endian),
                                 &plan.ia_vb_va[b])) {
          plan.ia = false;
          break;
        }
      }
      if (plan.ia && pr.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
          !vb_mirror_->Acquire(pr.guest_index_base, plan.ib_req_size,
                               uint32_t(pr.host_shader_index_endian), &plan.ia_ib_va)) {
        plan.ia = false;
        plan.ia_ib_va = 0;
      }
    }
  }
  // Shader modifications.
  uint32_t ps_param_gen_pos = UINT32_MAX;
  const uint32_t interpolator_mask =
      ps ? (vs->writes_interpolators() &
            ps->GetInterpolatorInputMask(regs.Get<reg::SQ_PROGRAM_CNTL>(),
                                         regs.Get<reg::SQ_CONTEXT_MISC>(), ps_param_gen_pos))
         : 0;
  DxbcShaderTranslator::Modification vmod = pipeline_cache_->GetCurrentVertexShaderModification(
      *vs, pr.host_vertex_shader_type, interpolator_mask, false);
  vmod.vertex.ia_fetch = plan.ia ? 1 : 0;
  vmod.vertex.tex_direct = vtl_draw ? vtl_pack : 0;
  DxbcShaderTranslator::Modification pmod =
      ps ? pipeline_cache_->GetCurrentPixelShaderModification(*ps, interpolator_mask,
                                                              ps_param_gen_pos, plan.ndc)
         : DxbcShaderTranslator::Modification(0);
  if (ps) {
    // [psf] gate (IssueDrawImpl's, without its census).
    bool psf_draw = false;
    const std::vector<Shader::TextureBinding>& tbs = ps->texture_bindings();
    if (!tbs.empty()) {
      psf_draw = true;
      for (const Shader::TextureBinding& tb : tbs) {
        if (tb.fetch_instr.dimension != xenos::FetchOpDimension::k2D) continue;
        const xenos::xe_gpu_texture_fetch_t tf = regs.GetTextureFetch(tb.fetch_constant);
        if (tf.type != xenos::FetchConstantType::kTexture || !FastSignsOk(tf) ||
            tf.exp_adjust != 0 || tf.lod_bias != 0) {
          psf_draw = false;
          break;
        }
      }
    }
    pmod.pixel.tex_direct = psf_draw ? 1 : 0;
  }
  plan.color_mask = ps ? draw_util::GetNormalizedColorMask(regs, ps->writes_color_targets()) : 0;
  // Render targets.
  const uint64_t rt_runs0 = render_target_cache_->nr_update_body_runs();
  if (!render_target_cache_->Update(rast, plan.ndc, plan.color_mask, *vs)) {
    plan.failed = true;
    return 0;
  }
  if (render_target_cache_->nr_update_body_runs() == rt_runs0) {
    NrNativeBump(&NrNativeCounters::rt_update_skipped);
  }
  // Pipeline: the PSO table, else ConfigurePipeline.
  auto* vt = static_cast<D3D12Shader::D3D12Translation*>(vs->GetOrCreateTranslation(vmod.value));
  auto* pt = ps ? static_cast<D3D12Shader::D3D12Translation*>(ps->GetOrCreateTranslation(pmod.value))
                : nullptr;
  const bool host_rt = render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
  plan.rt_bits = host_rt ? render_target_cache_->GetLastUpdateBoundRenderTargets(plan.rt_formats) : 0;
  FastPsoInputs pi;
  pi.rf = regs.values;
  pi.vs_translation = uint64_t(uintptr_t(vt));
  pi.ps_translation = uint64_t(uintptr_t(pt));
  pi.host_primitive_type = uint32_t(pr.host_primitive_type);
  pi.host_vertex_shader_type = uint32_t(pr.host_vertex_shader_type);
  pi.tessellation_mode = uint32_t(pr.tessellation_mode);
  pi.primitive_reset = pr.host_primitive_reset_enabled ? 1u : 0u;
  pi.host_index_format = uint32_t(pr.host_index_format);
  pi.normalized_depth_control = plan.ndc.value;
  pi.normalized_color_mask = plan.color_mask;
  pi.rt_bits = plan.rt_bits;
  std::memcpy(pi.rt_formats, plan.rt_formats, sizeof(pi.rt_formats));
  pi.path = uint32_t(render_target_cache_->GetPath());
  uint32_t pso_words[kFastPsoKeyWords];
  FastComposePsoKey(pi, pso_words);
  const uint64_t pso_hash = FastHash64(pso_words, kFastPsoKeyWords, 0x0A6B7C8D9E0F1A2Bull);
  FastPsoSlot& ps_slot = g_psos[uint32_t(pso_hash) & ((1u << kFastPsoBits) - 1)];
  bool npso_build = false, npso_bind = false;
  NrFastNativePsoFlags(&npso_build, &npso_bind);
  if (ps_slot.valid && ps_slot.gen == nr_fast_gen_ &&
      !std::memcmp(ps_slot.words, pso_words, sizeof(pso_words))) {
    plan.pso = ps_slot.pso;
    plan.rootsig = ps_slot.rootsig;
    ++g_stats.pso_table_hit;
    NrNativeBump(&NrNativeCounters::pso_hit);
  } else {
    if (!pipeline_cache_->ConfigurePipeline(vt, pt, pr, plan.ndc, plan.color_mask, plan.rt_bits,
                                            plan.rt_formats, &plan.pso, &plan.rootsig)) {
      plan.failed = true;
      return 0;
    }
    NrNativeBump(&NrNativeCounters::pso_miss);
    ps_slot.valid = false;  // filled after final pipeline/root-signature resolution
  }
  if (!pipeline_cache_->ResolvePipelineForDraw(plan.pso, &plan.rootsig)) {
    plan.failed = true;
    return 0;
  }
  if (!ps_slot.valid || ps_slot.gen != nr_fast_gen_ ||
      std::memcmp(ps_slot.words, pso_words, sizeof(pso_words))) {
    ps_slot.valid = true;
    ps_slot.gen = nr_fast_gen_;
    std::memcpy(ps_slot.words, pso_words, sizeof(pso_words));
    ps_slot.pso = plan.pso;
    ps_slot.rootsig = plan.rootsig;
    ps_slot.npso = nullptr;
    ps_slot.npso_done = false;
  }
  // Textures: per used slot from the texture table, else RequestTextures.
  plan.tex_mask = vs->GetUsedTextureMaskAfterTranslation() |
                  (ps ? ps->GetUsedTextureMaskAfterTranslation() : 0u);
  const uint32_t tn = uint32_t(__builtin_popcount(plan.tex_mask));
  if (tn > kFastTexMax) return kFastFbTexCount;
  {
    const uint32_t tex_gen = texture_cache_->nr_texture_destroy_generation();
    const uint64_t view_heap = view_bindless_heap_gpu_start_.ptr;
    bool all = true;
    uint32_t n = 0;
    uint32_t m = plan.tex_mask;
    while (m) {
      const uint32_t slot = uint32_t(__builtin_ctz(m));
      m &= m - 1;
      uint32_t fetch[6];
      std::memcpy(fetch, &regs.values[kFastRegFetchBase + 6 * slot], sizeof(fetch));
      const FastTexSlot& t = g_texs[uint32_t(FastTexKey(fetch)) & ((1u << kFastTexBits) - 1)];
      if (!t.valid || t.gen != nr_fast_gen_ || t.tex_gen != tex_gen || t.view_heap != view_heap ||
          std::memcmp(t.snap.fetch, fetch, sizeof(fetch))) {
        all = false;
        break;
      }
      plan.tex[n] = t.snap;
      plan.tex[n].slot = uint8_t(slot);
      ++n;
    }
    if (all && (n == 0 || texture_cache_->NrTexRestore(plan.tex, n))) {
      plan.tex_count = n;
      if (plan.tex_mask) {
        texture_cache_->NrTexEnsureLoaded(plan.tex_mask);
        texture_cache_->NrTexBarriersOnly(plan.tex_mask);
      }
      NrNativeBump(&NrNativeCounters::tex_hit, tn);
    } else {
      texture_cache_->RequestTextures(plan.tex_mask);
      NrNativeBump(&NrNativeCounters::tex_miss, tn);
      const uint32_t cn =
          plan.tex_mask ? texture_cache_->NrTexCapture(plan.tex_mask, plan.tex, kFastTexMax) : 0u;
      if (cn == UINT32_MAX) return kFastFbTexCount;
      plan.tex_count = cn;
      for (uint32_t i = 0; i < cn; ++i) {
        FastTexSlot& t = g_texs[uint32_t(FastTexKey(plan.tex[i].fetch)) & ((1u << kFastTexBits) - 1)];
        t.valid = true;
        t.gen = nr_fast_gen_;
        t.tex_gen = tex_gen;
        t.view_heap = view_heap;
        t.snap = plan.tex[i];
      }
    }
  }
  // The native pipeline object (built once per PSO slot).
  if (npso_build) {
    if (!ps_slot.npso_done) {
      ps_slot.npso = pipeline_cache_->NrNativePipeline(plan.pso, plan.rootsig);
      ps_slot.npso_done = true;
    }
    plan.npso = ps_slot.npso;
  } else {
    plan.npso = nullptr;
  }
  // Host viewport (the IssueDrawImpl key cache kept coherent).
  {
    const bool convert_z = host_rt && render_target_cache_->depth_float24_convert_in_pixel_shader();
    const bool ps_writes_depth = ps && ps->writes_depth();
    ViewportCacheKey vk;
    vk.pa_cl_clip_cntl = regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
    vk.pa_cl_vte_cntl = regs[XE_GPU_REG_PA_CL_VTE_CNTL];
    vk.pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
    vk.pa_su_vtx_cntl = regs[XE_GPU_REG_PA_SU_VTX_CNTL];
    vk.pa_sc_window_offset = regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
    vk.normalized_depth_control = plan.ndc.value;
    std::memcpy(vk.vport_regs, &regs[XE_GPU_REG_PA_CL_VPORT_XSCALE], sizeof(vk.vport_regs));
    vk.flags = (uint32_t(convert_z) << 0) | (uint32_t(host_rt) << 1) | (uint32_t(ps_writes_depth) << 2);
    if (viewport_cache_valid_ && vk == previous_viewport_key_) {
      plan.viewport = previous_viewport_info_;
    } else {
      draw_util::GetHostViewportInfo(regs, texture_cache_->draw_resolution_scale_x(),
                                     texture_cache_->draw_resolution_scale_y(), true,
                                     D3D12_VIEWPORT_BOUNDS_MAX, D3D12_VIEWPORT_BOUNDS_MAX, false,
                                     plan.ndc, convert_z, host_rt, ps_writes_depth, plan.viewport);
      previous_viewport_key_ = vk;
      previous_viewport_info_ = plan.viewport;
      viewport_cache_valid_ = true;
    }
  }
  // The key was taken with the texture masks as they were before this
  // derivation translated the shaders; record under the mask the next
  // lookup of this object will compute (same rule: the record's pixel
  // shader, before the pixel-shader-needed rule).
  const uint32_t lookup_mask =
      vs->GetUsedTextureMaskAfterTranslation() |
      (plan.ps_record ? plan.ps_record->GetUsedTextureMaskAfterTranslation() : 0u);
  if (lookup_mask != plan.key_tex_mask) {
    // Over the overlaid file (the miss path applied it above).
    plan.key_inputs.tex_mask = lookup_mask;
    plan.key_inputs.rf = regs.values;
    const uint32_t n = std::min(FastComposeObjectKey(plan.key_inputs, g_key_words, kFastKeyMaxWords),
                                kFastKeyMaxWords);
    const XXH128_hash_t h = XXH3_128bits(g_key_words, size_t(n) * sizeof(uint32_t));
    plan.key.h1 = h.low64;
    plan.key.h2 = h.high64;
    plan.key_tex_mask = lookup_mask;
  }
  return 0;
}

// ── Emission ────────────────────────────────────────────────────────────────
// IssueDrawImpl's tail for a derived plan, without its census blocks. False
// where IssueDrawImpl returns false.
bool D3D12CommandProcessor::NrFastEmit(NrFastPlan& plan) {
  const RegisterFile& regs = *register_file_;
  D3D12Shader* vs = plan.vs;
  D3D12Shader* ps = plan.ps;
  const PrimitiveProcessor::ProcessingResult& pr = plan.prim;
  // [ftrace] derive / const / bind / emit, one open at a time.
  std::optional<ftrace::SplitScope> ft;
  ft.emplace(&ftrace::SplitTotals::nr_derive_ticks);
  // [jade-native-derive] sub-stage laps (one clock read per lap, trace on only).
  const bool probe = ftrace::g_exec_split;
  std::optional<FastProbeSpan> span;
  span.emplace(probe);
  auto lap = [&](uint64_t& acc) { span->Lap(acc); };
  if (probe) ++g_derive.n;
  // Direct submission (gpu_native_direct): an owned hit whose entry baked its
  // head under the current environment stamp (nr_direct_env.h; the lookup
  // dropped it otherwise). Mode 1 emits through the head; mode 2 prepares
  // it, draws through the full path and diffs the classes.
  const int32_t dmode = plan.dmode;
  FastEntry* const e = plan.hit ? plan.entry : nullptr;
  const bool host_rt = render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
  const uint8_t env_convert_z =
      uint8_t(host_rt && render_target_cache_->depth_float24_convert_in_pixel_shader());
  const uint32_t sx = texture_cache_->draw_resolution_scale_x();
  const uint32_t sy = texture_cache_->draw_resolution_scale_y();
  const uint32_t* rf = regs.values;
  bool direct = false, cmp = false;
  if (dmode != 0 && e) {
    if (!e->direct_ok) {
      ++g_direct.nodirect;
    } else if (!plan.direct_env_ok) {
      ++g_direct.refused_env;
    } else {
      direct = dmode == 1 && plan.hit && e->direct_ok != 0;
      cmp = dmode == 2;
    }
  }
  plan.direct = direct;
  // Vertex fetch constants through the map when the record is mapped.
  auto vfetch = [&](uint32_t index) {
    xenos::xe_gpu_vertex_fetch_t vf;
    if (plan.mapped) {
      vf.dword_0 = g_map.Read(rf, kFastRegFetchBase + 2 * index);
      vf.dword_1 = g_map.Read(rf, kFastRegFetchBase + 2 * index + 1);
    } else {
      vf = regs.GetVertexFetch(index);
    }
    return vf;
  };
  // The blend factor is unkeyed (RB_BLEND_RED..ALPHA are outside the [dcache]
  // ranges): never baked, read live where the fixed-function state is set,
  // from the record's register list (the map) when the overlay may be absent.
  auto read_blend = [&](float out[4]) {
    if (plan.mapped) {
      for (uint32_t k = 0; k < 4; ++k) {
        out[k] = std::bit_cast<float>(g_map.Read(rf, XE_GPU_REG_RB_BLEND_RED + k));
      }
      return;
    }
    out[0] = regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    out[1] = regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    out[2] = regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    out[3] = regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
  };
  if (cmp && plan.tex_count) {
    // (b) compare: the restore reproduced the entry's slot bindings and
    // descriptor indices.
    TextureCache::NrTexSnap cap[kFastTexMax];
    const uint32_t n = texture_cache_->NrTexCapture(plan.tex_mask, cap, kFastTexMax);
    bool teq = n == plan.tex_count;
    uint32_t bad = UINT32_MAX;
    for (uint32_t i = 0; teq && i < n; ++i) {
      if (!TexSnapEq(cap[i], e->tex[i])) {
        teq = false;
        bad = cap[i].slot;
      }
    }
    DirectCount(kFastDirectTex, teq, *plan.rec, n, plan.tex_count, bad);
  }
  // Mode 2: the direct reads taken before the full path moves the state.
  float cmp_blend[4] = {};
  uint64_t cmp_pool_key = 0;
  bool cmp_pool_cls = false;
  if (cmp) {
    read_blend(cmp_blend);
    NrFastPoolBake(regs, &cmp_pool_key, &cmp_pool_cls);
  }
  // [hiz] the window survives only while depth can only move nearer; the
  // fast draw takes no Hi-Z test of its own.
  occ_draw_cull_ = 0;
  occ_draw_dbg_ = 0;
  hiz_draw_.valid = false;
  hiz_draw_.dir_hint = 0;
  if (hiz_phase_ != 0) {
    ++hiz_draws_seen_;
    if (hiz_window_.open) {
      if (direct) {
        // The direction is never baked: the entry's survival mask is read by
        // the open window's direction now.
        if (FastHizCloses(e->hiz_mono_mask, hiz_window_.dir)) HizWindowClose(3);
      } else {
        bool close = false;
        if (plan.ndc.z_enable && plan.ndc.z_write_enable) {
          const xenos::CompareFunction f = plan.ndc.zfunc;
          const bool mono =
              f == xenos::CompareFunction::kEqual || f == xenos::CompareFunction::kNever ||
              (hiz_window_.dir == 1 ? (f == xenos::CompareFunction::kGreater ||
                                       f == xenos::CompareFunction::kGreaterEqual)
                                    : (f == xenos::CompareFunction::kLess ||
                                       f == xenos::CompareFunction::kLessEqual));
          close = !mono;
        }
        if (cmp) {
          const bool baked = FastHizCloses(e->hiz_mono_mask, hiz_window_.dir);
          DirectCount(kFastDirectHiz, baked == close, *plan.rec, baked, close, hiz_window_.dir);
        }
        if (close) HizWindowClose(3);
      }
    }
  }
  lap(g_derive.hiz);
  // [DRAW-POOL] pass / class bookkeeping (the fast draw never joins a batch):
  // the entry's baked pass key and class through the same compares and
  // PoolClose calls, or the register-reading prologue.
  bool pool_on = false;
  bool pool_class_ok = false;
  if (direct) {
    pool_class_ok = NrFastPoolApply(e->pool_key, e->pool_cls != 0, &pool_on);
  } else {
    if (cmp) {
      DirectCount(kFastDirectPool,
                  cmp_pool_key == e->pool_key && cmp_pool_cls == (e->pool_cls != 0), *plan.rec,
                  cmp_pool_key, e->pool_key, (cmp_pool_cls ? 1u : 0u) | (e->pool_cls ? 2u : 0u));
    }
    pool_class_ok = NrFastPoolPrologue(regs, &pool_on);
    if (!plan.hit) NrFastPoolBake(regs, &plan.pool_key, &plan.pool_cls);
  }
  // Pipeline bind, through the tracker (SetExternalPipeline /
  // SetPipelineStateHandle with the current_* latches), exactly as
  // IssueDrawImpl binds.
  bool npso_build = false, npso_bind = false;
  NrFastNativePsoFlags(&npso_build, &npso_bind);
  // Lean: with the sorter off SortSegBegin is its census only (a hash set
  // insert per draw); no segment is opened, so none is closed.
  const bool sort_seg = pool_on && (!plan.lean || NrFastSortOn());
  if (sort_seg) {
    SortSegBegin(plan.npso && npso_bind ? static_cast<const void*>(plan.npso)
                                        : static_cast<const void*>(plan.pso),
                 pool_class_ok);
  }
  lap(g_derive.pool);
  if (plan.npso && npso_bind) {
    SetExternalPipeline(plan.npso);
    current_guest_pipeline_ = nullptr;
  } else if (current_guest_pipeline_ != plan.pso) {
    deferred_command_list_.SetPipelineStateHandle(plan.pso);
    current_guest_pipeline_ = plan.pso;
    current_external_pipeline_ = nullptr;
  }
  lap(g_derive.pso);
  // Viewport key cache (the full path's hit): IssueDrawImpl's cache kept
  // coherent. The direct path never touches it: previous_viewport_key_ /
  // previous_viewport_info_ stay the pair the last full draw left.
  if (!direct && plan.hit) {
    const bool convert_z = env_convert_z != 0;
    const bool ps_writes_depth = ps && ps->writes_depth();
    ViewportCacheKey vk;
    vk.pa_cl_clip_cntl = regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
    vk.pa_cl_vte_cntl = regs[XE_GPU_REG_PA_CL_VTE_CNTL];
    vk.pa_su_sc_mode_cntl = regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL];
    vk.pa_su_vtx_cntl = regs[XE_GPU_REG_PA_SU_VTX_CNTL];
    vk.pa_sc_window_offset = regs[XE_GPU_REG_PA_SC_WINDOW_OFFSET];
    vk.normalized_depth_control = plan.ndc.value;
    std::memcpy(vk.vport_regs, &regs[XE_GPU_REG_PA_CL_VPORT_XSCALE], sizeof(vk.vport_regs));
    vk.flags = (uint32_t(convert_z) << 0) | (uint32_t(host_rt) << 1) | (uint32_t(ps_writes_depth) << 2);
    previous_viewport_key_ = vk;
    previous_viewport_info_ = plan.viewport;
    viewport_cache_valid_ = true;
  }
  if (!direct) lap(g_derive.vport);
  // Scissor. GetScissor is a pure function of keyed registers (the entry
  // bakes its guest-pixel rect). Step 3 bakes the de-tile widen too: its
  // six register inputs are keyed and its other inputs (enabled, armed,
  // pass signature, full height, guard, segment limit) are environment
  // stamp writers (nr_detile.cpp), so the widened + scaled host rect is a
  // function of the entry plus the stamp. Under a widen segment limit the
  // widen reads the frame's tap progress per draw: not baked, the live
  // FastWidenBaked over the entry's six registers stands.
  draw_util::Scissor scissor;
  FastRect host_rect;
  bool cmp_sc_guest = true;
  if (direct && e->sc_baked) {
    host_rect = e->sc_host;
  } else {
    if (direct) {
      scissor = e->sc_guest;
    } else {
      draw_util::GetScissor(regs, scissor);
    }
    cmp_sc_guest = !cmp || !std::memcmp(&scissor, &e->sc_guest, sizeof(scissor));
    if (!plan.hit) plan.sc_guest = scissor;
    if (direct) {
      FastWidenBaked(e->dt_regs, scissor);
    } else {
      NrFastWidenScissor(regs, scissor);
    }
    host_rect = FastHostScissor(scissor.offset, scissor.extent, sx, sy);
    if (!direct) {
      scissor.offset[0] *= sx;
      scissor.offset[1] *= sy;
      scissor.extent[0] *= sx;
      scissor.extent[1] *= sy;
    }
    if (!plan.hit) {
      plan.sc_host = host_rect;
      plan.sc_baked = !nr::DetileWidenSegLimited();
    }
  }
  // Mode 2: the baked rect (the record-time widen) against the live widen.
  const FastRect cmp_rect = cmp && e->sc_baked ? e->sc_host : host_rect;
  lap(g_derive.scis);
  // Fixed-function state, through the SDK's per-list latches (ff_*, reset by
  // BeginSubmission): the direct path sets the entry's viewport, the scaled
  // rect and the entry's stencil reference with the same compare-and-set
  // helpers UpdateFixedFunctionState uses, and the blend factor live.
  if (direct) {
    SetViewport(e->vp);
    const D3D12_RECT rect{host_rect.left, host_rect.top, host_rect.right, host_rect.bottom};
    SetScissorRect(rect);
    float bf[4];
    read_blend(bf);
    // std::memcmp instead of != so in case of NaN, every draw won't be
    // invalidating it (UpdateFixedFunctionState's rule).
    ff_blend_factor_update_needed_ |= std::memcmp(ff_blend_factor_, bf, sizeof(bf)) != 0;
    if (ff_blend_factor_update_needed_) {
      std::memcpy(ff_blend_factor_, bf, sizeof(bf));
      deferred_command_list_.D3DOMSetBlendFactor(ff_blend_factor_);
      ff_blend_factor_update_needed_ = false;
    }
    SetStencilReference(e->stencil_ref);
  } else {
    UpdateFixedFunctionState(plan.viewport, scissor, plan.polygonal, plan.ndc);
    if (!plan.hit) {
      // The head the entry bakes: the host viewport as UpdateFixedFunctionState
      // built it (reversed-Z z_min / z_max copied as derived) and the stencil
      // reference by its face rule.
      const FastViewport fv = FastHostViewport(plan.viewport.xy_offset, plan.viewport.xy_extent,
                                               plan.viewport.z_min, plan.viewport.z_max);
      plan.vp = D3D12_VIEWPORT{fv.x, fv.y, fv.w, fv.h, fv.z_min, fv.z_max};
      const bool back = FastStencilUsesBackFace(plan.polygonal, plan.ndc.value,
                                                regs[XE_GPU_REG_PA_SU_SC_MODE_CNTL]);
      plan.stencil_ref =
          regs.Get<reg::RB_STENCILREFMASK>(back ? XE_GPU_REG_RB_STENCILREFMASK_BF
                                                : XE_GPU_REG_RB_STENCILREFMASK)
              .stencilref;
    }
    if (cmp) {
      const float* a = &e->vp.TopLeftX;
      const float* b = &ff_viewport_.TopLeftX;
      uint32_t vp_diff = UINT32_MAX;
      for (uint32_t i = 0; i < 6 && vp_diff == UINT32_MAX; ++i) {
        if (a[i] != b[i]) vp_diff = i;
      }
      DirectCount(kFastDirectVp, vp_diff == UINT32_MAX, *plan.rec,
                  vp_diff == UINT32_MAX ? 0 : std::bit_cast<uint32_t>(a[vp_diff]),
                  vp_diff == UINT32_MAX ? 0 : std::bit_cast<uint32_t>(b[vp_diff]), vp_diff);
      const FastRect latched{int32_t(ff_scissor_.left), int32_t(ff_scissor_.top),
                             int32_t(ff_scissor_.right), int32_t(ff_scissor_.bottom)};
      DirectCount(kFastDirectSc, cmp_sc_guest && cmp_rect == latched, *plan.rec,
                  (uint64_t(uint32_t(cmp_rect.left)) << 32) | uint32_t(cmp_rect.bottom),
                  (uint64_t(uint32_t(latched.left)) << 32) | uint32_t(latched.bottom),
                  (cmp_sc_guest ? 1u : 0u) | (e->sc_baked ? 2u : 0u));
      DirectCount(kFastDirectStencil, host_rt ? e->stencil_ref == ff_stencil_ref_ : true,
                  *plan.rec, e->stencil_ref, ff_stencil_ref_, 0);
      DirectCount(kFastDirectBlend,
                  !host_rt || !std::memcmp(cmp_blend, ff_blend_factor_, sizeof(cmp_blend)),
                  *plan.rec, std::bit_cast<uint32_t>(cmp_blend[0]),
                  std::bit_cast<uint32_t>(ff_blend_factor_[0]),
                  FirstDiffDword(cmp_blend, ff_blend_factor_, sizeof(cmp_blend)));
    }
  }
  lap(g_derive.ffs);
  // System constants.
  if (direct) {
    // The entry's block into the upload source (the mirror NrUpdateBindings
    // uploads). The bound CBV already holds these bytes when the latch
    // matches (same frame, hash and address: no other upload happened since
    // ours); the SDK's own invariant (up to date => the upload equals the
    // mirror) keeps a still-valid upload when the mirror did not change.
    // Otherwise NrFastBindings uploads it below.
    // The enabled user clip planes are not identity-keyed (PA_CL_UCP_* move
    // per draw): patched into the entry's block from the draw's own
    // registers (the map: the record's list over the live file), packed as
    // NrSysUpdate packs them; the hash of the patched block is the latch key.
    nr::NrSysConstants block = e->sys;
    const uint32_t clip_cntl = g_map.Read(rf, XE_GPU_REG_PA_CL_CLIP_CNTL);
    uint64_t sys_hash = e->sys_hash;
    if (FastSysUcpLiveDwords(clip_cntl)) {
      uint32_t ucp[24];
      for (uint32_t k = 0; k < 24; ++k) ucp[k] = g_map.Read(rf, kFastRegPaClUcp0X + k);
      FastSysPatchUcp(reinterpret_cast<uint32_t*>(&block), clip_cntl, ucp);
      sys_hash = XXH3_64bits(&block, sizeof(block));
    }
    plan.sys_hash_draw = sys_hash;
    const bool changed = NrFastSysStateSet(&block);
    const bool latched = nr_direct_latch_.frame == frame_current_ &&
                         nr_direct_latch_.sys_hash == sys_hash &&
                         nr_direct_latch_.address == cbuffer_binding_system_.address;
    if (latched) ++g_direct.sys_skip;
    cbuffer_binding_system_.up_to_date =
        latched || (cbuffer_binding_system_.up_to_date && !changed);
  } else {
    UpdateSystemConstantValues(false, plan.polygonal, pr.line_loop_closing_index,
                               pr.host_shader_index_endian, plan.viewport, plan.tex_mask,
                               plan.ndc, plan.color_mask);
    if (!plan.hit) {
      // Without host render targets UpdateFixedFunctionState sets no blend
      // factor or stencil reference: the head is not bakeable.
      plan.direct_ready = host_rt;
    }
  }
  lap(g_derive.sys);
  span.reset();
  ft.reset();
  ft.emplace(&ftrace::SplitTotals::nr_const_ticks);
  uint64_t lean_bytes = 0;
  if (plan.lean) {
    // Lean floats: composed from the record over the live file straight into
    // the upload, so NrUpdateBindings finds the stage up to date. A stage is
    // uploaded when NrUpdateBindings would have (dirty or a map change) or
    // when the record carries constants for it; a record constant inside the
    // shader's map marks the stage for the invalidation after the draw.
    const NrNativeDrawRecord& rec = *plan.rec;
    g_lean_map.Build(rec);
    auto upload = [&](uint32_t stage, const Shader::ConstantRegisterMap& cm,
                      ConstantBufferBinding& cb, uint64_t* cur_map, uint32_t root_bit) -> bool {
      const bool map_same = !std::memcmp(cur_map, cm.float_bitmap, sizeof(cm.float_bitmap));
      if (cb.up_to_date && map_same && !g_lean_map.stage_n[stage]) return true;
      std::memcpy(cur_map, cm.float_bitmap, sizeof(cm.float_bitmap));
      const uint32_t size = 16u * std::max<uint32_t>(cm.float_count, 1u);
      uint8_t* dst = constant_buffer_pool_->Request(
          frame_current_, size, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr, nullptr,
          &cb.address);
      if (!dst) return false;
      uint32_t hits = 0;
      FastComposeFloatsRecord(regs.values, stage, cm.float_bitmap, rec, g_lean_map, dst, size,
                              &hits);
      cb.up_to_date = true;
      current_graphics_root_up_to_date_ &= ~(1u << root_bit);
      lean_bytes += size;
      if (hits) plan.lean_floats |= uint8_t(1u << stage);
      return true;
    };
    if (!upload(0, vs->constant_register_map(), cbuffer_binding_float_vertex_,
                current_float_constant_map_vertex_,
                uint32_t(kRootParameter_Bindless_FloatConstantsVertex))) {
      return false;
    }
    if (ps && !upload(1, ps->constant_register_map(), cbuffer_binding_float_pixel_,
                      current_float_constant_map_pixel_,
                      uint32_t(kRootParameter_Bindless_FloatConstantsPixel))) {
      return false;
    }
  }
  // (d) With the overlay skipped NrUpdateBindings would compose the fetch and
  // bool/loop buffers from the live file: when the record changes a register
  // of either range (the dry classify's classes, raised as dirty before the
  // draw), the buffer is composed here from the map instead and the stage
  // marked up to date (NrUpdateBindings then skips its own upload).
  if (plan.overlay_skip) {
    auto upload_range = [&](uint32_t base, uint32_t dwords, ConstantBufferBinding& cb,
                            uint32_t root_bit) -> bool {
      uint8_t* dst = constant_buffer_pool_->Request(
          frame_current_, dwords * 4, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, nullptr,
          nullptr, &cb.address);
      if (!dst) return false;
      uint32_t* out = reinterpret_cast<uint32_t*>(dst);
      for (uint32_t i = 0; i < dwords; ++i) out[i] = g_map.Read(rf, base + i);
      cb.up_to_date = true;
      current_graphics_root_up_to_date_ &= ~(1u << root_bit);
      lean_bytes += dwords * 4;
      return true;
    };
    if (plan.dry.bool_loop &&
        !upload_range(kFastRegBoolBase, kFastRegLoopEnd - kFastRegBoolBase, cbuffer_binding_bool_loop_,
                      uint32_t(kRootParameter_Bindless_BoolLoopConstants))) {
      return false;
    }
    if ((plan.dry.tex_fetch || plan.dry.vtx_fetch[0] || plan.dry.vtx_fetch[1]) &&
        !upload_range(kFastRegFetchBase, kFastRegFetchEnd - kFastRegFetchBase, cbuffer_binding_fetch_,
                      uint32_t(kRootParameter_Bindless_FetchConstants))) {
      return false;
    }
  }
  // Constant buffer bytes this draw uploads (the bindings' dirty rules).
  {
    uint64_t bytes = lean_bytes;
    if (!cbuffer_binding_system_.up_to_date) bytes += sizeof(system_constants_);
    const Shader::ConstantRegisterMap& vcm = vs->constant_register_map();
    if (!cbuffer_binding_float_vertex_.up_to_date ||
        (vcm.float_count &&
         std::memcmp(current_float_constant_map_vertex_, vcm.float_bitmap, sizeof(vcm.float_bitmap)))) {
      bytes += 16ull * std::max<uint32_t>(vcm.float_count, 1);
    }
    const uint32_t pcount = ps ? ps->constant_register_map().float_count : 0u;
    if (!cbuffer_binding_float_pixel_.up_to_date ||
        (pcount && std::memcmp(current_float_constant_map_pixel_,
                               ps->constant_register_map().float_bitmap,
                               sizeof(current_float_constant_map_pixel_)))) {
      bytes += 16ull * std::max<uint32_t>(pcount, 1);
    }
    if (!cbuffer_binding_bool_loop_.up_to_date) bytes += nr::kBindBoolLoopBytes;
    if (!cbuffer_binding_fetch_.up_to_date) bytes += nr::kBindFetchBytes;
    NrNativeBump(&NrNativeCounters::cbv_bytes, bytes);
  }
  // Bindings: restored from the entry's record when it has one.
  ft.reset();
  ft.emplace(&ftrace::SplitTotals::nr_bind_ticks);
  const void* blob = plan.hit && plan.entry && plan.entry->blob_ok ? plan.entry->blob : nullptr;
  // Step 2 bind: a direct hit with a blob binds its descriptor-indices
  // packs by the latch in NrUpdateBindings and skips the member restore;
  // refused while the retail comparator captures the bound packs.
  plan.direct_bind = direct && blob != nullptr && !nr_retail_cmp_active_;
  const D3D12_GPU_VIRTUAL_ADDRESS di_addr0[2] = {
      cbuffer_binding_descriptor_indices_vertex_.address,
      cbuffer_binding_descriptor_indices_pixel_.address};
  if (!NrFastBindings(vs, ps, plan.rootsig, blob, plan.direct_bind, &plan.bind_ok)) return false;
  if (plan.direct_bind) {
    if (nr_direct_latch_.di_skipped[0]) ++g_direct.di_skip;
    if (nr_direct_latch_.di_skipped[1]) ++g_direct.di_skip;
  }
  if (cmp && blob && plan.bind_ok) {
    // The latch verdict against the full rule: where the latch matched, the
    // full path must have uploaded nothing, left the binding's address in
    // place, and the bound pack must be the blob's.
    const uint32_t *ev = nullptr, *ep = nullptr, *cv = nullptr, *cp = nullptr;
    uint32_t env_n = 0, enp_n = 0, cnv = 0, cnp = 0;
    bool ebv = false, ebp = false, cbv = false, cbp = false;
    NrFastBoundPacks(blob, &ev, &env_n, &ep, &enp_n, &ebv, &ebp);
    NrFastBoundPacks(nullptr, &cv, &cnv, &cp, &cnp, &cbv, &cbp);
    const D3D12_GPU_VIRTUAL_ADDRESS di_addr1[2] = {
        cbuffer_binding_descriptor_indices_vertex_.address,
        cbuffer_binding_descriptor_indices_pixel_.address};
    const bool bound[2] = {ebv, ebp};
    const uint32_t* entry_pack[2] = {ev, ep};
    const uint32_t* cur_pack[2] = {cv, cp};
    const uint32_t entry_n[2] = {env_n, enp_n};
    const uint32_t cur_n[2] = {cbv ? cnv : UINT32_MAX, cbp ? cnp : UINT32_MAX};
    for (uint32_t s = 0; s < 2; ++s) {
      if (!bound[s] || !nr_direct_latch_.di_latched[s]) continue;
      const bool same_pack = cur_n[s] == entry_n[s] &&
                             !std::memcmp(cur_pack[s], entry_pack[s], entry_n[s] * sizeof(uint32_t));
      const bool eq = !nr_direct_latch_.di_uploaded[s] && di_addr0[s] == di_addr1[s] && same_pack;
      DirectCount(s ? kFastDirectDiP : kFastDirectDiV, eq, *plan.rec, di_addr0[s], di_addr1[s],
                  (nr_direct_latch_.di_uploaded[s] ? 1u : 0u) | (same_pack ? 0u : 2u) |
                      (cur_n[s] << 8));
    }
  }
  if (direct && cbuffer_binding_system_.up_to_date) {
    // The CBV now bound holds the entry's block: the next direct hit with the
    // same block in this frame skips its upload.
    nr_direct_latch_.frame = frame_current_;
    nr_direct_latch_.sys_hash = plan.sys_hash_draw;
    nr_direct_latch_.address = cbuffer_binding_system_.address;
  }
  // The system constants block the bindings uploaded for this draw: the
  // upload source itself (g_nr_sys_state through NrFastSysStateGet, the
  // symbol the direct path writes), read after the upload, never the lazily
  // re-synced member. The unread packed clip-plane dwords are history (see
  // FastSysNormalizeUcp): zeroed in the bake and on both compare sides.
  if (!direct && (!plan.hit || cmp)) {
    nr::NrSysConstants uploaded;
    NrFastSysStateGet(&uploaded);
    const uint32_t clip_cntl = regs[XE_GPU_REG_PA_CL_CLIP_CNTL];
    if (!plan.hit) {
      plan.sys = uploaded;
      FastSysNormalizeUcp(reinterpret_cast<uint32_t*>(&plan.sys), clip_cntl);
      plan.sys_hash = XXH3_64bits(&plan.sys, sizeof(plan.sys));
    }
    if (cmp) {
      // The entry's block with the enabled planes patched from the draw's
      // registers (what the direct path uploads) against the full
      // derivation's upload with its unread plane dwords zeroed. A difference
      // outside the plane dwords is a sys mismatch; one inside them only is
      // sys_ucp_live_ne (0 means the patch reproduces the derivation).
      nr::NrSysConstants patched = e->sys;
      {
        uint32_t ucp[24];
        for (uint32_t k = 0; k < 24; ++k) ucp[k] = g_map.Read(regs.values, kFastRegPaClUcp0X + k);
        FastSysPatchUcp(reinterpret_cast<uint32_t*>(&patched), clip_cntl, ucp);
      }
      const uint32_t* a = reinterpret_cast<const uint32_t*>(&patched);
      const uint32_t raw = FirstDiffDword(&uploaded, &patched, sizeof(uploaded));
      FastSysNormalizeUcp(reinterpret_cast<uint32_t*>(&uploaded), clip_cntl);
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&uploaded);
      uint32_t d = UINT32_MAX, d_ucp = UINT32_MAX;
      for (uint32_t i = 0; i < nr::kNrSysConstantsDwords; ++i) {
        if (a[i] == b[i]) continue;
        const bool plane = i >= kFastSysUcpFirstDword && i < kFastSysUcpFirstDword + kFastSysUcpDwords;
        if (plane) {
          if (d_ucp == UINT32_MAX) d_ucp = i;
        } else if (d == UINT32_MAX) {
          d = i;
        }
      }
      if (d == UINT32_MAX && d_ucp == UINT32_MAX && raw != UINT32_MAX) ++g_direct.sys_ucp_stale;
      if (d == UINT32_MAX && d_ucp != UINT32_MAX) ++g_direct.sys_ucp_live_ne;
      const bool log = (d != UINT32_MAX || d_ucp != UINT32_MAX) && g_direct.logged[kFastDirectSys] < 4;
      DirectCount(kFastDirectSys, d == UINT32_MAX, *plan.rec, d == UINT32_MAX ? 0 : a[d],
                  d == UINT32_MAX ? 0 : b[d], d);
      if (log && d == UINT32_MAX) ++g_direct.logged[kFastDirectSys];
      if (log) {
        // The first 8 differing dwords by name, and the draw's identity (the
        // producer's token is 0 outside the census: the shaders and the
        // index range name the draw instead).
        std::string diffs;
        uint32_t listed = 0, total = 0;
        for (uint32_t i = 0; i < nr::kNrSysConstantsDwords; ++i) {
          if (a[i] == b[i]) continue;
          ++total;
          if (listed >= 8) continue;
          ++listed;
          char name[64];
          nr::NrSysDwordName(i, name, sizeof(name));
          diffs += fmt::format(" d{} {} entry={:08X} uploaded={:08X};", i, name, a[i], b[i]);
        }
        REXGPU_WARN("[jade-native-direct] sys detail token={:08X} frame={} vs={:08X}/{} "
                    "ps={:08X}/{} ib={:08X} n={} prim={} ucp_live_dwords={} differing={}:{}",
                    plan.rec->object_token, plan.rec->frame, plan.rec->vs_address,
                    plan.rec->vs_dwords, plan.rec->ps_address, plan.rec->ps_dwords,
                    plan.rec->index_base, plan.rec->index_count, plan.rec->prim_type,
                    FastSysUcpLiveDwords(clip_cntl), total, diffs);
      }
    }
  }
  ft.reset();
  ft.emplace(&ftrace::SplitTotals::nr_emit_ticks, &ftrace::SplitTotals::nr_n);
  // Vertex buffer residency (IssueDrawImpl's loop, without its probes).
  const Shader::ConstantRegisterMap& cmv = vs->constant_register_map();
  for (uint32_t i = 0; i < rex::countof(cmv.vertex_fetch_bitmap); ++i) {
    uint32_t bits = cmv.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(bits, &j)) {
      bits &= ~(uint32_t(1) << j);
      const uint32_t vfetch_index = i * 32 + j;
      const uint64_t vfetch_bit = uint64_t(1) << (vfetch_index & 63);
      if (vertex_buffers_in_sync_[vfetch_index >> 6] & vfetch_bit) continue;
      const xenos::xe_gpu_vertex_fetch_t vf = vfetch(vfetch_index);
      if (vf.type != xenos::FetchConstantType::kVertex &&
          !(vf.type == xenos::FetchConstantType::kInvalidVertex &&
            REXCVAR_GET(gpu_allow_invalid_fetch_constants))) {
        REXGPU_WARN("[jade-native-fast] vertex fetch constant {} ({:08X} {:08X}) is invalid",
                    vfetch_index, vf.dword_0, vf.dword_1);
        return false;
      }
      VertexBufferState& state = vertex_buffer_states_[vfetch_index];
      if (state.address == vf.address && state.size == vf.size) {
        vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
        continue;
      }
      if (!shared_memory_->RequestRange(vf.address << 2, vf.size << 2)) {
        REXGPU_ERROR("[jade-native-fast] failed to request vertex buffer at 0x{:08X} (size {})",
                     vf.address << 2, vf.size << 2);
        return false;
      }
      state.address = vf.address;
      state.size = vf.size;
      vertex_buffers_in_sync_[vfetch_index >> 6] |= vfetch_bit;
    }
  }
  NrFastAfterResidency();
  // [ia] fills and vertex buffer views (IssueDrawImpl's block). Step 3:
  // the direct path takes the keyed VGT_INDX_OFFSET from the entry.
  const uint32_t indx_offset = direct ? e->indx_offset : g_map_indx_offset(plan.mapped, rf, regs);
  if (cmp) {
    DirectCount(kFastDirectArgs,
                e->indx_offset == g_map_indx_offset(plan.mapped, rf, regs) &&
                    e->prim.host_draw_vertex_count == pr.host_draw_vertex_count,
                *plan.rec, e->indx_offset, g_map_indx_offset(plan.mapped, rf, regs),
                pr.host_draw_vertex_count);
    D3D_PRIMITIVE_TOPOLOGY topo = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
    const bool topo_ok = HostTopology(e->prim.host_primitive_type, &topo) && topo == e->topology;
    DirectCount(kFastDirectTopo, topo_ok, *plan.rec, uint32_t(e->topology), uint32_t(topo),
                uint32_t(e->prim.host_primitive_type));
  }
  ia_arg_dword2_ = 0;
  ia_arg_dword3_ = 0;
  // A sort window may move segments: under the sorter every draw binds its
  // views (IssueDrawImpl's rule).
  const bool sort_on = NrFastSortOn();
  if (plan.ia) {
    const std::vector<Shader::VertexBinding>& vb = vs->vertex_bindings();
    for (size_t b = 0; b < vb.size(); ++b) {
      const xenos::xe_gpu_vertex_fetch_t vf = vfetch(vb[b].fetch_constant);
      vb_mirror_->QueueFills(vf.address << 2, vf.size << 2, uint32_t(vf.endian));
    }
    if (plan.ia_ib_va && pr.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA) {
      vb_mirror_->QueueFills(pr.guest_index_base, plan.ib_req_size,
                             uint32_t(pr.host_shader_index_endian));
    }
    if (vb_mirror_->HasPendingFills()) {
      // The fill's compute pipeline replaces the draw's: rebind after
      // (run 862), through the same latches.
      void* saved_guest = current_guest_pipeline_;
      auto saved_external = current_external_pipeline_;
      shared_memory_->UseForReading();
      vb_mirror_->EmitFills();
      if (saved_external) {
        SetExternalPipeline(saved_external);
      } else if (saved_guest) {
        deferred_command_list_.SetPipelineStateHandle(saved_guest);
        current_guest_pipeline_ = saved_guest;
        current_external_pipeline_ = nullptr;
      }
    }
    for (size_t b = 0; b < vb.size(); ++b) {
      const xenos::xe_gpu_vertex_fetch_t vf = vfetch(vb[b].fetch_constant);
      D3D12_VERTEX_BUFFER_VIEW view;
      view.BufferLocation = plan.ia_vb_va[b];
      view.SizeInBytes = vf.size << 2;
      view.StrideInBytes = vb[b].stride_words * 4;
      IaVbLatch& l = ia_vb_latch_[b];
      if (sort_on || l.va != view.BufferLocation || l.size != view.SizeInBytes ||
          l.stride != view.StrideInBytes) {
        deferred_command_list_.D3DIASetVertexBuffers(UINT(b), 1, &view);
        l.va = view.BufferLocation;
        l.size = view.SizeInBytes;
        l.stride = view.StrideInBytes;
      }
    }
    if (pr.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kNone) {
      ia_arg_dword2_ = indx_offset;
    } else {
      ia_arg_dword3_ = indx_offset;
    }
  }
  SetPrimitiveTopology(plan.topology);
  // Draw.
  const uint32_t count = pr.host_draw_vertex_count;
  if (pr.index_buffer_type == PrimitiveProcessor::ProcessedIndexBufferType::kNone) {
    shared_memory_->UseForReading();
    SubmitBarriers();
    // Lean: the eligibility decode (a hash map find) feeds only an open
    // occlusion submission's tag.
    if (!plan.lean || occ_sub_active_) OccSetElig(vs, regs);
    OccDrawBegin();
    deferred_command_list_.D3DDrawInstanced(count, 1, ia_arg_dword2_, 0);
    if (nr_retail_cmp_active_) NrRetailCmpArgs(false, count, nullptr);
    OccDrawEnd(count, 0);
  } else {
    // Step 3: the index buffer view is baked for an indexed entry whose base
    // is static (the shared memory window plus the guest base, or the
    // builtin buffer); an IA entry's guest-DMA view is the live acquire
    // (vb_mirror_ moves it), composed here as before.
    D3D12_INDEX_BUFFER_VIEW ibv;
    const bool ia_dma = plan.ia && pr.index_buffer_type ==
                                       PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA;
    if (direct && e->ibv_ok && !ia_dma) {
      ibv.BufferLocation = e->ibv.location;
      ibv.SizeInBytes = e->ibv.size_bytes;
      ibv.Format = e->ibv.index32 ? DXGI_FORMAT_R32_UINT : DXGI_FORMAT_R16_UINT;
    } else {
      if (direct) ++g_direct.ia_live;
      ibv.SizeInBytes = count;
      if (pr.host_index_format == xenos::IndexFormat::kInt16) {
        ibv.SizeInBytes *= sizeof(uint16_t);
        ibv.Format = DXGI_FORMAT_R16_UINT;
      } else {
        ibv.SizeInBytes *= sizeof(uint32_t);
        ibv.Format = DXGI_FORMAT_R32_UINT;
      }
      switch (pr.index_buffer_type) {
        case PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA:
          ibv.BufferLocation =
              plan.ia_ib_va ? plan.ia_ib_va : shared_memory_->GetGPUAddress() + pr.guest_index_base;
          break;
        case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto:
        case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA:
          ibv.BufferLocation =
              primitive_processor_->GetBuiltinIndexBufferGpuAddress(pr.host_index_buffer_handle);
          break;
        default:
          return false;  // converted index buffers are refused at derivation
      }
      const FastIbv live = FastBakeIbv(ibv.BufferLocation, count,
                                       pr.host_index_format != xenos::IndexFormat::kInt16);
      if (cmp) {
        // The baked view vs the live composition; an IA guest-DMA entry
        // compares everything but the base (detail bit 0).
        const bool loc_ok = ia_dma || live.location == e->ibv.location;
        DirectCount(kFastDirectIbv,
                    loc_ok && live.size_bytes == e->ibv.size_bytes && live.index32 == e->ibv.index32,
                    *plan.rec, e->ibv.location, live.location,
                    (ia_dma ? 1u : 0u) | (e->ibv_ok ? 2u : 0u) | (live.size_bytes << 8));
      }
      if (!plan.hit) {
        plan.ibv = live;
        plan.ibv_ok = !ia_dma;
      }
    }
    deferred_command_list_.D3DIASetIndexBuffer(&ibv);
    shared_memory_->UseForReading();
    SubmitBarriers();
    if (!plan.lean || occ_sub_active_) OccSetElig(vs, regs);
    OccDrawBegin();
    deferred_command_list_.D3DDrawIndexedInstanced(count, 1, 0, INT(ia_arg_dword3_), 0);
    if (nr_retail_cmp_active_) NrRetailCmpArgs(true, count, &ibv);
    OccDrawEnd(count, 0);
  }
  if (nr_retail_cmp_active_) NrRetailCmpCapture(vs, ps, pr);
  if (sort_seg) SortSegEnd();
  NrFastNoteDrawIssued(count);
  return true;
}

// ── Entry record ────────────────────────────────────────────────────────────
void D3D12CommandProcessor::NrFastRecordEntry(NrFastPlan& plan, bool bindings_from_v1) {
  FastEntry* e = g_objects.Victim(plan.key);
  if (!e) return;
  if (e->valid && (e->h1 != plan.key.h1 || e->h2 != plan.key.h2)) ++g_stats.evictions;
  if (!e->valid) ++g_stats.entries;
  e->valid = false;
  e->h1 = plan.key.h1;
  e->h2 = plan.key.h2;
  e->stamp = g_stamp;
  e->gen = nr_fast_gen_;
  e->tex_gen = texture_cache_->nr_texture_destroy_generation();
  e->smp_heap = sampler_bindless_heap_gpu_start_.ptr;
  e->view_heap = view_bindless_heap_gpu_start_.ptr;
  e->vs = plan.vs;
  e->ps = plan.ps;
  e->kind = plan.kind;
  e->reason = plan.reason;
  e->ia = plan.ia ? 1 : 0;
  e->polygonal = plan.polygonal ? 1 : 0;
  e->ndc = plan.ndc.value;
  e->color_mask = plan.color_mask;
  e->topology = plan.topology;
  e->prim = plan.prim;
  e->ib_req_base = plan.ib_req_base;
  e->ib_req_size = plan.ib_req_size;
  e->pso = plan.pso;
  e->rootsig = plan.rootsig;
  e->npso = plan.npso;
  e->rt_bits = plan.rt_bits;
  std::memcpy(e->rt_formats, plan.rt_formats, sizeof(e->rt_formats));
  e->viewport = plan.viewport;
  e->tex_mask = plan.tex_mask;
  e->tex_count = uint8_t(plan.tex_count);
  for (uint32_t i = 0; i < plan.tex_count; ++i) e->tex[i] = plan.tex[i];
  e->blob_ok = 0;
  if (plan.kind == kFastKindDraw) {
    // The bindings NrUpdateBindings left for this draw (the fast emission's,
    // or v1's in compare mode, whose IssueDrawImpl ran the same function).
    const bool can = bindings_from_v1 ? (g_cap.valid && g_cap.emitted) : plan.bind_ok;
    if (can && NrFastBindCapture(e->blob, plan.vs, plan.ps)) {
      e->blob_ok = 1;
    } else {
      ++g_stats.blob_refused;
    }
  }
  // The direct head (gpu_native_direct), from the miss emission's values.
  e->direct_ok = 0;
  if (plan.kind == kFastKindDraw && plan.direct_ready) {
    e->direct_ok = 1;
    e->hiz_mono_mask = uint8_t(FastHizMonoMask(plan.ndc.value));
    e->pool_cls = plan.pool_cls ? 1 : 0;
    e->pool_key = plan.pool_key;
    // Step 4: the stamp and its fields; step 3: the widened scaled scissor
    // and the index buffer view of the miss emission.
    e->env = nr::NrDirectEnvStamp();
    e->env_fields = NrFastEnvFields();
    e->sc_baked = plan.sc_baked ? 1 : 0;
    e->sc_host = plan.sc_host;
    e->ibv_ok = plan.ibv_ok ? 1 : 0;
    e->ibv = plan.ibv;
    e->stencil_ref = plan.stencil_ref;
    e->sys_hash = plan.sys_hash;
    e->vp = plan.vp;
    e->sc_guest = plan.sc_guest;
    e->sys = plan.sys;
    // Step 2 bakes, from the overlaid file (the restore comes after this).
    const uint32_t* rf = register_file_->values;
    uint32_t blend[4];
    for (uint32_t i = 0; i < 4; ++i) blend[i] = rf[reg::RB_BLENDCONTROL::rt_register_indices[i]];
    uint32_t reads = 0;
    e->xfer_used = uint8_t(FastXferVerdict(plan.ndc.value, plan.color_mask, blend, &reads, e->xfer_reason));
    e->xfer_reads = uint8_t(reads);
    e->indx_offset = rf[XE_GPU_REG_VGT_INDX_OFFSET] & 0xFFFFFFu;
    const uint32_t dt_idx[6] = {XE_GPU_REG_PA_SC_WINDOW_OFFSET, XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL,
                                XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR, XE_GPU_REG_RB_SURFACE_INFO,
                                XE_GPU_REG_RB_COLOR_INFO, XE_GPU_REG_RB_DEPTH_INFO};
    for (uint32_t i = 0; i < 6; ++i) e->dt_regs[i] = rf[dt_idx[i]];
    // The latches of the miss derivation: its Update ran (or guard-hit) with
    // these inputs, its index request succeeded (the miss path's Process).
    e->rt_serial = render_target_cache_->nr_update_serial();
    e->ib_latched = plan.prim.index_buffer_type ==
                            PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA
                        ? 1
                        : 0;
    e->shm_gen = g_shm_gen.load(std::memory_order_relaxed);
  }
  // The texture generation latch is taken on the first hit's restore.
  e->tex_latched = 0;
  e->tex_bgen = 0;
  e->valid = true;
  ++g_stamp;
}

// ── Compare capture (inside v1, before its restore) ─────────────────────────
void D3D12CommandProcessor::NrRetailCmpBegin(uint32_t slot_sequence) {
  if (nr_retail_cmp_active_) ++g_retail_stats.superseded;
  nr_retail_cmp_active_ = false;
  NrNativeDrawRecord record = {};
  bool torn = false;
  if (NrNativeAcquire(slot_sequence, &record, &torn) != NrNativeVerdict::kOk ||
      !NrNativeRecordBoundsOk(record.register_count, record.constant_count) ||
      !(record.flags & kNrNativeRetailFollows) || (record.flags & kNrNativeVerifyOnly)) return;
  g_retail = RetailScope{};
  g_retail.ref = slot_sequence;
  g_retail.record = record;
  nr_retail_cmp_active_ = true;
  ++g_retail_stats.begun;
}

void D3D12CommandProcessor::NrRetailCmpDrawStart(xenos::PrimitiveType primitive_type,
                                               uint32_t index_count,
                                               const IndexBufferInfo* ibi) {
  if (nr_retail_cmp_native_) return;
  if (++g_retail.retail_draws != 1) return;
  const auto& rec = g_retail.record;
  g_retail.retail_prim = uint32_t(primitive_type);
  g_retail.retail_count = index_count;
  g_retail.retail_base = ibi ? ibi->guest_base : 0;
  g_retail.retail_format = ibi ? uint32_t(ibi->format) : 0;
  g_retail.retail_endian = ibi ? uint32_t(ibi->endianness) : 0;
  g_retail.input_match = uint32_t(primitive_type) == rec.prim_type && index_count == rec.index_count &&
      bool(ibi) == bool(rec.flags & kNrNativeIndexed) &&
      (!ibi || (ibi->guest_base == rec.index_base && uint32_t(ibi->endianness) == rec.index_endian &&
       ibi->format == ((rec.flags & kNrNativeIndex32) ? xenos::IndexFormat::kInt32 : xenos::IndexFormat::kInt16)));
}

void D3D12CommandProcessor::NrRetailCmpArgs(bool indexed, uint32_t count,
                                          const D3D12_INDEX_BUFFER_VIEW* ibv) {
  if (!nr_retail_cmp_native_ && g_retail.retail_draws != 1) return;
  auto& c = nr_retail_cmp_native_ ? g_retail.native : g_retail.retail;
  c.direct = true;
  c.indexed = indexed;
  c.args[0] = count;
  c.args[1] = 1;
  c.args[2] = indexed ? 0 : ia_arg_dword2_;
  c.args[3] = indexed ? ia_arg_dword3_ : 0;
  c.args[4] = 0;
  if (ibv) c.ibv = *ibv;
}

void D3D12CommandProcessor::NrRetailCmpCapture(const D3D12Shader* vs, const D3D12Shader* ps,
                                             const PrimitiveProcessor::ProcessingResult& prim) {
  if (!nr_retail_cmp_native_ && g_retail.retail_draws != 1) return;
  auto& out = nr_retail_cmp_native_ ? g_retail.native : g_retail.retail;
  if (!out.direct || !vs) return;
  auto& c = out.state;
  c.valid = true;
  c.vs = vs;
  c.ps = ps;
  const uint32_t* rf = GetActiveDrawRegisterFile().values;
  const uint64_t zero_map[4] = {};
  c.fv_n = FastComposeFloats(rf, kFastRegFloatBase, vs->constant_register_map().float_bitmap,
                             c.fv, sizeof(c.fv));
  c.fp_n = FastComposeFloats(rf, kFastRegFloatBase + 0x400,
                             ps ? ps->constant_register_map().float_bitmap : zero_map,
                             c.fp, sizeof(c.fp));
  std::memcpy(c.bl, &rf[kFastRegBoolBase], sizeof(c.bl));
  std::memcpy(c.ft, &rf[kFastRegFetchBase], sizeof(c.ft));
  c.guest_pso = current_guest_pipeline_;
  c.ext_pso = current_external_pipeline_;
  c.rootsig = current_graphics_root_signature_;
  c.vp = previous_viewport_info_;
  c.tex_mask = vs->GetUsedTextureMaskAfterTranslation() |
      (ps ? ps->GetUsedTextureMaskAfterTranslation() : 0);
  c.tex_n = c.tex_mask ? texture_cache_->NrTexCapture(c.tex_mask, c.tex, kFastTexMax) : 0;
  const uint32_t *v = nullptr, *p = nullptr;
  NrFastBoundPacks(nullptr, &v, &c.nv, &p, &c.np, &c.bv, &c.bp);
  if (c.tex_n == UINT32_MAX || c.nv > 64 || c.np > 64 ||
      (c.nv && !v) || (c.np && !p)) { c.valid = false; return; }
  if (c.nv) std::memcpy(c.div, v, c.nv * sizeof(uint32_t));
  if (c.np) std::memcpy(c.dip, p, c.np * sizeof(uint32_t));
  // CPU mirrors of the intended constant payloads, not reads of the bound
  // upload heap. Equal values here do not exclude a stale CBV binding.
  NrRetailCmpCopySystem(&out.system);
  out.scissor = ff_scissor_;
  out.prim = prim;
  out.targets_valid = render_target_cache_->NrBoundRenderTargetIdentities(out.targets);
}

void D3D12CommandProcessor::NrNativeEnd(uint32_t slot_sequence) {
  if (!nr_retail_cmp_active_) return;
  const auto verdict = NrNativeCheckRetailEnd(g_retail.ref, slot_sequence, g_retail.retail_draws,
      g_retail.native.direct && g_retail.native.state.valid,
      g_retail.retail.direct && g_retail.retail.state.valid);
  if (verdict == NrNativeRetailEndVerdict::kWrongEnd) { ++g_retail_stats.wrong_end; return; }
  nr_retail_cmp_active_ = false;
  if (verdict != NrNativeRetailEndVerdict::kPaired) {
    if (verdict == NrNativeRetailEndVerdict::kNoRetail) ++g_retail_stats.no_retail;
    else if (verdict == NrNativeRetailEndVerdict::kMultiple) ++g_retail_stats.multiple;
    else ++g_retail_stats.unavailable;
    return;
  }
  ++g_retail_stats.paired;
  const auto& a = g_retail.native;
  const auto& b = g_retail.retail;
  const auto& x = a.state;
  const auto& y = b.state;
  auto compare = [&](const char* cls, bool equal, uint32_t detail = UINT32_MAX,
                     uint64_t native_value = 0, uint64_t retail_value = 0) {
    uint32_t ci = 0;
    while (ci < kRetailClassCount && std::strcmp(cls, kRetailClasses[ci])) ++ci;
    if (ci == kRetailClassCount) return;
    if (equal) { ++g_retail_stats.classes_eq; ++g_retail_stats.eq[ci]; return; }
    ++g_retail_stats.classes_ne;
    ++g_retail_stats.ne[ci];
    if (g_retail_stats.class_logged[ci] >= 2) return;
    ++g_retail_stats.class_logged[ci];
    ++g_retail_stats.logged;
    REXGPU_WARN("[jade-native-retail] MISMATCH #{} class={} ref={:08X} token={:08X} frame={} "
                "detail={} native={:016X} retail={:016X}", g_retail_stats.logged, cls,
                g_retail.ref, g_retail.record.object_token, g_retail.record.frame,
                detail, native_value, retail_value);
    if ((!std::strcmp(cls, "fv") || !std::strcmp(cls, "fp")) && detail != UINT32_MAX) {
      const bool pixel = !std::strcmp(cls, "fp");
      const D3D12Shader* ns = pixel ? x.ps : x.vs;
      const D3D12Shader* rs = pixel ? y.ps : y.vs;
      if (ns && rs) {
        const uint32_t nr = PackedFloatRegister(ns->constant_register_map().float_bitmap, detail);
        const uint32_t rr = PackedFloatRegister(rs->constant_register_map().float_bitmap, detail);
        const uint32_t lane = detail & 3;
        bool in_record = false;
        uint32_t record_value = 0;
        for (uint32_t i = 0; i < g_retail.record.constant_count; ++i) {
          const auto& constant = g_retail.record.constants[i];
          if (constant.reg == nr + (pixel ? 256u : 0u)) {
            in_record = true;
            record_value = std::bit_cast<uint32_t>(constant.v[lane]);
          }
        }
        REXGPU_WARN("[jade-native-retail] constant ref={:08X} stage={} native_c={} retail_c={} "
                    "lane={} gpu_reg={:04X} in_record={} record_bits={:08X} "
                    "ucode={:08X}/{} hash={:016X}/{:016X}",
                    g_retail.ref, pixel ? "ps" : "vs", nr, rr, lane,
                    kFastRegFloatBase + (pixel ? 0x400u : 0u) + 4 * nr + lane,
                    in_record, record_value,
                    pixel ? g_retail.record.ps_address : g_retail.record.vs_address,
                    pixel ? g_retail.record.ps_dwords : g_retail.record.vs_dwords,
                    ns->ucode_data_hash(), rs->ucode_data_hash());
      }
    }
  };
  if (!g_retail.input_match) {
    ++g_retail_stats.identity_ne;
    compare("identity", false, g_retail.retail_prim,
            (uint64_t(g_retail.record.index_base) << 32) | g_retail.record.index_count,
            (uint64_t(g_retail.retail_base) << 32) | g_retail.retail_count);
  }
  compare("shader_vs", x.vs == y.vs, 0, uintptr_t(x.vs), uintptr_t(y.vs));
  compare("shader_ps", x.ps == y.ps, 0, uintptr_t(x.ps), uintptr_t(y.ps));
  compare("pso", x.guest_pso == y.guest_pso && x.ext_pso == y.ext_pso, 0,
          uintptr_t(x.ext_pso ? static_cast<void*>(x.ext_pso) : x.guest_pso),
          uintptr_t(y.ext_pso ? static_cast<void*>(y.ext_pso) : y.guest_pso));
  compare("root", x.rootsig == y.rootsig, 0, uintptr_t(x.rootsig), uintptr_t(y.rootsig));
  auto bytes = [&](const char* cls, const void* av, const void* bv, uint32_t n) {
    const uint32_t d = FirstDiffDword(av, bv, n);
    compare(cls, d == UINT32_MAX, d, d == UINT32_MAX ? 0 : static_cast<const uint32_t*>(av)[d],
            d == UINT32_MAX ? 0 : static_cast<const uint32_t*>(bv)[d]);
  };
  compare("float_sizes", x.fv_n == y.fv_n && x.fp_n == y.fp_n);
  bytes("fv", x.fv, y.fv, std::min(x.fv_n, y.fv_n));
  bytes("fp", x.fp, y.fp, std::min(x.fp_n, y.fp_n));
  // Inspect every paired packed word, including words beyond the first
  // mismatch. This remains bounded by the two fixed 256-register captures.
  static_assert(nr::kBindFloatMaxBytes == 256 * 4 * sizeof(uint32_t));
  auto float_words = [&](uint32_t stage, const uint8_t* av, uint32_t an,
                         const uint8_t* bv, uint32_t bn,
                         const D3D12Shader* as, const D3D12Shader* bs) {
    if (an != bn) ++g_retail_stats.float_size_different_draws[stage];
    const uint32_t n = std::min(an, bn) / sizeof(uint32_t);
    uint32_t ar_map[256], br_map[256];
    auto register_map = [](const D3D12Shader* shader, uint32_t* map) {
      std::fill_n(map, 256, UINT32_MAX);
      if (!shader) return;
      uint32_t packed = 0;
      for (uint32_t w = 0; w < 4; ++w) {
        uint64_t bits = shader->constant_register_map().float_bitmap[w];
        while (bits) {
          const uint32_t bit = uint32_t(__builtin_ctzll(bits));
          bits &= bits - 1;
          map[packed++] = w * 64 + bit;
        }
      }
    };
    register_map(as, ar_map);
    register_map(bs, br_map);
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t a, b;
      std::memcpy(&a, av + i * sizeof(uint32_t), sizeof(a));
      std::memcpy(&b, bv + i * sizeof(uint32_t), sizeof(b));
      AccumulateFloatWord(g_retail_stats.float_words[stage], a, b, ar_map[i / 4], br_map[i / 4], i & 3);
    }
  };
  float_words(0, x.fv, x.fv_n, y.fv, y.fv_n, x.vs, y.vs);
  float_words(1, x.fp, x.fp_n, y.fp, y.fp_n, x.ps, y.ps);
  bytes("bl", x.bl, y.bl, sizeof(x.bl));
  bytes("ft", x.ft, y.ft, sizeof(x.ft));
  bytes("system", &a.system, &b.system, sizeof(a.system));
  bytes("viewport", &x.vp, &y.vp, sizeof(x.vp));
  bytes("scissor", &a.scissor, &b.scissor, sizeof(a.scissor));
  if (a.targets_valid && b.targets_valid) bytes("targets", a.targets, b.targets, sizeof(a.targets));
  else ++g_retail_stats.unavailable;
  compare("packs_meta", x.nv == y.nv && x.np == y.np && x.bv == y.bv && x.bp == y.bp);
  bytes("div", x.div, y.div, std::min(x.nv, y.nv) * sizeof(uint32_t));
  bytes("dip", x.dip, y.dip, std::min(x.np, y.np) * sizeof(uint32_t));
  bool tex_equal = x.tex_mask == y.tex_mask && x.tex_n == y.tex_n;
  for (uint32_t i = 0; tex_equal && i < x.tex_n; ++i) tex_equal = TexSnapEq(x.tex[i], y.tex[i]);
  compare("texture", tex_equal);
  const auto& ap = a.prim;
  const auto& bp = b.prim;
  compare("primitive", ap.guest_primitive_type == bp.guest_primitive_type &&
      ap.host_primitive_type == bp.host_primitive_type && ap.host_vertex_shader_type == bp.host_vertex_shader_type &&
      ap.tessellation_mode == bp.tessellation_mode && ap.guest_draw_vertex_count == bp.guest_draw_vertex_count &&
      ap.host_draw_vertex_count == bp.host_draw_vertex_count && ap.index_buffer_type == bp.index_buffer_type &&
      ap.guest_index_base == bp.guest_index_base && ap.host_index_format == bp.host_index_format &&
      ap.host_shader_index_endian == bp.host_shader_index_endian &&
      ap.host_primitive_reset_enabled == bp.host_primitive_reset_enabled &&
      ap.line_loop_closing_index == bp.line_loop_closing_index,
      0, ap.guest_index_base, bp.guest_index_base);
  compare("indexed", a.indexed == b.indexed);
  bytes("draw_args", a.args, b.args, sizeof(a.args));
  compare("ib_view", a.ibv.BufferLocation == b.ibv.BufferLocation &&
      a.ibv.SizeInBytes == b.ibv.SizeInBytes && a.ibv.Format == b.ibv.Format, 0,
      a.ibv.BufferLocation, b.ibv.BufferLocation);
}

void D3D12CommandProcessor::NrFastCmpCaptureV1(bool drew) {
  FastCmpCapture& c = g_cap;
  c.valid = true;
  c.drew = drew;
  const uint32_t* rf = register_file_->values;
  const uint64_t zero_map[4] = {0, 0, 0, 0};
  c.fv_n = FastComposeFloats(rf, kFastRegFloatBase, c.vs->constant_register_map().float_bitmap,
                             c.fv, sizeof(c.fv));
  c.fp_n = FastComposeFloats(rf, kFastRegFloatBase + 0x400,
                             c.ps ? c.ps->constant_register_map().float_bitmap : zero_map, c.fp,
                             sizeof(c.fp));
  std::memcpy(c.bl, &rf[kFastRegBoolBase], sizeof(c.bl));
  std::memcpy(c.ft, &rf[kFastRegFetchBase], sizeof(c.ft));
}

// Called only at this draw's direct emission, before sorter/occlusion tails
// or a subsequent pool flush can replace its pipeline and bindings.
void D3D12CommandProcessor::NrFastCmpCaptureDraw(void* description_handle) {
  FastCmpCapture& c = g_cap;
  c.emitted = true;
  c.guest_pso = current_guest_pipeline_;
  c.ext_pso = current_external_pipeline_;
  c.rootsig = current_graphics_root_signature_;
  c.pso_description_handle = description_handle;
  c.pso_desc_valid = pipeline_cache_->NrCopyPipelineDescription(description_handle, c.pso_desc);
  c.vp = previous_viewport_info_;
  const uint32_t n =
      c.tex_mask ? texture_cache_->NrTexCapture(c.tex_mask, c.tex, kFastTexMax) : 0u;
  c.tex_n = n == UINT32_MAX ? 0u : n;
  const uint32_t *v = nullptr, *p = nullptr;
  NrFastBoundPacks(nullptr, &v, &c.nv, &p, &c.np, &c.bv, &c.bp);
  c.nv = std::min<uint32_t>(c.nv, 64);
  c.np = std::min<uint32_t>(c.np, 64);
  std::memcpy(c.div, v, c.nv * sizeof(uint32_t));
  std::memcpy(c.dip, p, c.np * sizeof(uint32_t));
}

void D3D12CommandProcessor::NrFastCmpSkipDraw(uint32_t kind) {
  if (kind == 1) ++g_cmp.excluded_pool;
  else if (kind == 2) ++g_cmp.excluded_instanced;
  else if (kind == 3) ++g_cmp.excluded_indirect;
}

// ── 1 Hz reports ────────────────────────────────────────────────────────────
void D3D12CommandProcessor::NrFastReport1Hz() {
  static uint64_t retail_sig = 0;
  if (g_retail_stats.begun != retail_sig) {
    retail_sig = g_retail_stats.begun;
    REXGPU_INFO("[jade-native-retail] begun={} paired={} superseded={} wrong_end={} "
                "no_retail={} multiple={} identity_ne={} unavailable={} eq={} ne={}",
                g_retail_stats.begun, g_retail_stats.paired, g_retail_stats.superseded,
                g_retail_stats.wrong_end, g_retail_stats.no_retail, g_retail_stats.multiple,
                g_retail_stats.identity_ne, g_retail_stats.unavailable,
                g_retail_stats.classes_eq, g_retail_stats.classes_ne);
    std::string classes;
    for (uint32_t i = 0; i < kRetailClassCount; ++i)
      classes += fmt::format(" {}={}/{}", kRetailClasses[i], g_retail_stats.eq[i], g_retail_stats.ne[i]);
    REXGPU_INFO("[jade-native-retail] classes eq/ne{}", classes);
    for (uint32_t stage = 0; stage < 2; ++stage) {
      const auto& f = g_retail_stats.float_words[stage];
      REXGPU_INFO("[jade-native-retail] float_words stage={} checked={} different={} "
                  "size_ne_draws={} map_ne_words={} map_unknown_words={} "
                  "signed_zero_words={} sign_ne_words={} nonfinite_words={} nan_words={} infinity_words={} "
                  "finite_ulp_words 1/2/3-4/>4={}/{}/{}/{} max_ulp={} max_c={} lane={} "
                  "native_bits={:08X} retail_bits={:08X}",
                  stage ? "ps" : "vs", f.words, f.different_words,
                  g_retail_stats.float_size_different_draws[stage],
                  f.mapping_different_words, f.mapping_unknown_words, f.signed_zero_words,
                  f.sign_different_words, f.nonfinite_words, f.nan_words, f.infinity_words,
                  f.ulp1_words, f.ulp2_words, f.ulp3_4_words, f.ulp_gt4_words,
                  f.max_ulp, f.max_register, f.max_lane, f.max_native_bits, f.max_retail_bits);
    }
  }
  if (g_derive.n) {
    const double freq = double(ftrace::TickFrequency());
    const double k = freq > 0.0 ? 1e6 / (freq * double(g_derive.n)) : 0.0;
    const uint64_t total = g_derive.hiz + g_derive.pool + g_derive.pso + g_derive.vport +
                           g_derive.scis + g_derive.ffs + g_derive.sys + g_derive.rest +
                           g_derive.ovl + g_derive.ovr + g_derive.sub + g_derive.ib +
                           g_derive.ia + g_derive.rt + g_derive.tex + g_derive.miss +
                           g_derive.map + g_derive.txb;
    REXGPU_INFO("[jade-native-derive] n={} us/draw hiz={:.3f} pool={:.3f} pso={:.3f} vport={:.3f} scis={:.3f} ffs={:.3f} sys={:.3f} rest={:.3f} ovl={:.3f} ovr={:.3f} sub={:.3f} ib={:.3f} ia={:.3f} rt={:.3f} tex={:.3f} miss={:.3f} map={:.3f} txb={:.3f} total={:.3f}",
                g_derive.n, k * double(g_derive.hiz), k * double(g_derive.pool),
                k * double(g_derive.pso), k * double(g_derive.vport), k * double(g_derive.scis),
                k * double(g_derive.ffs), k * double(g_derive.sys), k * double(g_derive.rest),
                k * double(g_derive.ovl), k * double(g_derive.ovr), k * double(g_derive.sub),
                k * double(g_derive.ib), k * double(g_derive.ia), k * double(g_derive.rt),
                k * double(g_derive.tex), k * double(g_derive.miss), k * double(g_derive.map),
                k * double(g_derive.txb), k * double(total));
    g_derive = FastDeriveProbe{};
  }
  static uint64_t s_sig = 0;
  uint64_t sig = g_stats.entries + g_stats.nodraw + g_stats.delegate_hits + g_stats.pso_table_hit +
                 g_cmp.skip_emit + g_cmp.fallback + g_cmp.no_packs + g_direct.draws +
                 g_direct.refused_env + g_direct.nodirect + g_direct.rt_skip + g_direct.ib_skip +
                 g_direct.overlay_skip + g_direct.overlay_live_rt + g_direct.overlay_live_blob +
                 g_direct.overlay_live_other + g_direct.tex_skip + g_direct.tex_live +
                 g_direct.di_skip + g_direct.sys_skip + g_direct.ia_live + g_direct.env_refresh;
  for (uint32_t i = 0; i < kFastFbReasons; ++i) sig += g_stats.fb[i];
  for (uint32_t i = 0; i < kFastCmpClasses; ++i) sig += g_cmp.eq[i] + g_cmp.ne[i];
  for (uint32_t i = 0; i < kFastDirectClasses; ++i) sig += g_direct.eq[i] + g_direct.ne[i];
  if (sig == s_sig) return;
  s_sig = sig;
  std::string fb;
  for (uint32_t i = 1; i < kFastFbReasons; ++i) {
    if (!g_stats.fb[i]) continue;
    fb += fmt::format(" {}={}", FastFallbackName(i), g_stats.fb[i]);
  }
  REXGPU_INFO(
      "[jade-native-fast] mode={} lean={} direct_mode={} entries={} evict={} delegate_hits={} "
      "nodraw={} async={} rt_rederive={} tex_restore_fail={} blob_refused={} pso_table_hit={} "
      "shader_recheck={} recheck_ne={} direct={} refused env={} nodirect={} rt_skip={} "
      "ib_skip={} ovl_skip={} ovl_live rt/blob/other={}/{}/{} tex_skip={} tex_live={} "
      "tex_gen_miss={} tex_outdated={} direct_di_skip={} direct_sys_skip={} direct_ia={} "
      "direct_env_refresh={} env_stamp={} | fallback{}",
      REXCVAR_GET(gpu_native_exec), REXCVAR_GET(gpu_native_fast_lean),
      REXCVAR_GET(gpu_native_direct), g_stats.entries, g_stats.evictions, g_stats.delegate_hits,
      g_stats.nodraw, g_stats.nodraw_async, g_stats.rt_rederive, g_stats.tex_restore_fail,
      g_stats.blob_refused, g_stats.pso_table_hit, g_stats.shader_recheck,
      g_stats.shader_recheck_ne, g_direct.draws, g_direct.refused_env, g_direct.nodirect,
      g_direct.rt_skip, g_direct.ib_skip, g_direct.overlay_skip, g_direct.overlay_live_rt,
      g_direct.overlay_live_blob, g_direct.overlay_live_other, g_direct.tex_skip,
      g_direct.tex_live, g_direct.tex_gen_miss, g_direct.tex_outdated, g_direct.di_skip,
      g_direct.sys_skip, g_direct.ia_live, g_direct.env_refresh, nr::NrDirectEnvStamp(),
      fb.empty() ? std::string(" none") : fb);
  bool any_direct = false;
  for (uint32_t i = 0; i < kFastDirectClasses; ++i) {
    any_direct |= (g_direct.eq[i] | g_direct.ne[i]) != 0;
  }
  if (any_direct) {
    std::string cls;
    for (uint32_t i = 0; i < kFastDirectClasses; ++i) {
      cls += fmt::format(" {}={}/{}", FastDirectName(i), g_direct.eq[i], g_direct.ne[i]);
    }
    REXGPU_INFO("[jade-native-direct]{} sys_ucp_stale={} sys_ucp_live_ne={}", cls,
                g_direct.sys_ucp_stale, g_direct.sys_ucp_live_ne);
  }
  bool any_cmp = g_cmp.skip_emit || g_cmp.fallback || g_cmp.no_packs;
  for (uint32_t i = 0; i < kFastCmpClasses; ++i) any_cmp |= (g_cmp.eq[i] | g_cmp.ne[i]) != 0;
  if (any_cmp) {
    std::string cls;
    for (uint32_t i = 0; i < kFastCmpClasses; ++i) {
      cls += fmt::format(" {}={}/{}", FastCmpName(i), g_cmp.eq[i], g_cmp.ne[i]);
    }
    REXGPU_INFO(
        "[jade-native-cmp] eq/ne{} | skip_direct={} no_packs={} fallback={} logged={} "
        "pso_desc_eq/ne/unavailable={}/{}/{} excluded_pool/instanced/indirect={}/{}/{}", cls,
        g_cmp.skip_emit, g_cmp.no_packs, g_cmp.fallback, g_cmp.logged,
        g_cmp.pso_desc_eq, g_cmp.pso_desc_ne, g_cmp.pso_desc_unavailable,
        g_cmp.excluded_pool, g_cmp.excluded_instanced, g_cmp.excluded_indirect);
  }
}

}  // namespace rex::graphics::d3d12
