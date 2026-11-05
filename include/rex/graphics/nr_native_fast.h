#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <rex/graphics/nr_native_submit.h>

// R2 native executor v2 (gpu_native_exec 1 / 2), the pure layer.
//
// Everything here is header-only and free of D3D12 so the offline test
// (tools/test-nr-native-fast.py) can compile it with plain clang++:
//   - the retail draw registers a marker stands for (VGT_DRAW_INITIATOR,
//     VGT_DMA_BASE, VGT_DMA_SIZE), which the record carries only as fields;
//   - the raw overlay of a record onto a register array with a save list and
//     the change classes the command processor must invalidate;
//   - the object key and PSO key compositions (which words go in, in which
//     order), the texture key, the shader table and the object table.
// The SDK side (src/graphics/d3d12/nr_native_fast.cpp) hashes the composed
// key words with XXH3; the tables here take any 64/128-bit key.

namespace rex::graphics::nr_native {

// Bytes the SDK keeps per object entry for the [dcache]-format bindings
// record (sampler params and heap indices, layout uids, descriptor-indices
// packs); command_processor.cpp static_asserts that a DcEntry fits.
constexpr size_t kFastBlobBytes = 1536;

// ---- Register indices this layer needs (Xenos register table) ----
constexpr uint32_t kFastRegVgtDmaBase = 0x21FA;
constexpr uint32_t kFastRegVgtDmaSize = 0x21FB;
constexpr uint32_t kFastRegVgtDrawInitiator = 0x21FC;
constexpr uint32_t kFastRegFloatBase = 0x4000;   // SHADER_CONSTANT_000_X
constexpr uint32_t kFastRegFloatEnd = 0x4800;    // exclusive (512 float4)
constexpr uint32_t kFastRegFetchBase = 0x4800;   // SHADER_CONSTANT_FETCH_00_0
constexpr uint32_t kFastRegFetchEnd = 0x48C0;    // exclusive (32 x 6 dwords)
constexpr uint32_t kFastRegBoolBase = 0x4900;    // SHADER_CONSTANT_BOOL_000_031
constexpr uint32_t kFastRegLoopEnd = 0x4928;     // exclusive (8 bool + 32 loop)
constexpr uint32_t kFastRegSurfaceInfo = 0x2000;
constexpr uint32_t kFastRegColorInfo[4] = {0x2001, 0x2003, 0x2004, 0x2005};
constexpr uint32_t kFastRegDepthInfo = 0x2002;
constexpr uint32_t kFastRegModeControl = 0x2208;

// ---- The retail draw registers of a record ----
// The retail DRAW_INDX the marker replaces writes these three registers
// (jade_native_record.h BuildIndexedDraw is the producer's formula): source
// select DMA, major mode implicit, index size bit 11, count in bits 16..31;
// the size dword counts 16-bit words for 32-bit indices (count << 1), exactly
// as the retail packet carries it. An auto-index record writes only the
// initiator (source select 2).
struct FastDrawRegs {
  uint32_t initiator = 0;
  uint32_t dma_base = 0;
  uint32_t dma_size = 0;
  bool indexed = false;
};
constexpr bool FastDrawCountOk(uint32_t index_count) { return index_count <= 0xFFFFu; }
constexpr FastDrawRegs FastSynthDrawRegs(uint32_t prim_type, uint32_t index_count, uint32_t flags,
                                         uint32_t index_base, uint32_t index_endian) {
  FastDrawRegs d{};
  d.indexed = (flags & kNrNativeIndexed) != 0;
  const bool i32 = (flags & kNrNativeIndex32) != 0;
  d.initiator = ((index_count & 0xFFFFu) << 16) | (prim_type & 0x3Fu) | (i32 ? 0x800u : 0u) |
                (d.indexed ? 0u : (2u << 6));
  if (d.indexed) {
    d.dma_base = index_base;
    d.dma_size = (i32 ? ((index_count << 1) & 0x00FFFFFEu) : (index_count & 0x00FFFFFFu)) |
                 ((index_endian & 3u) << 30);
  }
  return d;
}

// ---- Raw overlay ----
// One saved register for the restore (newest first undoes duplicates).
struct FastSaved {
  uint32_t index;
  uint32_t value;
};
// Change classes: what a changed value in each register range invalidates on
// the command processor (WriteRegister's D3D12 tail). Only a write whose value
// DIFFERS from the current one is a change (an equal store needs nothing).
struct FastChanges {
  uint64_t float_bits[8] = {};    // float4 constant c (0..511): bit c (VS 0..255, PS 256..511)
  uint32_t tex_fetch = 0;         // texture fetch slot (0..31) whose 6 dwords changed
  uint64_t vtx_fetch[2] = {};     // vertex fetch index (0..95, 2 dwords each) that changed
  bool bool_loop = false;
  uint32_t changed = 0;           // changed dwords (any class)
  void Clear() { *this = FastChanges{}; }
  bool Any() const { return changed != 0; }
};
struct FastOverlayStats {
  uint32_t n_saved = 0;
  uint32_t regs_applied = 0, consts_applied = 0;
  uint32_t skipped_low = 0, skipped_high = 0, skipped_const = 0;
};
constexpr uint32_t kFastMaxSaved = kNrNativeMaxRegisters + 4 * kNrNativeMaxConstants + 3;

inline void FastClassify(uint32_t index, FastChanges& c) {
  ++c.changed;
  if (index >= kFastRegFloatBase && index < kFastRegFloatEnd) {
    const uint32_t k = (index - kFastRegFloatBase) >> 2;
    c.float_bits[k >> 6] |= uint64_t(1) << (k & 63);
  } else if (index >= kFastRegFetchBase && index < kFastRegFetchEnd) {
    const uint32_t off = index - kFastRegFetchBase;
    c.tex_fetch |= uint32_t(1) << (off / 6);
    const uint32_t v = off >> 1;
    c.vtx_fetch[v >> 6] |= uint64_t(1) << (v & 63);
  } else if (index >= kFastRegBoolBase && index < kFastRegLoopEnd) {
    c.bool_loop = true;
  }
}

// Stores one dword, saving the previous value and classifying a change.
inline void FastStore(uint32_t* rf, uint32_t index, uint32_t value, FastSaved* saved,
                      FastOverlayStats& st, FastChanges& ch) {
  const uint32_t old = rf[index];
  saved[st.n_saved++] = {index, old};
  if (old != value) {
    rf[index] = value;
    FastClassify(index, ch);
  }
}

// Applies the record onto `rf` (rf_count registers): the registers in
// [kNrNativeRegisterFloor, min(kNrNativeRegisterCeiling, rf_count)), the float
// constants (512 max), then the three draw registers when `draw` is non-null.
// The same skip rules and counters as the v1 delegate body. `saved` must hold
// kFastMaxSaved entries. `constants` false (the lean path): the float
// constants are counted exactly as before but never stored; the caller
// composes them from the record (FastComposeFloatsRecord).
inline void FastOverlayApply(uint32_t* rf, uint32_t rf_count, const NrNativeDrawRecord& rec,
                             const FastDrawRegs* draw, FastSaved* saved, FastOverlayStats& st,
                             FastChanges& ch, bool constants = true) {
  st = FastOverlayStats{};
  const uint32_t nr = rec.register_count <= kNrNativeMaxRegisters ? rec.register_count : 0;
  const uint32_t nc = rec.constant_count <= kNrNativeMaxConstants ? rec.constant_count : 0;
  for (uint32_t i = 0; i < nr; ++i) {
    const uint32_t index = rec.registers[i].index;
    if (index < kNrNativeRegisterFloor) {
      ++st.skipped_low;
      continue;
    }
    if (!NrNativeRegisterWritable(index) || index >= rf_count) {
      ++st.skipped_high;
      continue;
    }
    FastStore(rf, index, rec.registers[i].value, saved, st, ch);
    ++st.regs_applied;
  }
  for (uint32_t i = 0; i < nc; ++i) {
    const uint32_t base = NrNativeConstantRegister(rec.constants[i].reg);
    if (!base || base + 3 >= rf_count) {
      ++st.skipped_const;
      continue;
    }
    if (constants) {
      for (uint32_t k = 0; k < 4; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &rec.constants[i].v[k], sizeof(bits));
        FastStore(rf, base + k, bits, saved, st, ch);
      }
    }
    ++st.consts_applied;
  }
  if (draw) {
    if (kFastRegVgtDrawInitiator < rf_count) {
      FastStore(rf, kFastRegVgtDrawInitiator, draw->initiator, saved, st, ch);
    }
    if (draw->indexed && kFastRegVgtDmaSize < rf_count) {
      FastStore(rf, kFastRegVgtDmaBase, draw->dma_base, saved, st, ch);
      FastStore(rf, kFastRegVgtDmaSize, draw->dma_size, saved, st, ch);
    }
  }
}

// Undoes FastOverlayApply, newest first; `ch` receives the classes the
// restore changes (the same registers, back to their previous values).
inline void FastOverlayRestore(uint32_t* rf, const FastSaved* saved, uint32_t n_saved,
                               FastChanges& ch) {
  while (n_saved) {
    const FastSaved& s = saved[--n_saved];
    if (rf[s.index] != s.value) {
      rf[s.index] = s.value;
      FastClassify(s.index, ch);
    }
  }
}

// Packs the float4 constants a shader reads (its 4 x 64-bit bitmap), in the
// same order as the constant buffer the translated shader indexes
// (nr::BindComposeFloats / UpdateBindings). Returns the bytes written.
inline uint32_t FastComposeFloats(const uint32_t* rf, uint32_t base, const uint64_t bitmap[4],
                                  void* out, uint32_t out_cap) {
  uint8_t* dst = static_cast<uint8_t*>(out);
  uint32_t written = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint64_t bits = bitmap[i];
    while (bits) {
      const uint32_t b = uint32_t(__builtin_ctzll(bits));
      bits &= bits - 1;
      if (written + 16 > out_cap) return written;
      std::memcpy(dst + written, &rf[base + (i << 8) + (b << 2)], 16);
      written += 16;
    }
  }
  return written;
}

// ---- Lean constants (gpu_native_fast_lean 1) ----
// The record's float4 constants never enter the live register file: the
// float constant buffers are composed from the record OVER the live file.
// Same bytes as overlay + FastComposeFloats: last write of a reg wins, a reg
// at or above 512 is skipped. `stamp` makes Build O(record), no clear.
struct FastConstMap {
  uint32_t serial = 0;
  uint32_t stamp[kNrNativeConstantCount] = {};
  uint16_t idx[kNrNativeConstantCount] = {};
  uint32_t stage_n[2] = {};  // in-range constants per stage (0 vs, 1 ps)
  void Build(const NrNativeDrawRecord& rec) {
    if (++serial == 0) {
      std::memset(stamp, 0, sizeof(stamp));
      serial = 1;
    }
    stage_n[0] = stage_n[1] = 0;
    const uint32_t nc = rec.constant_count <= kNrNativeMaxConstants ? rec.constant_count : 0;
    for (uint32_t i = 0; i < nc; ++i) {
      const uint32_t reg = rec.constants[i].reg;
      if (reg >= kNrNativeConstantCount) continue;
      stamp[reg] = serial;
      idx[reg] = uint16_t(i);
      ++stage_n[reg >> 8];
    }
  }
  // The record entry for float4 `reg` (0..511), or UINT32_MAX.
  uint32_t Find(uint32_t reg) const {
    return reg < kNrNativeConstantCount && stamp[reg] == serial ? idx[reg] : UINT32_MAX;
  }
};
// FastComposeFloats for stage 0 (vs, 0x4000) or 1 (ps, 0x4400) with the
// record's constants taking precedence over the live file.
// `record_hits` (optional) receives the number of composed float4s taken
// from the record: nonzero means the upload differs from the live file.
inline uint32_t FastComposeFloatsRecord(const uint32_t* rf, uint32_t stage, const uint64_t bitmap[4],
                                        const NrNativeDrawRecord& rec, const FastConstMap& map,
                                        void* out, uint32_t out_cap,
                                        uint32_t* record_hits = nullptr) {
  uint8_t* dst = static_cast<uint8_t*>(out);
  uint32_t hits = 0;
  const uint32_t base = kFastRegFloatBase + (stage ? 0x400u : 0u);
  const uint32_t reg_base = stage ? 256u : 0u;
  uint32_t written = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint64_t bits = bitmap[i];
    while (bits) {
      const uint32_t b = uint32_t(__builtin_ctzll(bits));
      bits &= bits - 1;
      if (written + 16 > out_cap) {
        if (record_hits) *record_hits = hits;
        return written;
      }
      const uint32_t k = map.Find(reg_base + (i << 6) + b);
      if (k != UINT32_MAX) {
        std::memcpy(dst + written, rec.constants[k].v, 16);
        ++hits;
      } else {
        std::memcpy(dst + written, &rf[base + (i << 8) + (b << 2)], 16);
      }
      written += 16;
    }
  }
  if (record_hits) *record_hits = hits;
  return written;
}

// ---- Hashing (portable, used for the table indices and the offline test) ----
constexpr uint64_t FastMix64(uint64_t h) {
  h ^= h >> 33;
  h *= 0xFF51AFD7ED558CCDull;
  h ^= h >> 33;
  h *= 0xC4CEB9FE1A85EC53ull;
  h ^= h >> 33;
  return h;
}
inline uint64_t FastHash64(const uint32_t* w, uint32_t n, uint64_t seed) {
  uint64_t h = seed ^ (uint64_t(n) * 0x9E3779B97F4A7C15ull);
  for (uint32_t i = 0; i < n; ++i) {
    h = FastMix64(h ^ (uint64_t(w[i]) + 0x9E3779B97F4A7C15ull + (uint64_t(i) << 32)));
  }
  return FastMix64(h);
}
struct FastKey128 {
  uint64_t h1 = 0, h2 = 0;
  bool operator==(const FastKey128& o) const { return h1 == o.h1 && h2 == o.h2; }
};
inline FastKey128 FastHash128(const uint32_t* w, uint32_t n) {
  return {FastHash64(w, n, 0x243F6A8885A308D3ull), FastHash64(w, n, 0x13198A2E03707344ull)};
}

// ---- Object key composition ----
// The words the fast cache keys a draw on: the record's own draw identity,
// the two shader ucode hashes, the pass registers, the caller's register
// ranges (the [dcache] identity ranges, handed in so the two can never
// drift), the bool/loop constants, every used texture fetch constant, the
// three draw registers as overlaid, and the caller's variant word (the IA
// phase). NOT in it, on purpose, as in the [dcache] identity: the float
// constants (uploaded live) and the vertex fetch constants (animated geometry
// rotates its vertex buffer every frame; residency and the IA views are live).
struct FastKeyInputs {
  const uint32_t* rf = nullptr;             // the overlaid register file
  const uint32_t (*ranges)[2] = nullptr;    // [start, end) pairs
  uint32_t range_count = 0;
  uint64_t vs_hash = 0, ps_hash = 0;        // ucode hashes (ps 0 = none)
  uint32_t prim_type = 0, index_count = 0, flags = 0;
  uint32_t index_base = 0, index_endian = 0, index_length = 0;
  uint32_t tex_mask = 0;                    // used texture fetch slots
  uint32_t variant = 0;
};
constexpr uint32_t kFastKeyMaxWords = 1024;
// `read(index)` supplies the register words: the overlaid file, or the record
// map over the live file (FastRegMap) when the overlay is not applied.
template <class Read>
inline uint32_t FastComposeObjectKeyRead(const FastKeyInputs& in, Read read, uint32_t* out,
                                         uint32_t cap) {
  uint32_t n = 0;
  auto put = [&](uint32_t v) {
    if (n < cap) out[n] = v;
    ++n;
  };
  put(0x4A4E4631u);  // format tag 'JNF1'
  put(in.prim_type);
  put(in.index_count);
  put(in.flags & (kNrNativeIndexed | kNrNativeIndex32));
  put(in.index_base);
  put(in.index_endian);
  put(in.index_length);
  put(uint32_t(in.vs_hash));
  put(uint32_t(in.vs_hash >> 32));
  put(uint32_t(in.ps_hash));
  put(uint32_t(in.ps_hash >> 32));
  put(in.variant);
  put(in.tex_mask);
  put(read(kFastRegSurfaceInfo));
  put(read(kFastRegModeControl));
  for (uint32_t i = 0; i < 4; ++i) put(read(kFastRegColorInfo[i]));
  put(read(kFastRegDepthInfo));
  for (uint32_t r = 0; r < in.range_count; ++r) {
    for (uint32_t i = in.ranges[r][0]; i < in.ranges[r][1]; ++i) put(read(i));
  }
  for (uint32_t i = kFastRegBoolBase; i < kFastRegLoopEnd; ++i) put(read(i));
  uint32_t tm = in.tex_mask;
  while (tm) {
    const uint32_t t = uint32_t(__builtin_ctz(tm));
    tm &= tm - 1;
    for (uint32_t j = 0; j < 6; ++j) put(read(kFastRegFetchBase + 6 * t + j));
  }
  put(read(kFastRegVgtDrawInitiator));
  put(read(kFastRegVgtDmaBase));
  put(read(kFastRegVgtDmaSize));
  return n;
}
inline uint32_t FastComposeObjectKey(const FastKeyInputs& in, uint32_t* out, uint32_t cap) {
  const uint32_t* rf = in.rf;
  return FastComposeObjectKeyRead(in, [rf](uint32_t i) { return rf[i]; }, out, cap);
}

// ---- Record register map (direct submission without the overlay) ----
// What the overlay would have put into the live file, readable without the
// stores: the same skip rules as FastOverlayApply (floor, writable, in
// range), last write wins, the three synthesized draw registers last.
// Stamp-based: Build is O(record), Read is one compare over the live file.
struct FastRegMap {
  uint32_t serial = 0;
  uint32_t rf_count = 0;
  std::vector<uint32_t> stamp, value;
  void Init(uint32_t count) {
    rf_count = count;
    stamp.assign(count, 0u);
    value.assign(count, 0u);
    serial = 0;
  }
  void Put(uint32_t index, uint32_t v) {
    if (index >= rf_count) return;
    stamp[index] = serial;
    value[index] = v;
  }
  void Build(const NrNativeDrawRecord& rec, const FastDrawRegs* draw) {
    if (++serial == 0) {
      std::fill(stamp.begin(), stamp.end(), 0u);
      serial = 1;
    }
    const uint32_t nr = rec.register_count <= kNrNativeMaxRegisters ? rec.register_count : 0;
    for (uint32_t i = 0; i < nr; ++i) {
      const uint32_t index = rec.registers[i].index;
      if (index < kNrNativeRegisterFloor || !NrNativeRegisterWritable(index) || index >= rf_count) continue;
      Put(index, rec.registers[i].value);
    }
    if (draw) {
      Put(kFastRegVgtDrawInitiator, draw->initiator);
      if (draw->indexed) {
        Put(kFastRegVgtDmaBase, draw->dma_base);
        Put(kFastRegVgtDmaSize, draw->dma_size);
      }
    }
  }
  bool Has(uint32_t index) const { return index < rf_count && stamp[index] == serial; }
  uint32_t Read(const uint32_t* rf, uint32_t index) const {
    return Has(index) ? value[index] : rf[index];
  }
};

// The change classes FastOverlayApply raises (and its restore raises again)
// for this record over `rf`, without a store: a record register, or a
// synthesized draw register, whose value differs from the live one. The
// class set is exactly the apply's (a duplicate write differing at any point
// of the sequence lands in the same class). `constants` as in the apply.
inline void FastOverlayClassify(const uint32_t* rf, uint32_t rf_count, const NrNativeDrawRecord& rec,
                                const FastDrawRegs* draw, FastChanges& ch, bool constants = true) {
  const uint32_t nr = rec.register_count <= kNrNativeMaxRegisters ? rec.register_count : 0;
  const uint32_t nc = rec.constant_count <= kNrNativeMaxConstants ? rec.constant_count : 0;
  for (uint32_t i = 0; i < nr; ++i) {
    const uint32_t index = rec.registers[i].index;
    if (index < kNrNativeRegisterFloor) continue;
    if (!NrNativeRegisterWritable(index) || index >= rf_count) continue;
    if (rf[index] != rec.registers[i].value) FastClassify(index, ch);
  }
  if (constants) {
    for (uint32_t i = 0; i < nc; ++i) {
      const uint32_t base = NrNativeConstantRegister(rec.constants[i].reg);
      if (!base || base + 3 >= rf_count) continue;
      for (uint32_t k = 0; k < 4; ++k) {
        uint32_t bits;
        std::memcpy(&bits, &rec.constants[i].v[k], sizeof(bits));
        if (rf[base + k] != bits) FastClassify(base + k, ch);
      }
    }
  }
  if (draw) {
    if (kFastRegVgtDrawInitiator < rf_count && rf[kFastRegVgtDrawInitiator] != draw->initiator) {
      FastClassify(kFastRegVgtDrawInitiator, ch);
    }
    if (draw->indexed && kFastRegVgtDmaSize < rf_count) {
      if (rf[kFastRegVgtDmaBase] != draw->dma_base) FastClassify(kFastRegVgtDmaBase, ch);
      if (rf[kFastRegVgtDmaSize] != draw->dma_size) FastClassify(kFastRegVgtDmaSize, ch);
    }
  }
}

// The texture fetch constants of `slots` (a bit per slot, 6 dwords each)
// from the map into the live file, with a restore: the texture cache's
// snapshot restore validates against the live fetch constants, and with the
// overlay skipped those would be stale for the record's changed slots.
struct FastSparseTexOverlay {
  uint32_t n = 0;
  uint32_t index[32 * 6] = {};
  uint32_t saved[32 * 6] = {};
  void Apply(uint32_t* rf, const FastRegMap& map, uint32_t slots) {
    n = 0;
    while (slots) {
      const uint32_t s = uint32_t(__builtin_ctz(slots));
      slots &= slots - 1;
      for (uint32_t j = 0; j < 6; ++j) {
        const uint32_t i = kFastRegFetchBase + 6 * s + j;
        if (!map.Has(i)) continue;
        index[n] = i;
        saved[n] = rf[i];
        rf[i] = map.value[i];
        ++n;
      }
    }
  }
  void Restore(uint32_t* rf) {
    while (n) {
      --n;
      rf[index[n]] = saved[n];
    }
  }
};

// ---- PSO key composition ----
// Every input of PipelineCache::ConfigurePipeline for one draw: the two
// translations (shader + modification), the primitive processor's host
// facts, the normalized depth control and color mask, the bound render
// target bits and formats, and the registers GetCurrentStateDescription and
// its helpers read (IsPrimitivePolygonal, IsRasterizationPotentiallyDone,
// GetPreferredFacePolygonOffset, the stencil masks, the blend controls, the
// color infos, the depth format, MSAA).
constexpr uint32_t kFastPsoRegs[] = {
    0x2000,                          // RB_SURFACE_INFO (msaa, pitch)
    0x2001, 0x2003, 0x2004, 0x2005,  // RB_COLOR_INFO 0..3
    0x2002,                          // RB_DEPTH_INFO
    0x2180,                          // SQ_PROGRAM_CNTL (vs export mode)
    0x2284,                          // VGT_OUTPUT_PATH_CNTL
    0x21FC,                          // VGT_DRAW_INITIATOR (polygonal)
    0x2201, 0x2209, 0x220A, 0x220B,  // RB_BLENDCONTROL 0..3
    0x2204,                          // PA_CL_CLIP_CNTL (depth clip)
    0x2205,                          // PA_SU_SC_MODE_CNTL (cull, fill, offsets)
    0x2208,                          // RB_MODECONTROL
    0x210C, 0x210D,                  // RB_STENCILREFMASK_BF, RB_STENCILREFMASK
    0x2380, 0x2381, 0x2382, 0x2383,  // PA_SU_POLY_OFFSET front/back scale/offset
};
struct FastPsoInputs {
  const uint32_t* rf = nullptr;
  uint64_t vs_translation = 0, ps_translation = 0;  // object identities
  uint32_t host_primitive_type = 0, host_vertex_shader_type = 0, tessellation_mode = 0;
  uint32_t primitive_reset = 0, host_index_format = 0;
  uint32_t normalized_depth_control = 0, normalized_color_mask = 0;
  uint32_t rt_bits = 0;
  uint32_t rt_formats[5] = {};
  uint32_t path = 0;
};
constexpr uint32_t kFastPsoKeyWords = 18 + sizeof(kFastPsoRegs) / sizeof(uint32_t);
inline uint32_t FastComposePsoKey(const FastPsoInputs& in, uint32_t* out) {
  uint32_t n = 0;
  out[n++] = uint32_t(in.vs_translation);
  out[n++] = uint32_t(in.vs_translation >> 32);
  out[n++] = uint32_t(in.ps_translation);
  out[n++] = uint32_t(in.ps_translation >> 32);
  out[n++] = in.host_primitive_type;
  out[n++] = in.host_vertex_shader_type;
  out[n++] = in.tessellation_mode;
  out[n++] = in.primitive_reset;
  out[n++] = in.host_index_format;
  out[n++] = in.normalized_depth_control;
  out[n++] = in.normalized_color_mask;
  out[n++] = in.rt_bits;
  for (uint32_t i = 0; i < 5; ++i) {
    // A format is part of the description only when its bit is bound.
    out[n++] = (in.rt_bits & (1u << i)) ? in.rt_formats[i] : 0xFFFFFFFFu;
  }
  out[n++] = in.path;
  for (uint32_t r : kFastPsoRegs) out[n++] = in.rf[r];
  return n;
}
inline uint64_t FastPsoKey(const FastPsoInputs& in) {
  uint32_t w[kFastPsoKeyWords];
  const uint32_t n = FastComposePsoKey(in, w);
  return FastHash64(w, n, 0x0A6B7C8D9E0F1A2Bull);
}

// ---- Texture key ----
// A texture binding is a pure function of its 6 fetch dwords (plus the
// texture cache's own state, guarded by the destroy generation): the key is
// the words themselves; the table index is their hash.
inline uint64_t FastTexKey(const uint32_t fetch[6]) { return FastHash64(fetch, 6, 0x7E3A9C1D5B2F4086ull); }

// ---- Shader table ----
// Direct-mapped, keyed by (vs address, vs dwords, ps address, ps dwords); an
// entry also keeps four sampled ucode dwords of each stage (guest byte order)
// so a reused address with other ucode misses. Every kRecheckPeriod-th hit of
// an entry asks the caller for a full ucode check (LoadShader's hash).
constexpr uint32_t kFastShaderBits = 10;
constexpr uint32_t kFastShaderRecheckPeriod = 256;
inline void FastSampleUcode(const uint32_t* p, uint32_t n, uint32_t out[4]) {
  if (!p || !n) {
    out[0] = out[1] = out[2] = out[3] = 0;
    return;
  }
  out[0] = p[0];
  out[1] = p[n / 3];
  out[2] = p[(2 * n) / 3];
  out[3] = p[n - 1];
}
inline uint32_t FastShaderSlot(uint32_t vs_addr, uint32_t vs_dwords, uint32_t ps_addr,
                               uint32_t ps_dwords) {
  const uint32_t w[4] = {vs_addr, vs_dwords, ps_addr, ps_dwords};
  return uint32_t(FastHash64(w, 4, 0x5BD1E9955BD1E995ull)) & ((1u << kFastShaderBits) - 1);
}
template <class S>
struct FastShaderTable {
  struct Entry {
    uint32_t vs_addr = 0, vs_dwords = 0, ps_addr = 0, ps_dwords = 0;
    uint32_t gen = 0, hits = 0;
    uint32_t vs_sample[4] = {}, ps_sample[4] = {};
    // Guest memory can reuse a shader address without changing a sampled
    // word. Keep the exact source bytes, independent of Shader's endian.
    std::vector<uint32_t> vs_source, ps_source;
    S* vs = nullptr;
    S* ps = nullptr;
    bool valid = false;
  };
  Entry e[1u << kFastShaderBits];
  enum class Result { kHit, kHitRecheck, kMiss };
  // vs_ucode / ps_ucode: the guest dwords at the two addresses (null when absent).
  Result Lookup(uint32_t vs_addr, uint32_t vs_dwords, uint32_t ps_addr, uint32_t ps_dwords,
                const uint32_t* vs_ucode, const uint32_t* ps_ucode, uint32_t gen, S** vs_out,
                S** ps_out) {
    Entry& x = e[FastShaderSlot(vs_addr, vs_dwords, ps_addr, ps_dwords)];
    if (!x.valid || x.gen != gen || x.vs_addr != vs_addr || x.vs_dwords != vs_dwords ||
        x.ps_addr != ps_addr || x.ps_dwords != ps_dwords) {
      return Result::kMiss;
    }
    uint32_t vsm[4], psm[4];
    FastSampleUcode(vs_ucode, vs_dwords, vsm);
    FastSampleUcode(ps_ucode, ps_dwords, psm);
    if (std::memcmp(vsm, x.vs_sample, sizeof(vsm)) || std::memcmp(psm, x.ps_sample, sizeof(psm))) {
      return Result::kMiss;
    }
    if (x.vs_source.size() != vs_dwords || x.ps_source.size() != ps_dwords ||
        (vs_dwords && (!vs_ucode || std::memcmp(vs_ucode, x.vs_source.data(),
                                               size_t(vs_dwords) * sizeof(uint32_t)))) ||
        (ps_dwords && (!ps_ucode || std::memcmp(ps_ucode, x.ps_source.data(),
                                               size_t(ps_dwords) * sizeof(uint32_t))))) {
      return Result::kMiss;
    }
    *vs_out = x.vs;
    *ps_out = x.ps;
    return (++x.hits % kFastShaderRecheckPeriod) == 0 ? Result::kHitRecheck : Result::kHit;
  }
  void Insert(uint32_t vs_addr, uint32_t vs_dwords, uint32_t ps_addr, uint32_t ps_dwords,
              const uint32_t* vs_ucode, const uint32_t* ps_ucode, uint32_t gen, S* vs, S* ps) {
    Entry& x = e[FastShaderSlot(vs_addr, vs_dwords, ps_addr, ps_dwords)];
    x.vs_addr = vs_addr;
    x.vs_dwords = vs_dwords;
    x.ps_addr = ps_addr;
    x.ps_dwords = ps_dwords;
    x.gen = gen;
    x.hits = 0;
    FastSampleUcode(vs_ucode, vs_dwords, x.vs_sample);
    FastSampleUcode(ps_ucode, ps_dwords, x.ps_sample);
    x.vs_source.clear();
    x.ps_source.clear();
    if (vs_ucode && vs_dwords) x.vs_source.assign(vs_ucode, vs_ucode + vs_dwords);
    if (ps_ucode && ps_dwords) x.ps_source.assign(ps_ucode, ps_ucode + ps_dwords);
    x.vs = vs;
    x.ps = ps;
    x.valid = true;
  }
  void Clear() {
    for (Entry& x : e) x.valid = false;
  }
};

// ---- Object table ----
// Two-way set associative over a 128-bit key; the older way (by frame stamp)
// is replaced. `T` needs members h1, h2, stamp (uint32_t) and valid (bool or
// integer); the table never interprets the payload.
template <class T, uint32_t kSetBits>
struct FastObjectTable {
  static constexpr uint32_t kSets = 1u << kSetBits;
  T* ways = nullptr;  // 2 * kSets entries, owned by the caller
  static uint32_t Set(uint64_t h1) { return uint32_t(h1 ^ (h1 >> 29) ^ (h1 >> 47)) & (kSets - 1); }
  T* Find(const FastKey128& k, uint32_t stamp) {
    T* s = ways + 2 * Set(k.h1);
    for (uint32_t w = 0; w < 2; ++w) {
      if (s[w].valid && s[w].h1 == k.h1 && s[w].h2 == k.h2) {
        s[w].stamp = stamp;
        return &s[w];
      }
    }
    return nullptr;
  }
  // The way to (re)fill for this key: its own entry when present (even an
  // invalidated one), else an empty way, else the least recently stamped.
  T* Victim(const FastKey128& k) {
    T* s = ways + 2 * Set(k.h1);
    for (uint32_t w = 0; w < 2; ++w) {
      if (s[w].h1 == k.h1 && s[w].h2 == k.h2) return &s[w];
    }
    if (!s[0].valid) return &s[0];
    if (!s[1].valid) return &s[1];
    return s[0].stamp <= s[1].stamp ? &s[0] : &s[1];
  }
};

// ---- Fallback reasons (every refusal of the fast path runs the v1 body) ----
enum FastFallback : uint32_t {
  kFastFbNone = 0,
  kFastFbParallel,    // parallel-record segments own the draw path
  kFastFbInstance,    // gpu_instance coalescer on
  kFastFbNoSwap,      // bindless swap not armed (bindful or cold mirrors)
  kFastFbTile,        // a tile recording / replay is live
  kFastFbSpan,        // a span recording / replay is live
  kFastFbCount16,     // index count above 65535
  kFastFbMemexport,   // a shader exports to memory
  kFastFbHostVs,      // tessellated or non-kVertex host vertex shader
  kFastFbConverted,   // converted index buffer (frame-scoped)
  kFastFbLineLoop,    // line loop closing index
  kFastFbTexCount,    // more used textures than an entry holds
  kFastFbIa,          // the entry's IA variant cannot acquire its views now
  kFastFbEdram,       // not color+depth rasterization
  kFastFbProcess,     // primitive processing failed
  kFastFbReuse,       // the NR-RUB bundle machinery keys on the walk's stop, not a marker
  kFastFbReasons
};
inline const char* FastFallbackName(uint32_t r) {
  static const char* const kNames[kFastFbReasons] = {
      "none", "parallel", "instance", "noswap", "tile",  "span",  "count16", "memexport",
      "hostvs", "converted", "lineloop", "texcount", "ia", "edram", "process", "reuse"};
  return r < kFastFbReasons ? kNames[r] : "?";
}

// ---- Compare classes (gpu_native_exec 2) ----
enum FastCmpClass : uint32_t {
  kFastCmpShader = 0,  // shader table vs LoadShader
  kFastCmpPso,         // the pipeline object bound for the draw
  kFastCmpRootSig,     // the root signature
  kFastCmpTex,         // per used slot: texture, signed texture, SRV indices, swizzle, signs
  kFastCmpDiV,         // vertex descriptor-indices pack (SRV + sampler heap indices)
  kFastCmpDiP,         // pixel descriptor-indices pack
  kFastCmpFloatV,      // vertex float constant buffer bytes
  kFastCmpFloatP,      // pixel float constant buffer bytes
  kFastCmpBoolLoop,    // bool/loop constant buffer bytes
  kFastCmpFetch,       // fetch constant buffer bytes
  kFastCmpViewport,    // host viewport info
  kFastCmpClasses
};
inline const char* FastCmpName(uint32_t c) {
  static const char* const kNames[kFastCmpClasses] = {"shader", "pso", "rootsig", "tex", "div",
                                                      "dip",    "fv",  "fp",      "bl",  "ft",
                                                      "vp"};
  return c < kFastCmpClasses ? kNames[c] : "?";
}


// ---- Direct submission (gpu_native_direct), the pure pieces ----
// An object entry bakes the draw head IssueDrawImpl derives from keyed
// registers (viewport, scissor, stencil reference, Hi-Z survival, pool key,
// system constants); the command processor applies it through its own per
// list latches. What is never baked: the Hi-Z window direction, the blend
// factor (RB_BLEND_RED..ALPHA are not keyed) and the de-tile widen (it reads
// the frame's tap progress).

// Hi-Z survival mask from a normalized RB_DEPTHCONTROL value: bit 0 = a
// reversed-Z window (dir 1: GREATER / GREATEREQUAL) survives the draw, bit 1 =
// a normal window (LESS / LESSEQUAL) survives it. A window survives a draw
// that cannot move depth farther: no depth write, or a test monotone for its
// direction (EQUAL and NEVER for both). The open window's dir selects the bit
// at draw time (FastHizCloses).
constexpr uint32_t kFastDcZEnable = 1u << 1;
constexpr uint32_t kFastDcZWriteEnable = 1u << 2;
constexpr uint32_t kFastDcZFuncShift = 4;
constexpr uint32_t kFastDcBackfaceEnable = 1u << 7;
constexpr uint32_t FastHizMonoMask(uint32_t ndc) {
  constexpr uint32_t zw = kFastDcZEnable | kFastDcZWriteEnable;
  if ((ndc & zw) != zw) return 3u;
  const uint32_t zfunc = (ndc >> kFastDcZFuncShift) & 7u;  // xenos::CompareFunction
  const bool both = zfunc == 2u /*kEqual*/ || zfunc == 0u /*kNever*/;
  const bool rev = zfunc == 4u /*kGreater*/ || zfunc == 6u /*kGreaterEqual*/;
  const bool nrm = zfunc == 1u /*kLess*/ || zfunc == 3u /*kLessEqual*/;
  return (both || rev ? 1u : 0u) | (both || nrm ? 2u : 0u);
}
// True when a draw with survival mask `mask` closes the open window whose
// direction is `dir` (1 reversed, anything else normal).
constexpr bool FastHizCloses(uint32_t mask, uint32_t dir) {
  return ((mask >> (dir == 1u ? 0u : 1u)) & 1u) == 0u;
}

// The host viewport UpdateFixedFunctionState builds from a ViewportInfo: the
// integer offset / extent as floats, the depth range as derived (reversed-Z
// included: z_min / z_max are copied, never ordered).
struct FastViewport {
  float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f, z_min = 0.0f, z_max = 0.0f;
  constexpr bool operator==(const FastViewport&) const = default;
};
constexpr FastViewport FastHostViewport(const uint32_t xy_offset[2], const uint32_t xy_extent[2],
                                        float z_min, float z_max) {
  FastViewport v{};
  v.x = float(xy_offset[0]);
  v.y = float(xy_offset[1]);
  v.w = float(xy_extent[0]);
  v.h = float(xy_extent[1]);
  v.z_min = z_min;
  v.z_max = z_max;
  return v;
}

// The host scissor rectangle from a guest-pixel scissor (offset, extent) and
// the draw resolution scale, as IssueDrawImpl scales then
// UpdateFixedFunctionState converts (right = offset + extent, both scaled).
struct FastRect {
  int32_t left = 0, top = 0, right = 0, bottom = 0;
  constexpr bool operator==(const FastRect&) const = default;
};
constexpr FastRect FastHostScissor(const uint32_t offset[2], const uint32_t extent[2],
                                   uint32_t scale_x, uint32_t scale_y) {
  FastRect r{};
  const uint32_t x0 = offset[0] * scale_x;
  const uint32_t y0 = offset[1] * scale_y;
  const uint32_t w = extent[0] * scale_x;
  const uint32_t h = extent[1] * scale_y;
  r.left = int32_t(x0);
  r.top = int32_t(y0);
  r.right = int32_t(x0 + w);
  r.bottom = int32_t(y0 + h);
  return r;
}

// UpdateFixedFunctionState's stencil reference selection: the back-face
// register only when drawing only back faces (polygonal, backface_enable,
// cull_front set, cull_back clear in PA_SU_SC_MODE_CNTL).
constexpr bool FastStencilUsesBackFace(bool polygonal, uint32_t ndc, uint32_t pa_su_sc_mode_cntl) {
  const bool cull_front = (pa_su_sc_mode_cntl & 1u) != 0u;
  const bool cull_back = (pa_su_sc_mode_cntl & 2u) != 0u;
  return polygonal && (ndc & kFastDcBackfaceEnable) != 0u && cull_front && !cull_back;
}

// The system constants block's user clip planes are PACKED: the derivation
// writes the first popcount(PA_CL_CLIP_CNTL.ucp_ena) planes (none under
// clip_disable, bit 16) and leaves the rest as history, and the vertex shader
// (modification user_clip_plane_count, from the same keyed register) reads
// exactly those. The unread dwords are history, not state: an entry bakes
// them as zero, and the compare normalizes both sides the same way.
constexpr uint32_t kFastSysUcpFirstDword = 8;   // offsetof(NrSysConstants, user_clip_planes) / 4
constexpr uint32_t kFastSysUcpDwords = 24;      // 6 planes x 4
constexpr uint32_t FastSysUcpLiveDwords(uint32_t pa_cl_clip_cntl) {
  if (pa_cl_clip_cntl & (1u << 16)) return 0u;
  uint32_t n = 0;
  for (uint32_t i = 0; i < 6; ++i) n += (pa_cl_clip_cntl >> i) & 1u;
  return 4u * n;
}
constexpr void FastSysNormalizeUcp(uint32_t* block, uint32_t pa_cl_clip_cntl) {
  for (uint32_t d = FastSysUcpLiveDwords(pa_cl_clip_cntl); d < kFastSysUcpDwords; ++d) {
    block[kFastSysUcpFirstDword + d] = 0u;
  }
}

// The enabled planes patched from the draw's own PA_CL_UCP_0_X.. registers
// (24 dwords at 0x2388; not identity-keyed: the coefficients move per draw)
// into a block whose packed planes are otherwise zero, in the order
// NrSysUpdate packs them (ascending plane index, 4 dwords each).
constexpr uint32_t kFastRegPaClUcp0X = 0x2388;
constexpr void FastSysPatchUcp(uint32_t* block, uint32_t pa_cl_clip_cntl, const uint32_t ucp[24]) {
  if (pa_cl_clip_cntl & (1u << 16)) return;
  uint32_t* dst = block + kFastSysUcpFirstDword;
  for (uint32_t plane = 0; plane < 6; ++plane) {
    if (!((pa_cl_clip_cntl >> plane) & 1u)) continue;
    for (uint32_t k = 0; k < 4; ++k) dst[k] = ucp[plane * 4 + k];
    dst += 4;
  }
}

// The [xfer] read census verdict D3D12RenderTargetCache::Update derives per
// bound target from the normalized depth control, the normalized color mask
// and RB_BLENDCONTROL0..3 (all identity-keyed). Returns the used-target mask
// (bit 0 depth, bit 1 + i color i, as the base Update selects them for a
// rasterized draw); *reads_out bit i = target i reads its destination;
// reasons_out[i] the census tags (Z 1, stencil 2, blend 4, mask 8).
constexpr uint32_t kFastXferZ = 1u, kFastXferStencil = 2u, kFastXferBlend = 4u, kFastXferMask = 8u;
constexpr uint32_t FastXferVerdict(uint32_t ndc, uint32_t color_mask, const uint32_t blend[4],
                                   uint32_t* reads_out, uint8_t reasons_out[5]) {
  auto op_writes = [](uint32_t op) { return op == 2u /*kReplace*/ || op == 1u /*kZero*/; };
  auto reads_dest = [](uint32_t f) {
    return f == 8u /*kDstColor*/ || f == 9u /*kOneMinusDstColor*/ || f == 10u /*kDstAlpha*/ ||
           f == 11u /*kOneMinusDstAlpha*/ || f == 16u /*kSrcAlphaSaturate*/;
  };
  uint32_t used = 0, reads = 0;
  for (uint32_t i = 0; i < 5; ++i) reasons_out[i] = 0;
  const bool stencil_enable = (ndc & 1u) != 0u;
  const bool z_enable = (ndc & kFastDcZEnable) != 0u;
  if (z_enable || stencil_enable) {
    used |= 1u;
    const uint32_t zfunc = (ndc >> kFastDcZFuncShift) & 7u;
    const bool z_reads = z_enable && zfunc != 7u /*kAlways*/;
    bool stencil_reads = false;
    if (stencil_enable) {
      const bool backface = (ndc & kFastDcBackfaceEnable) != 0u;
      const uint32_t sfunc = (ndc >> 8) & 7u, szpass = (ndc >> 14) & 7u, szfail = (ndc >> 17) & 7u;
      const uint32_t sfunc_bf = (ndc >> 20) & 7u, szpass_bf = (ndc >> 26) & 7u,
                     szfail_bf = (ndc >> 29) & 7u;
      const bool front = sfunc == 7u && op_writes(szpass) && (!z_reads || op_writes(szfail));
      const bool back =
          !backface || (sfunc_bf == 7u && op_writes(szpass_bf) && (!z_reads || op_writes(szfail_bf)));
      stencil_reads = !(front && back);
    }
    if (z_reads || stencil_reads) reads |= 1u;
    reasons_out[0] = uint8_t((z_reads ? kFastXferZ : 0u) | (stencil_reads ? kFastXferStencil : 0u));
  }
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t nibble = (color_mask >> (4 * i)) & 0xFu;
    if (!nibble) continue;
    used |= 1u << (1 + i);
    const uint32_t b = blend[i];
    const uint32_t csrc = b & 0x1Fu, cop = (b >> 5) & 7u, cdst = (b >> 8) & 0x1Fu;
    const uint32_t asrc = (b >> 16) & 0x1Fu, aop = (b >> 21) & 7u, adst = (b >> 24) & 0x1Fu;
    const bool blend_reads = reads_dest(csrc) || cdst != 0u || cop == 2u /*kMin*/ ||
                             cop == 3u /*kMax*/ || reads_dest(asrc) || adst != 0u || aop == 2u ||
                             aop == 3u;
    const bool mask_reads = nibble != 0xFu;
    if (blend_reads || mask_reads) reads |= 1u << (1 + i);
    reasons_out[1 + i] = uint8_t((blend_reads ? kFastXferBlend : 0u) | (mask_reads ? kFastXferMask : 0u));
  }
  *reads_out = reads;
  return used;
}

// ---- Direct compare classes (gpu_native_direct 2) ----
enum FastDirectClass : uint32_t {
  kFastDirectVp = 0,   // host viewport: baked vs the full path's latch
  kFastDirectSc,       // host scissor: the baked widened + scaled rect vs the live widen's latch
  kFastDirectStencil,  // stencil reference
  kFastDirectBlend,    // blend factor: the direct read site vs the full path's latch
  kFastDirectSys,      // system constants block: entry vs the full derivation
  kFastDirectHiz,      // Hi-Z close verdict: mask bit vs the live rule
  kFastDirectPool,     // pool pass key + class: baked vs derived
  kFastDirectRt,       // render target latch: Update called anyway runs no body, binds the same
  kFastDirectXfer,     // the baked [xfer] verdict vs the SDK bitfield derivation
  kFastDirectTex,      // the restored slot bindings and descriptor indices vs the entry
  kFastDirectIb,       // index residency latch: the request made anyway succeeds
  kFastDirectOvl,      // the record map vs the overlaid file at every direct consumer
  kFastDirectTxl,      // texture generation latch: the live slots already equal the entry, loaded
  kFastDirectIbv,      // index buffer view: baked vs the live composition
  kFastDirectArgs,     // draw arguments: baked count and VGT_INDX_OFFSET vs the live reads
  kFastDirectTopo,     // host topology: baked vs the entry's processing result
  kFastDirectDiV,      // vertex descriptor-indices latch: a hit implies the full rule uploads nothing
  kFastDirectDiP,      // pixel descriptor-indices latch, the same
  kFastDirectEnv,      // environment fields: a matching stamp implies every field matches
  kFastDirectClasses
};
inline const char* FastDirectName(uint32_t c) {
  static const char* const kNames[kFastDirectClasses] = {
      "vp",  "sc",  "stencil", "blend", "sys",  "hiz",  "pool", "rt",  "xfer", "tex",
      "ib",  "ovl", "txl",     "ibv",   "args", "topo", "div",  "dip", "env"};
  return c < kFastDirectClasses ? kNames[c] : "?";
}

// ---- Step 3: the baked index buffer view ----
// D3D12_INDEX_BUFFER_VIEW without the D3D12 header: the base address, the
// byte size from the host vertex count and the index format (true: 32-bit).
struct FastIbv {
  uint64_t location = 0;
  uint32_t size_bytes = 0;
  uint32_t index32 = 0;
  constexpr bool operator==(const FastIbv&) const = default;
};
constexpr FastIbv FastBakeIbv(uint64_t location, uint32_t host_vertex_count, bool index32) {
  FastIbv v{};
  v.location = location;
  v.size_bytes = host_vertex_count * (index32 ? 4u : 2u);
  v.index32 = index32 ? 1u : 0u;
  return v;
}

// ---- Step 4: the environment fields behind the stamp ----
// Everything a baked head reads that no register and no key names, recorded
// with the entry; the mode 2 class `env` compares them against the live
// values whenever the stamp matches (a mismatch names a writer the stamp
// missed). FastEnvFirstDiff: the first differing field's index, or
// UINT32_MAX. Field order: path, convert_z, gamma16, fixed16, pso_bind,
// scale_x, scale_y, widen, shm_gpu.
struct FastEnvFields {
  uint32_t path = 0, convert_z = 0, gamma16 = 0, fixed16 = 0, pso_bind = 0;
  uint32_t scale_x = 0, scale_y = 0, reserved = 0;
  uint64_t widen = 0;    // nr::DetileWidenStamp()
  uint64_t shm_gpu = 0;  // the shared memory GPU address (the index view base)
};
constexpr uint32_t FastEnvFirstDiff(const FastEnvFields& a, const FastEnvFields& b) {
  if (a.path != b.path) return 0;
  if (a.convert_z != b.convert_z) return 1;
  if (a.gamma16 != b.gamma16) return 2;
  if (a.fixed16 != b.fixed16) return 3;
  if (a.pso_bind != b.pso_bind) return 4;
  if (a.scale_x != b.scale_x) return 5;
  if (a.scale_y != b.scale_y) return 6;
  if (a.widen != b.widen) return 7;
  if (a.shm_gpu != b.shm_gpu) return 8;
  return 0xFFFFFFFFu;
}

}  // namespace rex::graphics::nr_native
