/**
 ******************************************************************************
 * ReXGlue                                                                    *
 ******************************************************************************
 * Copyright 2025 the ReXGlue authors. All rights reserved.                   *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef REX_GRAPHICS_NR_SHADER_DB_H_
#define REX_GRAPHICS_NR_SHADER_DB_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// [NR-SDB] The shader database reader: native-renderer build-out increment
// 4b-2.
//
// Increment 4b-1 closed the input model (the depth-1 IB stream plus the bin
// state carries the entire recovery context). What a draw still needs before
// it can be issued natively is its SHADERS, and increment 4a's city census
// said how they arrive: 868-1553 distinct shaders per window against a corpus
// of 3,320, with the distinct-set counter OVERFLOWING (>=2,634-3,165/s). The
// city touches essentially the whole corpus every second, so a live LRU
// translation cache would thrash by construction. The corpus is therefore
// translated AHEAD OF TIME, which the offline census already proved safe:
// all 3,320 unique ucode blobs pass AnalyzeUcode + DxbcShaderTranslator, 0
// fail, 0 crash.
//
// This unit is the reader that makes that possible: it walks an XDK shader
// database (xeshader.sdb) for shader containers, keys each one by the SAME
// hash the runtime uses for a shader loaded from guest memory (XXH3-64 over
// the raw big-endian ucode -- see PipelineCache::LoadShader), and parses each
// container's constant table for its symbol names.
//
// Container layout (XenosRecomp's shader.h, all fields big-endian):
//
//   +0x00  u32  flags       0x102A1100 = PIXEL, 0x102A1101 = VERTEX
//   +0x04  u32  virtualSize   metadata region size (header, ctab, Shader)
//   +0x08  u32  physicalSize  payload region size (the ucode itself)
//   +0x0C  u32  fieldC
//   +0x10  u32  constantTableOffset
//   +0x14  u32  definitionTableOffset   (0 in 766 of Naruto's 4,467)
//   +0x18  u32  shaderOffset  -> {physicalOffset, size} of the ucode
//   +0x1C  u32  field1C     always 0 (scan discriminator)
//   +0x20  u32  field20     always 0 (scan discriminator)
//
// The ucode lives at container + virtualSize + physicalOffset and is stored
// big-endian, exactly as the guest uploads it -- so hashing those bytes
// verbatim yields the runtime's key with no swapping. (rex::graphics::Shader
// byteswaps the ucode in its own constructor; pre-swapping analyzes garbage
// that still "passes" ~96% of the time. That trap cost the first census run.)
//
// The constant table at +constantTableOffset is NOT a bare D3DX blob and NOT
// a wholly custom format either, which the earlier note in the handoff got
// half right: it is the standard D3DXSHADER_CONSTANTTABLE record with the
// 'CTAB' FourCC replaced by a u32 total-size prefix. So:
//
//   ctab_base = container + constantTableOffset + 4      (past the prefix)
//   +0x00 u32 size (0x1C)   +0x04 u32 creator   +0x08 u32 version token
//   +0x0C u32 constants     +0x10 u32 constantInfo       +0x14 u32 flags
//   +0x18 u32 target
//
// with every offset relative to ctab_base, and constantInfo pointing at
// `constants` D3DXSHADER_CONSTANTINFO records of 20 bytes:
//
//   +0x00 u32 name   +0x04 u16 registerSet   +0x06 u16 registerIndex
//   +0x08 u16 registerCount  +0x0A u16 reserved
//   +0x0C u32 typeInfo       +0x10 u32 defaultValue
//
// The version token (0xFFFF0300 = ps_3_0 / 0xFFFE0300 = vs_3_0) is an
// INDEPENDENT witness of the stage that the flags' low bit also encodes, so
// the scan cross-checks the two and counts disagreements rather than trusting
// either alone.
//
// No SDK dependencies beyond XXH3 (header-only, in the .cpp), no globals, no
// threads: tools/nr-shader-db-test.cpp builds this file bare. Pointers in the
// returned structs alias the caller's file buffer and are valid only as long
// as it is.

namespace rex {
namespace graphics {
namespace nr {

// D3DXREGISTER_SET, as stored in a constant record.
enum class ShaderRegisterSet : uint16_t {
  kBool = 0,
  kInt4 = 1,
  kFloat4 = 2,
  kSampler = 3,
};

// One named constant or sampler. `name` points into the caller's buffer.
struct ShaderSymbol {
  const char* name = nullptr;
  uint16_t register_set = 0;
  uint16_t register_index = 0;
  uint16_t register_count = 0;
};

// One compiled-shader container found in the database.
struct ShaderContainer {
  size_t file_offset = 0;  // container start, for drilling with the census
  bool pixel = false;      // from flags bit 0 (0 = PIXEL; the 00 suffix)

  // Raw big-endian ucode. NOT guaranteed 4-byte aligned: containers sit at
  // arbitrary offsets inside the database's own records, so copy before
  // handing it to anything that wants dwords.
  const uint8_t* ucode = nullptr;
  uint32_t ucode_bytes = 0;
  // XXH3-64 over those bytes == the key PipelineCache::LoadShader computes
  // for the same shader loaded from guest memory.
  uint64_t ucode_hash = 0;

  // Constant table, or nullptr when the container has none in range.
  const uint8_t* ctab = nullptr;  // past the size prefix
  uint32_t ctab_bytes = 0;        // the prefix's declared size
  uint32_t constant_count = 0;
  uint32_t version_token = 0;    // 0xFFFF0300 ps_3_0 / 0xFFFE0300 vs_3_0
  const char* target = nullptr;  // "ps_3_0" / "vs_3_0"
  const char* creator = nullptr;  // XDK compiler version string
};

// Scan counters. Every rejection has its own counter: a silent skip and a
// genuine absence must never look alike.
struct ShaderDbStats {
  uint32_t containers = 0;
  uint32_t pixel = 0;
  uint32_t vertex = 0;
  uint32_t bad_ucode_bounds = 0;  // magic matched, ucode span did not
  uint32_t ctab_absent = 0;       // offset 0 or outside the metadata region
  uint32_t ctab_malformed = 0;    // present but the record does not parse
  uint32_t ctab_empty = 0;        // parses, declares zero constants
  uint32_t ctab_ends_at_shader = 0;  // prefix size lands exactly on Shader
  uint32_t stage_token_mismatch = 0;  // flags bit 0 disagrees with the token
};

// Walks `data` for containers, appending each to `out`. Advances ONE BYTE on
// a miss (a stride-4 scan finds nothing: the containers are not aligned in
// the .sdb) and by the whole span on a hit. `stats` may be null.
void ScanShaderDatabase(const uint8_t* data, size_t size,
                        std::vector<ShaderContainer>* out,
                        ShaderDbStats* stats);

// Parses `container`'s constant records into `out` (cleared first). Returns
// false when the container has no constant table or the records do not fit;
// a container with a well-formed table and zero constants returns true with
// an empty vector.
bool ParseShaderSymbols(const ShaderContainer& container,
                        std::vector<ShaderSymbol>* out);

// XXH3-64 over raw big-endian ucode: the runtime's shader key.
uint64_t HashUcode(const uint8_t* ucode, uint32_t bytes);

// "bool" / "int4" / "float4" / "sampler" / "set<n>".
const char* RegisterSetName(uint16_t register_set);

// A database loaded and indexed by the runtime's shader key, so a live
// shader load can ask "is this blob in the corpus?" in constant time.
//
// The question that has to be answered BEFORE anything is built on this key:
// the census keys containers by XXH3-64 over the container's own ucode span,
// while the runtime keys a live shader by XXH3-64 over `size_dwords` dwords
// of guest memory, where size_dwords comes from the IM_LOAD packet. Those are
// the same bytes only if the guest declares the same length the container
// stores. If it ever declares a different one, the key differs even though
// the shader is identical -- an AOT cache keyed by the container hash would
// silently miss. So the index also supports a PREFIX query: on a key miss,
// it looks for a container whose ucode agrees with the queried blob over
// their common length, which distinguishes a length disagreement (the same
// shader under another key) from a blob the database simply does not hold.
//
// Owns its file bytes: unlike the raw ShaderContainer pointers above, an
// index stays valid for its own lifetime.
class ShaderDbIndex {
 public:
  struct Entry {
    uint64_t hash = 0;
    uint32_t ucode_bytes = 0;
    uint32_t containers = 0;    // duplicates of this blob in the database
    bool pixel = false;
    size_t first_container = 0;  // index into containers()
  };

  // Result of a prefix query. `equal_bytes` is the length actually compared.
  struct PrefixMatch {
    bool found = false;
    uint64_t hash = 0;
    uint32_t db_bytes = 0;
    uint32_t equal_bytes = 0;
    bool pixel = false;
  };

  // Reads and indexes `path`. Returns false if the file cannot be read; a
  // readable file holding no containers loads successfully with zero entries
  // (an empty corpus and an unreadable one must not look alike).
  bool LoadFile(const char* path);
  // Same, over bytes already in hand (what the unit test uses).
  bool LoadBytes(std::vector<uint8_t> data);

  bool loaded() const { return loaded_; }
  const ShaderDbStats& stats() const { return stats_; }
  const std::vector<ShaderContainer>& containers() const { return containers_; }
  size_t unique_count() const { return by_hash_.size(); }
  size_t file_bytes() const { return data_.size(); }

  // Null when the key is absent.
  const Entry* Lookup(uint64_t hash) const;

  // Only meaningful for a blob whose key is absent. Matches on the leading
  // bytes, then compares the whole common length; a container that agrees
  // over its own full length is preferred over a longer one.
  PrefixMatch FindByPrefix(const uint8_t* ucode, uint32_t bytes) const;

 private:
  bool Index();

  std::vector<uint8_t> data_;
  std::vector<ShaderContainer> containers_;
  ShaderDbStats stats_;
  std::unordered_map<uint64_t, Entry> by_hash_;
  // Leading-bytes hash -> indices into an entry list, for the prefix query.
  std::unordered_multimap<uint64_t, uint64_t> by_prefix_;  // -> entry hash
  bool loaded_ = false;
};

}  // namespace nr
}  // namespace graphics
}  // namespace rex

#endif  // REX_GRAPHICS_NR_SHADER_DB_H_
