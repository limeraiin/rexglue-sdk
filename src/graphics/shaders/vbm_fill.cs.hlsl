// [ia] The host-order vertex/index buffer mirror fill (see
// d3d12/vb_mirror.cpp). Copies `vbm_dword_count` dwords from the shared
// memory buffer (guest byte order, as the recompiled game wrote them) into
// the mirror arena, byte-swapped per the fetch constant's endian mode, so the
// input assembler can read the vertex and index data directly. One thread
// per 4 dwords.
//
// Build (from src/graphics/shaders):
//   fxc /T cs_5_1 /O3 /E vbm_fill /Fh bytecode/d3d12_5_1/vbm_fill_cs.h /Vn vbm_fill_cs vbm_fill.cs.hlsl

cbuffer VbmConstants : register(b0) {
  uint vbm_src_dword;    // first dword in the shared memory buffer
  uint vbm_dst_dword;    // first dword in the mirror chunk
  uint vbm_dword_count;  // dwords to copy
  uint vbm_endian;       // xenos::Endian: 0 none, 1 8in16, 2 8in32, 3 16in32
};

ByteAddressBuffer vbm_src : register(t0);
RWByteAddressBuffer vbm_dst : register(u0);

uint4 vbm_swap(uint4 v) {
  if (vbm_endian == 1u || vbm_endian == 2u) {
    // 8-in-16, or one half of 8-in-32.
    v = ((v & 0x00FF00FFu) << 8u) | ((v >> 8u) & 0x00FF00FFu);
  }
  if (vbm_endian == 2u || vbm_endian == 3u) {
    // 16-in-32, or the other half of 8-in-32.
    v = (v << 16u) | (v >> 16u);
  }
  return v;
}

[numthreads(64, 1, 1)]
void vbm_fill(uint3 thread_id : SV_DispatchThreadID) {
  uint i = thread_id.x * 4u;
  if (i >= vbm_dword_count) {
    return;
  }
  uint src = (vbm_src_dword + i) * 4u;
  uint dst = (vbm_dst_dword + i) * 4u;
  if (i + 4u <= vbm_dword_count) {
    vbm_dst.Store4(dst, vbm_swap(vbm_src.Load4(src)));
    return;
  }
  // The tail: fewer than 4 dwords left.
  for (uint k = 0u; i + k < vbm_dword_count; ++k) {
    vbm_dst.Store(dst + k * 4u, vbm_swap(uint4(vbm_src.Load(src + k * 4u), 0u, 0u, 0u)).x);
  }
}
