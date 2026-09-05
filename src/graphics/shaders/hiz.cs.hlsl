// [hiz] ISSUEDRAW increment 3: the Hi-Z checkpoint (see command_processor.cpp
// "[hiz]"). Two compute entry points sharing one root signature:
//
//   hiz_build: one 8x8 thread group per 8x8-pixel tile of the bound host
//   depth buffer; every sample of every pixel of the tile is read and the
//   tile's (min, max) depth lands in the Hi-Z buffer. The tile covers every
//   sample it contains, so a rect test against it is conservative.
//
//   hiz_test: one thread per table entry (a cull-eligible draw recorded after
//   the checkpoint): its screen rect in pixels, its nearest host depth and
//   the direction of its depth test. The draw is hidden when every tile of
//   the rect is nearer than the draw's nearest point. The thread writes the
//   draw's indirect arguments (InstanceCount 0 when hidden and the mode is
//   skip) and the verdict word the verify readback reads.
//
// Build (from src/graphics/shaders):
//   fxc /T cs_5_1 /O3 /E hiz_build /D HIZ_SAMPLES=4 /Fh bytecode/d3d12_5_1/hiz_build_4x_cs.h /Vn hiz_build_4x_cs hiz.cs.hlsl
//   fxc /T cs_5_1 /O3 /E hiz_build /D HIZ_SAMPLES=2 /Fh bytecode/d3d12_5_1/hiz_build_2x_cs.h /Vn hiz_build_2x_cs hiz.cs.hlsl
//   fxc /T cs_5_1 /O3 /E hiz_build /D HIZ_SAMPLES=1 /Fh bytecode/d3d12_5_1/hiz_build_1x_cs.h /Vn hiz_build_1x_cs hiz.cs.hlsl
//   fxc /T cs_5_1 /O3 /E hiz_test /Fh bytecode/d3d12_5_1/hiz_test_cs.h /Vn hiz_test_cs hiz.cs.hlsl

#ifndef HIZ_SAMPLES
#define HIZ_SAMPLES 1
#endif

cbuffer HizConstants : register(b0) {
  uint hiz_tiles_x;   // Hi-Z width in tiles
  uint hiz_tiles_y;   // Hi-Z height in tiles
  uint hiz_depth_w;   // depth buffer width in pixels
  uint hiz_depth_h;   // depth buffer height in pixels
  uint hiz_mode;      // 0 Hi-Z invalid (nothing hidden), 1 verify (InstanceCount kept), 2 skip
  uint hiz_entry_cap; // entries the table holds; runs follow them (threads >= cap)
  uint hiz_pad1;
  uint hiz_pad2;
};

#if HIZ_SAMPLES > 1
Texture2DMS<float, HIZ_SAMPLES> hiz_depth : register(t0);
#else
Texture2D<float> hiz_depth : register(t0);
#endif
RWStructuredBuffer<float2> hiz_buffer : register(u0);   // per tile (min, max)
ByteAddressBuffer hiz_table : register(t1);              // the window's entries
RWByteAddressBuffer hiz_args : register(u1);             // indirect arguments, 32 B per slot
RWByteAddressBuffer hiz_verdict : register(u2);          // 4 B per slot, 1 = hidden

groupshared float hiz_gs_min[64];
groupshared float hiz_gs_max[64];

[numthreads(8, 8, 1)]
void hiz_build(uint3 group_id : SV_GroupID, uint3 thread_id : SV_GroupThreadID,
               uint group_index : SV_GroupIndex) {
  uint2 pixel = group_id.xy * 8u + thread_id.xy;
  float mn = 2.0f, mx = -1.0f;
  if (pixel.x < hiz_depth_w && pixel.y < hiz_depth_h) {
#if HIZ_SAMPLES > 1
    [unroll]
    for (uint s = 0; s < HIZ_SAMPLES; ++s) {
      float d = hiz_depth.Load(int2(pixel), int(s));
      mn = min(mn, d);
      mx = max(mx, d);
    }
#else
    float d = hiz_depth.Load(int3(int2(pixel), 0));
    mn = d;
    mx = d;
#endif
  }
  hiz_gs_min[group_index] = mn;
  hiz_gs_max[group_index] = mx;
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for (uint stride = 32u; stride > 0u; stride >>= 1u) {
    if (group_index < stride) {
      hiz_gs_min[group_index] = min(hiz_gs_min[group_index], hiz_gs_min[group_index + stride]);
      hiz_gs_max[group_index] = max(hiz_gs_max[group_index], hiz_gs_max[group_index + stride]);
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if (group_index == 0u && group_id.x < hiz_tiles_x && group_id.y < hiz_tiles_y) {
    hiz_buffer[group_id.y * hiz_tiles_x + group_id.x] = float2(hiz_gs_min[0], hiz_gs_max[0]);
  }
}

// [hiz-sub] The sub-box test (drive 840: the whole-box rule hid 21% of the
// frame's indices where 58% are hidden and cullable; an object running
// away from the camera has its nearest corner far in front of the tiles
// its far end covers). When the whole box is not hidden and the entry
// carries its clip matrix, object-space box and viewport (flag bit 5), the
// box is split at its midpoints into 8 sub-boxes, each projected here (8
// corners through the 4x5 clip matrix, the same viewport transform and
// precision margin as the CPU's whole-box path) and tested by the same
// rule; the draw is hidden when every sub-box is (verdict word 2).
//
// Table layout: dword 0 = entry count, dword 1 = run count; entries at byte
// 16, 208 bytes each; runs at 16 + entry_cap * 208, 16 bytes each: {base slot,
// capacity, count, pad} = a per-instance pool batch whose elements
// [count, capacity) the thread zeroes (the batch executes capacity elements
// with no GPU-side count read).
//   float4 rect (x0, y0, x1, y1) in depth-buffer pixels
//   float  z_near        the draw's nearest host depth over the bounds
//   uint   flags         bit 0: reversed (GEQUAL/GREATER: hidden when
//                        z_near < tile min); else LESS/LEQUAL (hidden when
//                        z_near > tile max); bit 1: never hidden (an
//                        untestable instance of a per-instance pool batch);
//                        bits 8..31: the instance index written as the
//                        slot's root constant (argument dword 0)
//   uint   slot          indirect-argument / verdict slot
//   uint   args4         argument dword 4 (the base vertex of indexed draws)
//   uint4  args0         DrawIndexed/Draw argument dwords 0..3
//   [hiz-sub] dwords 12..50 (flag bit 5): float m[20] (4x5 row-major clip =
//   m x (pos.xyz, pos.w, 1)), float4 box_min (xyz, pos.w), float3 box_max,
//   float3 ndc_scale, float3 ndc_offset, float2 xy_offset, float2 xy_extent,
//   float z_min, float z_max; dword 51 pad.
// Slot layout (32 B): dword 0 = the instance base root constant, dwords
// 1..5 = the draw arguments (the command signature is [constant, draw]).
// Every tile of [tx0, tx1] x [ty0, ty1] nearer than z_near (the whole-box rule).
bool hiz_tiles_hidden(int tx0, int ty0, int tx1, int ty1, float z_near, bool reversed) {
  tx0 = max(tx0, 0);
  ty0 = max(ty0, 0);
  tx1 = min(tx1, int(hiz_tiles_x) - 1);
  ty1 = min(ty1, int(hiz_tiles_y) - 1);
  bool hidden = tx0 <= tx1 && ty0 <= ty1;
  [loop]
  for (int ty = ty0; ty <= ty1 && hidden; ++ty) {
    [loop]
    for (int tx = tx0; tx <= tx1 && hidden; ++tx) {
      float2 t = hiz_buffer[uint(ty) * hiz_tiles_x + uint(tx)];
      if (reversed ? (z_near >= t.x) : (z_near <= t.y)) {
        hidden = false;
      }
    }
  }
  return hidden;
}

// [hiz-sub] all 8 sub-boxes of the entry's box hidden.
bool hiz_sub_hidden(uint base, bool reversed) {
  float m[20];
  [unroll]
  for (uint k = 0u; k < 20u; ++k) {
    m[k] = asfloat(hiz_table.Load(base + 48u + k * 4u));
  }
  float4 bmn = asfloat(hiz_table.Load4(base + 128u));
  float3 bmx = asfloat(hiz_table.Load3(base + 144u));
  float3 ndc_scale = asfloat(hiz_table.Load3(base + 156u));
  float3 ndc_offset = asfloat(hiz_table.Load3(base + 168u));
  float2 xy_off = asfloat(hiz_table.Load2(base + 180u));
  float2 xy_ext = asfloat(hiz_table.Load2(base + 188u));
  float z_min = asfloat(hiz_table.Load(base + 196u));
  float z_max = asfloat(hiz_table.Load(base + 200u));
  float3 mid = 0.5f * (bmn.xyz + bmx);
  bool all_hidden = true;
  [loop]
  for (uint sb = 0u; sb < 8u && all_hidden; ++sb) {
    float3 smn = float3((sb & 1u) ? mid.x : bmn.x, (sb & 2u) ? mid.y : bmn.y,
                        (sb & 4u) ? mid.z : bmn.z);
    float3 smx = float3((sb & 1u) ? bmx.x : mid.x, (sb & 2u) ? bmx.y : mid.y,
                        (sb & 4u) ? bmx.z : mid.z);
    float x0 = 1e30f, y0 = 1e30f, x1 = -1e30f, y1 = -1e30f;
    float zn = reversed ? -1e30f : 1e30f;
    bool ok = true;
    [loop]
    for (uint c = 0u; c < 8u && ok; ++c) {
      float3 p = float3((c & 1u) ? smx.x : smn.x, (c & 2u) ? smx.y : smn.y,
                        (c & 4u) ? smx.z : smn.z);
      float4 clip;
      [unroll]
      for (uint r = 0u; r < 4u; ++r) {
        clip[r] = m[r * 5u] * p.x + m[r * 5u + 1u] * p.y + m[r * 5u + 2u] * p.z +
                  m[r * 5u + 3u] * bmn.w + m[r * 5u + 4u];
      }
      if (!(clip.w > 1e-6f)) {
        ok = false;  // a corner at or behind the camera plane
        continue;
      }
      float iw = 1.0f / clip.w;
      float nx = clip.x * iw * ndc_scale.x + ndc_offset.x;
      float ny = clip.y * iw * ndc_scale.y + ndc_offset.y;
      float nz = clip.z * iw * ndc_scale.z + ndc_offset.z;
      if (isnan(nx) || isnan(ny) || isnan(nz)) {
        ok = false;
        continue;
      }
      nz = saturate(nz);
      float px = xy_off.x + (nx * 0.5f + 0.5f) * xy_ext.x;
      float py = xy_off.y + (0.5f - ny * 0.5f) * xy_ext.y;
      float hz = z_min + nz * (z_max - z_min);
      x0 = min(x0, px);
      x1 = max(x1, px);
      y0 = min(y0, py);
      y1 = max(y1, py);
      zn = reversed ? max(zn, hz) : min(zn, hz);
    }
    if (!ok) {
      all_hidden = false;
      continue;
    }
    x0 = max(x0, xy_off.x) - 1.0f;
    y0 = max(y0, xy_off.y) - 1.0f;
    x1 = min(x1, xy_off.x + xy_ext.x) + 1.0f;
    y1 = min(y1, xy_off.y + xy_ext.y) + 1.0f;
    if (x1 <= x0 || y1 <= y0) {
      continue;  // entirely outside the viewport: nothing of it is drawn
    }
    float eps = abs(zn) * (1.0f / 262144.0f) + (1.0f / 4194304.0f);
    zn = reversed ? zn + eps : zn - eps;
    if (!hiz_tiles_hidden(int(floor(x0 / 8.0f)), int(floor(y0 / 8.0f)),
                          int(ceil(x1 / 8.0f)) - 1, int(ceil(y1 / 8.0f)) - 1, zn, reversed)) {
      all_hidden = false;
    }
  }
  return all_hidden;
}

[numthreads(64, 1, 1)]
void hiz_test(uint3 dispatch_id : SV_DispatchThreadID) {
  uint i = dispatch_id.x;
  uint count = hiz_table.Load(0);
  if (i >= hiz_entry_cap) {
    uint r = i - hiz_entry_cap;
    if (r >= hiz_table.Load(4u)) {
      return;
    }
    uint rb = 16u + hiz_entry_cap * 208u + r * 16u;
    uint run_base = hiz_table.Load(rb);
    uint run_cap = hiz_table.Load(rb + 4u);
    uint run_count = hiz_table.Load(rb + 8u);
    [loop]
    for (uint s = run_count; s < run_cap; ++s) {
      uint sb = (run_base + s) * 32u;
      hiz_args.Store4(sb, uint4(0u, 0u, 0u, 0u));
      hiz_args.Store2(sb + 16u, uint2(0u, 0u));
    }
    return;
  }
  if (i >= count) {
    return;
  }
  uint base = 16u + i * 208u;
  float4 rect = asfloat(hiz_table.Load4(base));
  float z_near = asfloat(hiz_table.Load(base + 16u));
  uint flags = hiz_table.Load(base + 20u);
  uint slot = hiz_table.Load(base + 24u);
  uint args4 = hiz_table.Load(base + 28u);
  uint4 args0 = hiz_table.Load4(base + 32u);

  int tx0 = int(floor(rect.x / 8.0f));
  int ty0 = int(floor(rect.y / 8.0f));
  int tx1 = int(ceil(rect.z / 8.0f)) - 1;
  int ty1 = int(ceil(rect.w / 8.0f)) - 1;
  tx0 = max(tx0, 0);
  ty0 = max(ty0, 0);
  tx1 = min(tx1, int(hiz_tiles_x) - 1);
  ty1 = min(ty1, int(hiz_tiles_y) - 1);
  bool testable = hiz_mode != 0u && (flags & 2u) == 0u;
  bool reversed = (flags & 1u) != 0u;
  uint verdict = 0u;
  bool hidden = testable && hiz_tiles_hidden(tx0, ty0, tx1, ty1, z_near, reversed);
  if (hidden) {
    verdict = 1u;
  } else if (testable && (flags & 32u) != 0u && hiz_sub_hidden(base, reversed)) {
    hidden = true;
    verdict = 2u;  // [hiz-sub]
  }
  uint instances = (hidden && hiz_mode == 2u) ? 0u : args0.y;
  uint ab = slot * 32u;
  if ((flags & 16u) != 0u) {
    // [hiz-sig] the draw-only layout: the draw arguments at dword 0, no root
    // constant (the plain-draw site's draw-only command signature).
    hiz_args.Store4(ab, uint4(args0.x, instances, args0.z, args0.w));
    hiz_args.Store2(ab + 16u, uint2(args4, 0u));
  } else {
    hiz_args.Store(ab, flags >> 8u);  // the instance base root constant
    hiz_args.Store4(ab + 4u, uint4(args0.x, instances, args0.z, args0.w));
    hiz_args.Store(ab + 20u, args4);
  }
  hiz_verdict.Store(slot * 4u, verdict);
}
