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

// Table layout: dword 0 = entry count, dword 1 = run count; entries at byte
// 16, 48 bytes each; runs at 16 + entry_cap * 48, 16 bytes each: {base slot,
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
// Slot layout (32 B): dword 0 = the instance base root constant, dwords
// 1..5 = the draw arguments (the command signature is [constant, draw]).
[numthreads(64, 1, 1)]
void hiz_test(uint3 dispatch_id : SV_DispatchThreadID) {
  uint i = dispatch_id.x;
  uint count = hiz_table.Load(0);
  if (i >= hiz_entry_cap) {
    uint r = i - hiz_entry_cap;
    if (r >= hiz_table.Load(4u)) {
      return;
    }
    uint rb = 16u + hiz_entry_cap * 48u + r * 16u;
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
  uint base = 16u + i * 48u;
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
  bool hidden = hiz_mode != 0u && (flags & 2u) == 0u && tx0 <= tx1 && ty0 <= ty1;
  if (hidden) {
    bool reversed = (flags & 1u) != 0u;
    [loop]
    for (int ty = ty0; ty <= ty1 && hidden; ++ty) {
      [loop]
      for (int tx = tx0; tx <= tx1; ++tx) {
        float2 t = hiz_buffer[uint(ty) * hiz_tiles_x + uint(tx)];
        if (reversed ? (z_near >= t.x) : (z_near <= t.y)) {
          hidden = false;
          break;
        }
      }
    }
  }
  uint instances = (hidden && hiz_mode == 2u) ? 0u : args0.y;
  uint ab = slot * 32u;
  hiz_args.Store(ab, flags >> 8u);  // the instance base root constant
  hiz_args.Store4(ab + 4u, uint4(args0.x, instances, args0.z, args0.w));
  hiz_args.Store(ab + 20u, args4);
  hiz_verdict.Store(slot * 4u, hidden ? 1u : 0u);
}
