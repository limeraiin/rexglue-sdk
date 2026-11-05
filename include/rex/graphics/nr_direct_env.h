#pragma once
#include <atomic>
#include <cstdint>

// [NR-NATIVE-R2] N4 step 4: the direct submission environment stamp.
//
// An owned object entry bakes its draw head (host viewport and scissor,
// stencil reference, system constants block, index buffer view) from keyed
// registers PLUS process state that no register names. The stamp is bumped
// by every writer of that state, so an entry baked under an older stamp is
// re-derived instead of replayed. The writers (grep NrDirectEnvBump):
//   render_target_cache.cpp  Initialize: the path, the gamma unorm16 flag and
//                            the depth float24 convert flag are assigned
//   d3d12/command_processor  ClearCaches (every entry stale anyway) and the
//                            per-frame g_nr_native_pso_bind latch on a change
//   nr_detile.cpp            DetileSetEnabled on a change, DetileFrameEnd
//                            when the armed state, pass signature or full
//                            height changes, the resolve guard trip and
//                            DetileSetWidenSegLimit on a change (all inputs
//                            of the baked scissor widen)
// Values with no runtime writer, recorded in the entry and compared by the
// mode 2 class `env` so a missed writer shows as ne: the draw resolution
// scale (texture cache constructor), the fixed16 truncation (the path and a
// cvar), the shared memory GPU address (the index buffer view base).
namespace rex::graphics::nr {

inline std::atomic<uint32_t> g_nr_direct_env{1};

inline uint32_t NrDirectEnvStamp() { return g_nr_direct_env.load(std::memory_order_relaxed); }
inline void NrDirectEnvBump() { g_nr_direct_env.fetch_add(1, std::memory_order_relaxed); }

}  // namespace rex::graphics::nr
