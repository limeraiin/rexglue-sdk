/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <chrono>

#include <rex/audio/xma/context.h>
#include <rex/audio/xma/decoder.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/logging.h>
#include <rex/perf/counter.h>
#include <rex/math.h>
#include <rex/memory/ring_buffer.h>
#include <rex/string/buffer.h>
#include <rex/system/function_dispatcher.h>
#include <rex/system/thread_state.h>
#include <rex/system/xthread.h>

extern "C" {
#include "libavutil/log.h"
}  // extern "C"

REXCVAR_DEFINE_BOOL(ffmpeg_verbose, false, "Audio", "Verbose FFmpeg output (debug and above)");

// [NARUTO-XMA-PROBE]
REXCVAR_DEFINE_BOOL(apu_xma_probe, false, "Audio",
                    "Log XMA streaming diagnostics ([nrxma]): context lifecycle, buffer refills, "
                    "starvation, loop events, 1Hz per-context status");
REXCVAR_DEFINE_BOOL(apu_xma_locked_ops, false, "Audio",
                    "Serialize guest XMASet* context writes against the XMA decode worker "
                    "(candidate fix for long streamed sounds cutting out)");
// [NARUTO-XMA-STARVE] Root-cause fix for "long minigame music cuts out"
// (measured 2026-07-13, naruto_221 logs): at the END of a streamed track the
// final buffer's last frame claims to span into a next buffer that will never
// arrive; the post-2021-rewrite decoder holds the buffer valid + error_status=4
// forever, while the game waits for full consumption before looping the track
// => permanent deadlock, music dead. Pre-rewrite Xenia (last-good canary
// 168db250d) invalidated the exhausted buffer instead, so the game looped.
// With this ON, a context stuck >512 consecutive kicks (healthy refill storms
// measured <=121) at the SAME spanning-frame state swaps the buffer out and
// clears the error, unblocking the game's loop restart.
REXCVAR_DEFINE_BOOL(apu_xma_starved_swap, true, "Audio",
                    "Break XMA end-of-stream deadlock: a context stuck waiting for a next input "
                    "buffer that never arrives consumes its exhausted buffer so the game can "
                    "restart/loop the stream (fixes long streamed sounds cutting out)");
// [NARUTO-XMA-CUMUL] On real hardware the context's input_buffer_read_offset
// is CUMULATIVE across the whole submitted stream (it never resets per input
// buffer — 26 bits cover an 8MB stream). Games/XMAPlayback arm loop_start/
// loop_end as stream-global bit offsets (measured: Naruto music armed
// loop_end=18172890 ≈ the whole 2.2MB track on a context with 16384-bit
// buffers), so the loop wrap can only ever fire against a cumulative offset.
// Xenia's 2021 XMA rewrite made the offset per-buffer, so streamed loops
// never wrap => looping music dies at first track end. This restores the
// hardware semantics: guest-visible offset = consumed-buffer base + intra-
// buffer offset, and the loop wraps (buffer retire + rebase to loop_start)
// when the cumulative offset reaches loop_end.
REXCVAR_DEFINE_BOOL(apu_xma_cumulative_rdoff, true, "Audio",
                    "Hardware-correct cumulative XMA input read offset + streamed loop wrap "
                    "(fixes looping streamed music stopping at the first loop point)");
// [NARUTO-XMA-SPLIT] When the last frame of an input buffer spans into a not-
// yet-submitted next buffer, real hardware (and pre-rewrite Xenia, the last
// version where looping streamed music worked in Naruto) treats the exhausted
// buffer as CONSUMED immediately — it caches the partial frame internally,
// releases the buffer to the game, and completes the frame when the next
// buffer arrives. The post-2021-rewrite decoder instead holds the buffer
// valid with error_status=4 until the next buffer shows up, which stalls the
// game's packet-completion chain at a stream's final (loop-end-truncated)
// packet => the feeder never wraps => looping music dies.
REXCVAR_DEFINE_BOOL(apu_xma_split_frame_cache, true, "Audio",
                    "Release an exhausted XMA input buffer immediately and cache the partial "
                    "split frame until the next buffer arrives (hardware behavior; fixes "
                    "looping streamed music dying at the loop seam)");
// [NARUTO-XMA-SILENCE] Root-cause closure for "looping streamed music dies at
// the loop seam" (measured 2026-07-13, naruto_226 + guest XAudio feeder
// disassembly): the guest produces new XMA packets only while its RENDERED-
// SAMPLES watermark stays within 2 packets of production. The final packet of
// a looping stream is truncated at the loop point; its tail frames reference
// data past the cut, so our decoder can't produce their samples, the guest's
// watermark parks one packet short, and its producer never emits the
// loop-wrap packet (real hardware decodes those tail frames against stale
// buffer bytes and still RENDERS them). Emitting silent frames while a
// split-frame-starved stream stays kicked advances the watermark and
// unblocks the guest's loop wrap.
REXCVAR_DEFINE_BOOL(apu_xma_starved_silence, true, "Audio",
                    "Emit silent output frames when an XMA stream starves at a split frame so "
                    "the game's rendered-samples watermark advances and its stream feeder can "
                    "wrap the loop (fixes looping streamed music dying at the loop seam)");
// [NARUTO-XMA-LOOPEND] The game arms the XMA context's loop_end field with the
// loop point in OUTPUT PCM BYTES (measured 2026-07-14: ctx loop_end=18172890
// exactly equals the cumulative decoded output at the seam, 8873 frames x
// 512 x 2ch x 2B = 18171904, a 0.005% match). Counting decoded output bytes
// and predicting the seam when they reach loop_end lets the decoder emit the
// gap-filling silence at exactly the right instant on the FIRST-EVER encounter
// of any track — no learning pass, no per-track table — because it reads the
// game's own per-track loop point, exactly as real hardware did. Off => fall
// back to the (confirmed-working) learned/persisted input-domain prediction.
REXCVAR_DEFINE_BOOL(apu_xma_loopend_predict, true, "Audio",
                    "Predict the XMA loop seam from the game's loop_end field (output-byte "
                    "domain) for gapless looping on the first-ever encounter of a track");
// [NARUTO-XMA-SILENCE] The game's producer is watermark-gated: at a loop seam
// AND at mid-track streaming stalls it stops feeding until the rendered-sample
// watermark advances, but the watermark can't advance without input — a
// deadlock our silence breaks. Feeding resumes within ~11ms of the first
// silent frame (measured), so this threshold directly sets the audible gap
// length. Must stay above the longest NORMAL between-packet starvation (~34-74
// polls) to avoid injecting silence during healthy streaming; lower = shorter
// mid-track gaps. Tune live via --apu_xma_starve_kicks.
// ⚠ 192 is the PROVEN-SAFE value (working loops). Lowering it to shorten
// mid-track stall gaps is a DEAD END: 48 made all music "super stuttery"
// (false-fires silence into healthy streaming). Do not lower below 192.
REXCVAR_DEFINE_INT32(apu_xma_starve_kicks, 192, "Audio",
                     "XMA starvation polls before gap-filling silence breaks the producer's "
                     "watermark deadlock. 192 = proven safe; lower stutters healthy streaming.");
// [NARUTO-XMA-PACE] Root-cause fix for the ~7.7s-periodic mid-track music gap
// (measured 2026-07-15, runs 238/239): our decoder extracts packets ~10x
// faster than real XMA hardware, so the game's per-tick puller sucks whole
// packets into its XAudio submit queue as a LUMP. The game's rendered-samples
// watermark [obj+440] only advances as submitted audio COMPLETES, and its
// producer gate releases new packets keyed on that watermark; around the
// authored high-frame-count quiet packets (25-39 frames, one per musical
// phrase ~7.7s) the pull frontier runs ~400ms ahead of the watermark, the
// pipeline (2 packets deep) can't cover the phase error, the puller drains the
// ring, the watermark freezes, and the [NARUTO-XMA-SILENCE] backstop has to
// drip it free = the audible 250-450ms gap. Real hardware decodes in real
// time, so the lump can never form. This paces decode output to the stream's
// sample rate (+small headroom) for STREAMED contexts only (single-packet 2KB
// input buffers); in-memory SFX decode bursts are untouched.
REXCVAR_DEFINE_BOOL(apu_xma_realtime_pace, false, "Audio",
                    "Pace streamed XMA decode to the stream's real-time sample rate like real "
                    "hardware (fixes periodic mid-track music gaps from watermark phase error)");

// [NARUTO-XMA-RINGTRACE] Diagnostic (default OFF): emit a high-rate (~50Hz per
// context) trajectory of the guest output-ring occupancy (used/free blocks +
// read/write offsets) for streamed single-packet contexts. Purpose: measure
// whether the decoded-PCM stockpile lives IN the guest ring (occupancy stays
// high) or DOWNSTREAM of it (occupancy hugs empty => input-limited), and whether
// the pull is bursty (sawtooth) or smooth. Disambiguates the mid-track-gap fix
// (pacing vs elastic decode-ahead FIFO); see XMA-AUDIO-HANDOFF.md session 3.
REXCVAR_DEFINE_BOOL(apu_xma_ring_trace, false, "Audio",
                    "Trace streamed XMA output-ring occupancy over time (mid-track music-gap "
                    "stockpile/jitter forensics; [nrxma-ring])");

// [NARUTO-XMA-FIFO] The mid-track music-gap fix (measured 2026-07-15 run 243:
// the guest output ring is ~44ms and hugs empty; the authored producer freeze is
// 267-409ms, so the ~2-packet input pipeline starves and our silence backstop
// drips the audible gap). This inserts a host-side elastic buffer between the
// decoder and the tiny guest ring: decode fed packets EAGERLY into a large FIFO
// (freeing the 2 input slots fast so the guest's watermark-gated producer feeds
// its full ~6-packet lookahead), then meter FIFO->guest-ring at the guest pull
// rate. During the authored freeze the FIFO keeps the ring fed with REAL audio,
// so the render watermark advances and the producer gate reopens without a
// STARVED-SILENCE gap. Streamed single-packet contexts only; genuine
// FIFO-empty+input-dry starvation still falls through to the silence/seam path.
REXCVAR_DEFINE_BOOL(apu_xma_elastic_fifo, false, "Audio",
                    "Host-side elastic decode-ahead FIFO for streamed XMA (fixes the periodic "
                    "mid-track music gap by buffering the producer's full lookahead as PCM)");

// [NARUTO-XMA-STOCK] THE mid-track music-gap root cause (run-239 anatomy, 2026-07-16): the
// ~7.7s-periodic "producer freeze" is BY DESIGN - the feeder's +2-ENTRY lookahead gate,
// crossed with the authored 20-39-frame packets, pre-charges the game's downstream rendered
// stock with ~500ms of early-slurped PCM (measured: 25,664 samples delivered ahead vs 25,650
// drained during the freeze - a 0.05% match), which then plays out while the feeder idles.
// Real playback is continuously covered; mark advances from the stock the whole time. Our
// starve backstop (192-kick timeout) cannot see that stock, misreads the idle as starvation,
// and injects silent frames that the game slurps BEHIND the real stock and plays ~300ms
// later: the audible gap IS our own injected silence. Fix: estimate the guest's stock from
// the SDK side (cumulative real PCM delivered to the ring minus wall-clock x sample rate)
// and suppress the timeout-based silence drip while the stock still covers playback. When
// the estimate drains to the floor, fall through to today's exact behavior - true
// starvation (end-of-stream/loop park, where mark genuinely stops) still gets silence, just
// after the stock argument expires; the loop_end/learned SEAM paths are untouched.
REXCVAR_DEFINE_BOOL(apu_xma_stock_starve, false, "Audio",
                    "Suppress starved-silence injection while the guest's downstream stock "
                    "still covers playback (fixes the periodic mid-track music gap)");
REXCVAR_DEFINE_INT32(apu_xma_stock_floor_ms, 16, "Audio",
                     "Estimated guest-stock floor (ms) below which starved-silence injection "
                     "resumes (safety margin for estimate drift)");

// As with normal Microsoft, there are like twelve different ways to access
// the audio APIs. Early games use XMA*() methods almost exclusively to touch
// decoders. Later games use XAudio*() and direct memory writes to the XMA
// structures (as opposed to the XMA* calls), meaning that we have to support
// both.
//
// The XMA*() functions just manipulate the audio system in the guest context
// and let the normal XmaDecoder handling take it, to prevent duplicate
// implementations. They can be found in xboxkrnl_audio_xma.cc
//
// XMA details:
// https://devel.nuclex.org/external/svn/directx/trunk/include/xma2defs.h
// https://github.com/gdawg/fsbext/blob/master/src/xma_header.h
//
// XAudio2 uses XMA under the covers, and seems to map with the same
// restrictions of frame/subframe/etc:
// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.xaudio2.xaudio2_buffer(v=vs.85).aspx
//
// XMA contexts are 64b in size and tight bitfields. They are in physical
// memory not usually available to games. Games will use MmMapIoSpace to get
// the 64b pointer in user memory so they can party on it. If the game doesn't
// do this, it's likely they are either passing the context to XAudio or
// using the XMA* functions.

namespace rex::audio {

XmaDecoder::XmaDecoder(runtime::FunctionDispatcher* function_dispatcher)
    : memory_(function_dispatcher->memory()), function_dispatcher_(function_dispatcher) {}

XmaDecoder::~XmaDecoder() = default;

void av_log_callback(void* avcl, int level, const char* fmt, va_list va) {
  if (!REXCVAR_GET(ffmpeg_verbose) && level > AV_LOG_WARNING) {
    return;
  }

  string::StringBuffer buff;
  buff.AppendVarargs(fmt, va);
  auto msg = buff.to_string_view();

  switch (level) {
    case AV_LOG_ERROR:
      REXAPU_ERROR("ffmpeg: {}", msg);
      break;
    case AV_LOG_WARNING:
      REXAPU_WARN("ffmpeg: {}", msg);
      break;
    case AV_LOG_INFO:
      REXAPU_INFO("ffmpeg: {}", msg);
      break;
    case AV_LOG_VERBOSE:
    case AV_LOG_DEBUG:
    default:
      REXAPU_DEBUG("ffmpeg: {}", msg);
      break;
  }
}

X_STATUS XmaDecoder::Setup(system::KernelState* kernel_state) {
  // Setup ffmpeg logging callback
  av_log_set_callback(av_log_callback);

  // Register APU/XMA MMIO handlers
  // XMA registers are at 0x7FEA0000-0x7FEAFFFF
  memory()->AddVirtualMappedRange(
      0x7FEA0000,  // base address
      0xFFFF0000,  // mask
      0x0000FFFF,  // size (64KB)
      this,        // context (XmaDecoder*)
      reinterpret_cast<runtime::MMIOReadCallback>(MMIOReadRegisterThunk),
      reinterpret_cast<runtime::MMIOWriteCallback>(MMIOWriteRegisterThunk));
  REXAPU_DEBUG("XMA: Registered MMIO handlers at 0x7FEA0000-0x7FEAFFFF");

  // Setup XMA context data.
  // The Xbox 360 kernel allocates the contexts with X_PAGE_NOCACHE |
  // X_PAGE_READWRITE and writes MmGetPhysicalAddress for the address to the
  // register.
  context_data_first_ptr_ = memory()->SystemHeapAlloc(sizeof(XMA_CONTEXT_DATA) * kContextCount, 256,
                                                      memory::kSystemHeapPhysical);
  context_data_last_ptr_ = context_data_first_ptr_ + (sizeof(XMA_CONTEXT_DATA) * kContextCount - 1);
  register_file_[XmaRegister::ContextArrayAddress] =
      memory()->GetPhysicalAddress(context_data_first_ptr_);

  // Setup XMA contexts.
  for (size_t i = 0; i < kContextCount; ++i) {
    uint32_t guest_ptr = context_data_first_ptr_ + i * sizeof(XMA_CONTEXT_DATA);
    XmaContext& context = contexts_[i];
    if (context.Setup(i, memory(), guest_ptr)) {
      assert_always();
    }
  }
  register_file_[XmaRegister::NextContextIndex] = 1;
  context_bitmap_.Resize(kContextCount);

  worker_running_ = true;
  work_event_ = rex::thread::Event::CreateAutoResetEvent(false);
  assert_not_null(work_event_);
  worker_thread_ = system::object_ref<system::XHostThread>(
      new system::XHostThread(kernel_state, 128 * 1024, 0, [this]() {
        WorkerThreadMain();
        return 0;
      }));
  worker_thread_->set_name("XMA Decoder");

  worker_thread_->Create();

  return X_STATUS_SUCCESS;
}

void XmaDecoder::WorkerThreadMain() {
  while (worker_running_) {
    // Okay, let's loop through XMA contexts to find ones we need to decode!
    bool did_work = false;
    for (uint32_t n = 0; n < kContextCount && worker_running_; n++) {
      XmaContext& context = contexts_[n];
      bool worked = context.Work();
      if (worked) {
        context.SignalWorkDone();
        PROFILE_XMA_FRAME_DECODED();
      }
      did_work = did_work || worked;
    }

    if (paused_) {
      pause_fence_.Signal();
      resume_fence_.Wait();
    }

    if (did_work) {
      continue;
    }
    // No work done this iteration, block until signaled.
    rex::thread::Wait(work_event_.get(), false);
  }
}

void XmaDecoder::Shutdown() {
  if (!worker_thread_) {
    return;
  }

  worker_running_ = false;

  if (work_event_) {
    work_event_->Set();
  }

  if (paused_) {
    Resume();
  }

  // Wait up to 2 seconds for worker thread to exit gracefully.
  auto result = rex::thread::Wait(worker_thread_->thread(), false, std::chrono::milliseconds(2000));
  if (result == rex::thread::WaitResult::kTimeout) {
    REXAPU_WARN("XMA: Worker thread did not exit within 2s, abandoning");
  }
  worker_thread_.reset();

  if (context_data_first_ptr_) {
    memory()->SystemHeapFree(context_data_first_ptr_);
  }

  context_data_first_ptr_ = 0;
  context_data_last_ptr_ = 0;
}

int XmaDecoder::GetContextId(uint32_t guest_ptr) {
  static_assert_size(XMA_CONTEXT_DATA, 64);
  if (guest_ptr < context_data_first_ptr_ || guest_ptr > context_data_last_ptr_) {
    return -1;
  }
  assert_zero(guest_ptr & 0x3F);
  return (guest_ptr - context_data_first_ptr_) >> 6;
}

uint32_t XmaDecoder::AllocateContext() {
  size_t index = context_bitmap_.Acquire();
  if (index == -1) {
    // Out of contexts. [NARUTO-XMA-PROBE] Always log: new sounds silently fail
    // to play from this point on — a prime "sound cuts out over time" suspect.
    REXAPU_WARN("[nrxma] XMA context pool EXHAUSTED ({} in use) - new sound will NOT play",
                kContextCount);
    return 0;
  }

  XmaContext& context = contexts_[index];
  assert_false(context.is_allocated());
  context.set_is_allocated(true);
  return context.guest_ptr();
}

void XmaDecoder::ReleaseContext(uint32_t guest_ptr) {
  auto context_id = GetContextId(guest_ptr);
  assert_true(context_id >= 0);

  XmaContext& context = contexts_[context_id];
  assert_true(context.is_allocated());
  context.Release();
  context_bitmap_.Release(context_id);
}

bool XmaDecoder::BlockOnContext(uint32_t guest_ptr, bool poll) {
  auto context_id = GetContextId(guest_ptr);
  assert_true(context_id >= 0);

  XmaContext& context = contexts_[context_id];
  return context.Block(poll);
}

uint32_t XmaDecoder::ReadRegister(uint32_t addr) {
  auto r = (addr & 0xFFFF) / 4;

  assert_true(r < XmaRegisterFile::kRegisterCount);

  switch (r) {
    case XmaRegister::ContextArrayAddress:
      break;
    case XmaRegister::CurrentContextIndex: {
      // 0606h (1818h) is rotating context processing # set to hardware ID of
      // context being processed.
      // If bit 200h is set, the locking code will possibly collide on hardware
      // IDs and error out, so we should never set it (I think?).
      uint32_t& current_context_index = register_file_[XmaRegister::CurrentContextIndex];
      uint32_t& next_context_index = register_file_[XmaRegister::NextContextIndex];
      // To prevent games from seeing a stuck XMA context, return a rotating
      // number.
      current_context_index = next_context_index;
      next_context_index = (next_context_index + 1) % kContextCount;
      break;
    }
    default:
      const auto register_info = register_file_.GetRegisterInfo(r);
      if (register_info) {
        REXAPU_DEBUG("XMA: Read from unhandled register ({:04X}, {})", r, register_info->name);
      } else {
        REXAPU_DEBUG("XMA: Read from unknown register ({:04X})", r);
      }
      break;
  }

  return rex::byte_swap(register_file_[r]);
}

void XmaDecoder::WriteRegister(uint32_t addr, uint32_t value) {
  SCOPE_profile_cpu_f("apu");

  uint32_t r = (addr & 0xFFFF) / 4;
  value = rex::byte_swap(value);

  assert_true(r < XmaRegisterFile::kRegisterCount);
  register_file_[r] = value;

  if (r >= XmaRegister::Context0Kick && r <= XmaRegister::Context9Kick) {
    // Context kick command.
    // This will kick off the given hardware contexts.
    // Basically, this kicks the SPU and says "hey, decode that audio!"
    // XMAEnableContext

    // The context ID is a bit in the range of the entire context array.
    uint32_t base_context_id = (r - XmaRegister::Context0Kick) * 32;
    uint32_t kicked_value = value;
    for (int i = 0; value && i < 32; ++i, value >>= 1) {
      if (value & 1) {
        uint32_t context_id = base_context_id + i;
        auto& context = contexts_[context_id];
        // [NARUTO-XMA-PROBE] counts ALL kicks (kernel API and MMIO-direct).
        context.probe_kicks_.fetch_add(1, std::memory_order_relaxed);
        context.Enable();
      }
    }
    // Signal the decoder thread to start processing.
    work_event_->Set();
    // Block until the worker finishes, so the game sees updated context data.
    for (int i = 0; kicked_value && i < 32; ++i, kicked_value >>= 1) {
      if (kicked_value & 1) {
        uint32_t context_id = base_context_id + i;
        contexts_[context_id].WaitForWorkDone();
      }
    }
  } else if (r >= XmaRegister::Context0Lock && r <= XmaRegister::Context9Lock) {
    // Context lock command.
    // This requests a lock by flagging the context.
    // XMADisableContext
    uint32_t base_context_id = (r - XmaRegister::Context0Lock) * 32;
    for (int i = 0; value && i < 32; ++i, value >>= 1) {
      if (value & 1) {
        uint32_t context_id = base_context_id + i;
        auto& context = contexts_[context_id];
        context.Disable();
        // [XMA fix] Added Block(false) after Disable(). Without this, the game
        // could call XMADisableContext and start modifying the context struct
        // while a decode was still in progress on the worker thread. Block()
        // waits for the context mutex to be free (poll=false means wait, not spin).
        context.Block(false);
      }
    }
    // Signal the decoder thread to start processing.
    // work_event_->Set();
  } else if (r >= XmaRegister::Context0Clear && r <= XmaRegister::Context9Clear) {
    // Context clear command.
    // This will reset the given hardware contexts.
    uint32_t base_context_id = (r - XmaRegister::Context0Clear) * 32;
    for (int i = 0; value && i < 32; ++i, value >>= 1) {
      if (value & 1) {
        uint32_t context_id = base_context_id + i;
        XmaContext& context = contexts_[context_id];
        context.Clear();
      }
    }
  } else {
    // 0601h (1804h) is written to with 0x02000000 and 0x03000000 around a lock
    // operation
    switch (r) {
      default: {
        const auto register_info = register_file_.GetRegisterInfo(r);
        if (register_info) {
          REXAPU_DEBUG("XMA: Write to unhandled register ({:04X}, {}): {:08X}", r,
                       register_info->name, value);
        } else {
          REXAPU_DEBUG("XMA: Write to unknown register ({:04X}): {:08X}", r, value);
        }
        break;
      }
#pragma warning(suppress : 4065)
    }
  }
}

// [NARUTO-XMA-PROBE] 1Hz status of every allocated context + XAudio submit
// rate. One line per context so a cut stream's last known state is on record.
void XmaDecoder::ProbeDumpStatus(uint64_t xaudio_submits) {
  static std::chrono::steady_clock::time_point last_dump{};
  static uint64_t last_submits = 0;
  const auto now = std::chrono::steady_clock::now();
  if (now - last_dump < std::chrono::seconds(1)) {
    return;
  }
  const auto submits_delta = xaudio_submits - last_submits;
  last_dump = now;
  last_submits = xaudio_submits;

  uint32_t allocated = 0;
  for (uint32_t n = 0; n < kContextCount; ++n) {
    if (contexts_[n].is_allocated()) {
      allocated++;
    }
  }
  REXAPU_INFO("[nrxma] === status: alloc={} xaudio_submits/s={} ===", allocated, submits_delta);

  for (uint32_t n = 0; n < kContextCount; ++n) {
    XmaContext& context = contexts_[n];
    if (!context.is_allocated()) {
      continue;
    }
    const uint32_t decodes = context.probe_decodes_.exchange(0);
    const uint32_t swaps = context.probe_swaps_.exchange(0);
    const uint32_t starves = context.probe_starves_.exchange(0);
    const uint32_t stalls = context.probe_out_stalls_.exchange(0);
    const uint32_t p_in0 = context.probe_poll_in0_.exchange(0);
    const uint32_t p_in1 = context.probe_poll_in1_.exchange(0);
    const uint32_t p_owr = context.probe_poll_outwr_.exchange(0);
    const uint32_t p_ord = context.probe_poll_outrd_.exchange(0);
    const uint32_t p_oval = context.probe_poll_outval_.exchange(0);
    const uint32_t p_ird = context.probe_poll_inrd_.exchange(0);
    const uint32_t s_ord = context.probe_set_outrd_.exchange(0);
    const uint32_t s_oval = context.probe_set_outval_.exchange(0);
    const uint32_t kicks = context.probe_kicks_.exchange(0);

    XMA_CONTEXT_DATA data(memory()->TranslateVirtual(context.guest_ptr()));
    REXAPU_INFO(
        "[nrxma] ctx={:03d} en={} in0={} in1={} cur={} pkts={}/{} rdoff={} base={} "
        "out(rd={} wr={} val={} blk={}) loop(cnt={} s={} e={}) err={}/{} "
        "dec/s={} swaps={} starves={} outstalls={} "
        "guest(kick={} pIn0={} pIn1={} pOwr={} pOrd={} pOval={} pIrd={} sOrd={} sOval={})",
        n, context.is_enabled() ? 1 : 0, static_cast<uint32_t>(data.input_buffer_0_valid),
        static_cast<uint32_t>(data.input_buffer_1_valid),
        static_cast<uint32_t>(data.current_buffer),
        static_cast<uint32_t>(data.input_buffer_0_packet_count),
        static_cast<uint32_t>(data.input_buffer_1_packet_count),
        static_cast<uint32_t>(data.input_buffer_read_offset), context.stream_base_bits(),
        static_cast<uint32_t>(data.output_buffer_read_offset),
        static_cast<uint32_t>(data.output_buffer_write_offset),
        static_cast<uint32_t>(data.output_buffer_valid),
        static_cast<uint32_t>(data.output_buffer_block_count),
        static_cast<uint32_t>(data.loop_count), static_cast<uint32_t>(data.loop_start),
        static_cast<uint32_t>(data.loop_end), static_cast<uint32_t>(data.error_status),
        static_cast<uint32_t>(data.error_set), decodes, swaps, starves, stalls, kicks, p_in0,
        p_in1, p_owr, p_ord, p_oval, p_ird, s_ord, s_oval);
  }
}

void XmaDecoder::Pause() {
  if (paused_) {
    return;
  }
  paused_ = true;

  if (work_event_) {
    work_event_->Set();
  }
  pause_fence_.Wait();
}

void XmaDecoder::Resume() {
  if (!paused_) {
    return;
  }
  paused_ = false;

  resume_fence_.Signal();
}

}  // namespace rex::audio
