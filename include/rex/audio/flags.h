/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <rex/cvar.h>

REXCVAR_DECLARE(bool, audio_mute);
REXCVAR_DECLARE(bool, ffmpeg_verbose);

// [NARUTO-XMA-PROBE] streaming-audio diagnostics + candidate race fix.
REXCVAR_DECLARE(bool, apu_xma_probe);
REXCVAR_DECLARE(bool, apu_xma_locked_ops);
// [NARUTO-XMA-STARVE] end-of-stream deadlock breaker (see xma_context.cpp).
REXCVAR_DECLARE(bool, apu_xma_starved_swap);
// [NARUTO-XMA-CUMUL] hardware-correct cumulative read offset + streamed loop wrap.
REXCVAR_DECLARE(bool, apu_xma_cumulative_rdoff);
// [NARUTO-XMA-SPLIT] immediate buffer release + split-frame cache (old-Xenia/HW behavior).
REXCVAR_DECLARE(bool, apu_xma_split_frame_cache);
// [NARUTO-XMA-SILENCE] emit silent frames on terminal split-frame starvation.
REXCVAR_DECLARE(bool, apu_xma_starved_silence);
// [NARUTO-XMA-LOOPEND] predict the loop seam from the game's loop_end (output-byte domain).
REXCVAR_DECLARE(bool, apu_xma_loopend_predict);
// [NARUTO-XMA-SILENCE] kicks a stream must be starved before silence breaks the watermark deadlock.
REXCVAR_DECLARE(int32_t, apu_xma_starve_kicks);
// [NARUTO-XMA-PACE] pace streamed-context decode output to the stream's real-time sample rate.
REXCVAR_DECLARE(bool, apu_xma_realtime_pace);
// [NARUTO-XMA-RINGTRACE] high-rate output-ring occupancy trajectory for streamed contexts (diagnostic).
REXCVAR_DECLARE(bool, apu_xma_ring_trace);
// [NARUTO-XMA-FIFO] host-side elastic decode-ahead FIFO for streamed contexts (mid-track music-gap fix).
REXCVAR_DECLARE(bool, apu_xma_elastic_fifo);
// [NARUTO-XMA-STOCK] stock-aware starve suppression: don't inject silence while the guest's
// downstream rendered-stock still covers playback (the REAL mid-track music-gap fix).
REXCVAR_DECLARE(bool, apu_xma_stock_starve);
REXCVAR_DECLARE(int32_t, apu_xma_stock_floor_ms);
