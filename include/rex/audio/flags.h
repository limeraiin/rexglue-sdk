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

// [NARUTO-XMA-HWPAR] Hardware-parity XMA decode (see xma_context.cpp).
REXCVAR_DECLARE(bool, apu_xma_hw_parity);
// [NARUTO-XMA-HWPAR] Looper-tail starvation threshold (ms) before completing
// a guest-declared looper's pending split frame as silence.
REXCVAR_DECLARE(int32_t, apu_xma_looper_tail_ms);
// [NARUTO-XMA-HWPAR] Compact XMA decode probe logging ([nrxma] lines).
REXCVAR_DECLARE(bool, apu_xma_probe);
