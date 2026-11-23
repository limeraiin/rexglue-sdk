/**
 * @file        rex/cpu_sampler.h
 * @brief       In-process sampling profiler for hosts without an external one
 *
 * Cvars (src/core/cpu_sampler.cpp): cpu_sample_hz (0 = off, no thread is
 * created), cpu_sample_seconds (0 = until the runtime shuts down) and
 * cpu_sample_delay. One sampler thread suspends every thread of the process
 * that used CPU lately, reads its instruction pointer and a shallow list of
 * likely return addresses from the stack (no unwinding, no dbghelp) and
 * counts them per thread. The counts are written beside the log file as
 * <log stem>.cpu-samples.txt when the time is up or at Stop(), whichever
 * comes first. Samples are wall-clock: a thread waiting in the kernel is
 * sampled at its wait, and the file carries the thread's CPU time over the
 * window so the two can be told apart.
 *
 * Implemented for Windows x64, where it also runs under Wine. A no-op
 * elsewhere.
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace rex::cpu_sampler {

// Starts the sampler thread when cpu_sample_hz > 0. Call once, after the
// cvars are loaded and logging is initialized.
void Start();

// Stops sampling and waits for the sample file to be written. Does nothing
// when the sampler never started or already finished.
void Stop();

// Remembers the name given to a host thread, by system thread id, for the
// sample file. Called from the thread naming path; cheap and always on.
void NoteThreadName(uint32_t thread_id, std::string_view name);

}  // namespace rex::cpu_sampler
