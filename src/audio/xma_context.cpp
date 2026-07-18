/**
******************************************************************************
* Xenia : Xbox 360 Emulator Research Project                                 *
******************************************************************************
* Copyright 2021 Ben Vanik. All rights reserved.                             *
* Released under the BSD license - see LICENSE in the root for more details. *
******************************************************************************
*
* @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
*/

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unordered_map>

#include <rex/audio/flags.h>
#include <rex/audio/xma/context.h>
#include <rex/audio/xma/decoder.h>
#include <rex/audio/xma/helpers.h>
#include <rex/dbg.h>
#include <rex/logging.h>
#include <rex/memory/ring_buffer.h>
#include <rex/platform.h>
#include <rex/stream.h>

extern "C" {
#if REX_COMPILER_MSVC
#pragma warning(push)
#pragma warning(disable : 4101 4244 5033)
#endif
#include "libavcodec/avcodec.h"
#include "libavutil/error.h"
#if REX_COMPILER_MSVC
#pragma warning(pop)
#endif
}  // extern "C"

// Credits for most of this code goes to:
// https://github.com/koolkdev/libertyv/blob/master/libav_wrapper/xma2dec.c

namespace rex::audio {

using stream::BitStream;

// [NARUTO-XMA-SEAM] persistent per-track loop-length cache. Only ever touched
// from the XMA decoder worker thread. Lives next to the executable.
static const char* kSeamCacheFile = "xma_loop_cache.txt";

static std::unordered_map<uint64_t, uint64_t>& SeamCache() {
  static std::unordered_map<uint64_t, uint64_t>* map = [] {
    auto* m = new std::unordered_map<uint64_t, uint64_t>();
    std::ifstream f(kSeamCacheFile);
    uint64_t key = 0, len = 0;
    while (f >> std::hex >> key >> len) {
      if (len) (*m)[key] = len;
    }
    return m;
  }();
  return *map;
}

static void SeamCacheStore(uint64_t key, uint64_t len) {
  auto& m = SeamCache();
  auto it = m.find(key);
  if (it != m.end() && it->second == len) {
    return;
  }
  m[key] = len;
  std::ofstream f(kSeamCacheFile, std::ios::trunc);
  for (const auto& entry : m) {
    f << std::hex << entry.first << ' ' << entry.second << '\n';
  }
}

static uint64_t SeamHash(const uint8_t* data, size_t size, uint64_t seed) {
  uint64_t h = 14695981039346656037ull ^ seed;
  for (size_t i = 0; i < size; ++i) {
    h ^= data[i];
    h *= 1099511628211ull;
  }
  return h;
}

const uint32_t XmaContext::kBitsPerPacketHeader;
const uint32_t XmaContext::kOutputMaxSizeBytes;

XmaContext::XmaContext()
    : work_completion_event_(rex::thread::Event::CreateAutoResetEvent(false)) {}

XmaContext::~XmaContext() {
  if (av_context_) {
    avcodec_free_context(&av_context_);
  }
  if (av_frame_) {
    av_frame_free(&av_frame_);
  }
}

int XmaContext::Setup(uint32_t id, memory::Memory* memory, uint32_t guest_ptr) {
  id_ = id;
  memory_ = memory;
  guest_ptr_ = guest_ptr;

  // Allocate ffmpeg stuff:
  av_packet_ = av_packet_alloc();
  assert_not_null(av_packet_);
  av_packet_->buf = av_buffer_alloc(128 * 1024);

  // find the XMA2 audio decoder
  av_codec_ = avcodec_find_decoder(AV_CODEC_ID_XMAFRAMES);
  if (!av_codec_) {
    REXAPU_ERROR("XmaContext {}: Codec not found", id);
    return 1;
  }

  av_context_ = avcodec_alloc_context3(av_codec_);
  if (!av_context_) {
    REXAPU_ERROR("XmaContext {}: Couldn't allocate context", id);
    return 1;
  }

  // Initialize these to 0. They'll actually be set later.
  av_context_->channels = 0;
  av_context_->sample_rate = 0;

  av_frame_ = av_frame_alloc();
  if (!av_frame_) {
    REXAPU_ERROR("XmaContext {}: Couldn't allocate frame", id);
    return 1;
  }

  // FYI: We're purposely not opening the codec here. That is done later.
  return 0;
}

bool XmaContext::Work() {
  if (!is_allocated() || !is_enabled()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(lock_);
  set_is_enabled(false);

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);
  const XMA_CONTEXT_DATA initial_data = data;

  // [NARUTO-XMA-CUMUL] The guest-visible read offset is cumulative across the
  // whole stream (HW semantics). Internally we decode with an intra-buffer
  // offset, so translate on entry; StoreContextMerged adds the base back.
  if (REXCVAR_GET(apu_xma_cumulative_rdoff)) {
    const uint64_t guest_offset = data.input_buffer_read_offset;
    if (guest_offset < stream_base_bits_) {
      // Guest rewrote the offset (fresh INIT/seek) - restart the base.
      stream_base_bits_ = 0;
    }
    uint64_t intra = guest_offset - stream_base_bits_;
    const uint64_t buffer_bits =
        uint64_t(data.GetCurrentInputBufferPacketCount()) * kBitsPerPacket;
    if (buffer_bits && intra > buffer_bits) {
      // Base desynced from a guest-side write; adopt the raw value.
      stream_base_bits_ = 0;
      intra = guest_offset;
    }
    data.input_buffer_read_offset = static_cast<uint32_t>(intra);
  }

  if (!data.output_buffer_valid) {
    return true;
  }

  memory::RingBuffer output_rb = PrepareOutputRingBuffer(&data);

  // [NARUTO-XMA-STOCK] Drain the guest downstream-stock estimate at the
  // stream's render rate (bytes/s = Hz x 2 bytes x channels). Accrual happens
  // in Consume() for real (non-injected) frames only. Cheap; runs regardless
  // of the cvar so a mid-run toggle starts from a sane estimate.
  {
    const auto now = std::chrono::steady_clock::now();
    if (stock_clock_valid_ && stock_bytes_ > 0.0) {
      const double dt = std::chrono::duration<double>(now - stock_last_).count();
      const double rate_bytes = double(GetSampleRate(data.sample_rate)) *
                                double(kBytesPerSample) * (data.is_stereo ? 2.0 : 1.0);
      stock_bytes_ = std::max(0.0, stock_bytes_ - dt * rate_bytes);
    }
    stock_last_ = now;
    stock_clock_valid_ = true;
  }

  // [NARUTO-XMA-RINGTRACE] high-rate occupancy trajectory for streamed contexts.
  // Occupancy hugging empty => the decoded-PCM stockpile is DOWNSTREAM of the
  // ring (input-limited, ~2 pkts); occupancy staying high => the ring holds it.
  // Sawtooth vs steady => bursty vs smooth pull. Correlate w/ [nrfeed] mark by
  // timestamp. Throttled ~50Hz/ctx; single-packet (streamed) contexts only.
  if (REXCVAR_GET(apu_xma_ring_trace) && data.GetCurrentInputBufferPacketCount() == 1) {
    const int64_t now_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
    if (now_us - ring_trace_last_us_ >= 20000) {
      ring_trace_last_us_ = now_us;
      REXAPU_INFO(
          "[nrxma-ring] ctx={:03d} cap={} rd={} wr={} used={} free={} in0={} in1={} cur={} "
          "rdoff={}",
          id(), static_cast<uint32_t>(data.output_buffer_block_count),
          static_cast<uint32_t>(data.output_buffer_read_offset),
          static_cast<uint32_t>(data.output_buffer_write_offset),
          static_cast<uint32_t>(output_rb.read_count() / kOutputBytesPerBlock),
          static_cast<uint32_t>(output_rb.write_count() / kOutputBytesPerBlock),
          static_cast<uint32_t>(data.input_buffer_0_valid),
          static_cast<uint32_t>(data.input_buffer_1_valid),
          static_cast<uint32_t>(data.current_buffer),
          static_cast<uint32_t>(data.input_buffer_read_offset));
    }
  }

  // Consume-only context: no input, just drain remaining subframes.
  if (data.IsConsumeOnlyContext()) {
    if (current_frame_remaining_subframes_ == 0) {
      return true;
    }
    Consume(&output_rb, &data);
    data.output_buffer_write_offset = output_rb.write_offset() / kOutputBytesPerBlock;
    StoreContextMerged(data, initial_data, context_ptr);
    return true;
  }

  // Minimum free blocks needed before attempting a decode.
  // Use subframe_decode_count (clamped to 1) instead of full frame size.
  const uint32_t effective_sdc = std::max(static_cast<uint32_t>(1), data.subframe_decode_count);
  const int32_t minimum_subframe_decode_count =
      static_cast<int32_t>(effective_sdc) + data.output_buffer_padding;

  // [NARUTO-XMA-FIFO] Elastic decode-ahead for streamed (single-packet) contexts:
  // buffer the producer's full lookahead as PCM so a 267-409ms AUTHORED producer
  // freeze doesn't starve the ~44ms guest ring (measured run 243). Streamed music
  // + layer streams only; in-memory multi-packet SFX keep the burst path.
  if (REXCVAR_GET(apu_xma_elastic_fifo) && data.GetCurrentInputBufferPacketCount() == 1) {
    WorkElasticFifo(&data, &output_rb, effective_sdc);
  } else {
    if (minimum_subframe_decode_count > remaining_subframe_blocks_in_output_buffer_) {
      // [NARUTO-XMA-PROBE] Output backpressure (game not draining) - normal, counted only.
      probe_out_stalls_.fetch_add(1, std::memory_order_relaxed);
      StoreContextMerged(data, initial_data, context_ptr);
      return true;
    }
    DecodeLoopInto(&output_rb, &data, minimum_subframe_decode_count, /*skip_when_dry=*/false);
  }

  data.output_buffer_write_offset = output_rb.write_offset() / kOutputBytesPerBlock;

  // [NARUTO-XMA-FIFO] Keep output_buffer_valid set while the FIFO still holds PCM
  // to drain next Work() (else the early return at the top of Work() strands it).
  if (output_rb.empty() && FifoEmpty()) {
    data.output_buffer_valid = 0;
  }

  StoreContextMerged(data, initial_data, context_ptr);
  return true;
}

// [NARUTO-XMA-FIFO] The Work() decode loop body, factored so it can target either
// the guest output ring (legacy) or the elastic FIFO. `remaining_subframe_blocks_
// in_output_buffer_` MUST be primed to `rb`'s free block count by the caller.
// `skip_when_dry`: break BEFORE decoding when the input stream is dry, so the FIFO
// is never filled with silence (the silence/seam drip belongs on the guest ring).
void XmaContext::DecodeLoopInto(memory::RingBuffer* rb, XMA_CONTEXT_DATA* data,
                                int32_t minimum_blocks, bool skip_when_dry) {
  while (remaining_subframe_blocks_in_output_buffer_ >= minimum_blocks) {
    if (skip_when_dry && current_frame_remaining_subframes_ == 0 &&
        !data->IsAnyInputBufferValid()) {
      break;
    }
    Decode(data);
    Consume(rb, data);

    if (!data->IsAnyInputBufferValid() || data->error_status == 4) {
      // [NARUTO-XMA-PROBE] Stream ran dry (or errored) mid-work. Log once per
      // dry-out: a starve on a stream the game meant to keep feeding is the
      // "long sound cuts" smoking gun.
      if (REXCVAR_GET(apu_xma_probe) && probe_was_streaming_) {
        probe_was_streaming_ = false;
        probe_starves_.fetch_add(1, std::memory_order_relaxed);
        REXAPU_INFO(
            "[nrxma] ctx={:03d} INPUT DRY: in0={} in1={} cur={} rdoff={} err={} "
            "out(rd={} wr={})",
            id(), static_cast<uint32_t>(data->input_buffer_0_valid),
            static_cast<uint32_t>(data->input_buffer_1_valid),
            static_cast<uint32_t>(data->current_buffer),
            static_cast<uint32_t>(data->input_buffer_read_offset),
            static_cast<uint32_t>(data->error_status),
            static_cast<uint32_t>(data->output_buffer_read_offset),
            static_cast<uint32_t>(data->output_buffer_write_offset));
      }
      break;
    }
  }
}

// [NARUTO-XMA-FIFO] Elastic decode-ahead. Phase A: decode as far ahead as the
// input allows into the host FIFO (never silence). Phase B: meter FIFO -> guest
// ring, topping it up. Phase C: only when the FIFO is empty AND input is dry is
// it a genuine starve -> the legacy silence/seam drip into the guest ring. This
// converts the authored producer freeze from a silent gap into buffered real
// audio, while every end-of-stream / loop-seam path stays exactly as before.
void XmaContext::WorkElasticFifo(XMA_CONTEXT_DATA* data, memory::RingBuffer* output_rb,
                                 uint32_t effective_sdc) {
  if (fifo_.size() != kFifoBytes) {
    fifo_.assign(kFifoBytes, 0);
    fifo_read_ = fifo_write_ = 0;
  }
  memory::RingBuffer fifo(fifo_.data(), kFifoBytes);
  fifo.set_read_offset(fifo_read_);
  fifo.set_write_offset(fifo_write_);

  const int32_t min_fill = static_cast<int32_t>(effective_sdc);  // no guest padding for the FIFO

  // Phase A: decode-ahead into the FIFO (bounded by input availability + FIFO
  // space; the producer's watermark gate caps how far ahead it feeds). Reserve
  // one block so a fill can never make write==read (which this RingBuffer reads
  // as EMPTY, discarding the whole FIFO).
  remaining_subframe_blocks_in_output_buffer_ =
      static_cast<int32_t>(fifo.write_count() / kOutputBytesPerBlock) - 1;
  DecodeLoopInto(&fifo, data, min_fill, /*skip_when_dry=*/true);
  fifo_read_ = fifo.read_offset();
  fifo_write_ = fifo.write_offset();

  // Phase B: drain FIFO -> guest ring, in block-aligned chunks. Leave the same
  // reserve the legacy loop would (effective_sdc + output_buffer_padding blocks):
  // this RingBuffer has no full state (read==write reads as EMPTY), so writing
  // until write catches read would silently overwrite unread PCM.
  uint8_t tmp[4096];
  const uint32_t chunk_cap = sizeof(tmp) - (sizeof(tmp) % kOutputBytesPerBlock);
  const uint32_t reserve_bytes =
      (effective_sdc + data->output_buffer_padding) * kOutputBytesPerBlock;
  for (;;) {
    const uint32_t free = output_rb->write_count();
    if (free <= reserve_bytes) {
      break;
    }
    uint32_t n = std::min<uint32_t>(free - reserve_bytes, fifo.read_count());
    n -= n % kOutputBytesPerBlock;
    if (n == 0) {
      break;
    }
    if (n > chunk_cap) {
      n = chunk_cap;
    }
    fifo.Read(tmp, n);
    output_rb->Write(tmp, n);
  }
  fifo_read_ = fifo.read_offset();
  fifo_write_ = fifo.write_offset();

  // Phase C: genuine starvation (nothing buffered AND input dry) -> the legacy
  // silence/seam path, dripping into the guest ring exactly as non-FIFO mode.
  if (fifo.empty() && !data->IsAnyInputBufferValid()) {
    remaining_subframe_blocks_in_output_buffer_ =
        static_cast<int32_t>(output_rb->write_count() / kOutputBytesPerBlock);
    const int32_t minimum = static_cast<int32_t>(effective_sdc) + data->output_buffer_padding;
    DecodeLoopInto(output_rb, data, minimum, /*skip_when_dry=*/false);
  }
}

void XmaContext::Enable() {
  std::lock_guard<std::mutex> lock(lock_);
  set_is_enabled(true);
}

bool XmaContext::Block(bool poll) {
  if (!lock_.try_lock()) {
    if (poll) {
      return false;
    }
    lock_.lock();
  }
  lock_.unlock();
  return true;
}

void XmaContext::Clear() {
  std::lock_guard<std::mutex> lock(lock_);
  REXAPU_NOISY_DEBUG("XmaContext: reset context {}", id());
  // [NARUTO-XMA-PROBE]
  if (REXCVAR_GET(apu_xma_probe)) {
    REXAPU_INFO("[nrxma] ctx={:03d} CLEAR (guest reset)", id());
  }

  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  XMA_CONTEXT_DATA data(context_ptr);
  ClearLocked(&data);
  data.Store(context_ptr);
}

void XmaContext::ClearLocked(XMA_CONTEXT_DATA* data) {
  data->input_buffer_0_valid = 0;
  data->input_buffer_1_valid = 0;
  data->output_buffer_valid = 0;

  data->input_buffer_read_offset = kBitsPerPacketHeader;
  data->output_buffer_read_offset = 0;
  data->output_buffer_write_offset = 0;

  current_frame_remaining_subframes_ = 0;
  loop_frame_output_limit_ = 0;
  loop_start_skip_pending_ = false;
  probe_was_streaming_ = false;  // [NARUTO-XMA-PROBE]
  starve_run_count_ = 0;         // [NARUTO-XMA-STARVE]
  stream_base_bits_ = 0;         // [NARUTO-XMA-CUMUL]
  split_pending_ = false;        // [NARUTO-XMA-SPLIT]
  split_starve_kicks_ = 0;       // [NARUTO-XMA-SILENCE]
  seam_predicted_ = false;       // [NARUTO-XMA-SEAM]
  loop_input_len_bits_ = 0;
  last_seam_input_bits_ = 0;
  tentative_loop_len_bits_ = 0;
  decoded_output_bytes_ = 0;     // [NARUTO-XMA-LOOPEND]
  ResetFifo();                   // [NARUTO-XMA-FIFO]
  ResetStock();                  // [NARUTO-XMA-STOCK]
}

void XmaContext::Disable() {
  std::lock_guard<std::mutex> lock(lock_);
  set_is_enabled(false);
}

void XmaContext::Release() {
  std::lock_guard<std::mutex> lock(lock_);
  assert_true(is_allocated());

  set_is_allocated(false);
  auto context_ptr = memory()->TranslateVirtual(guest_ptr());
  std::memset(context_ptr, 0, sizeof(XMA_CONTEXT_DATA));

  // [NARUTO-XMA-SPLIT]/[NARUTO-XMA-CUMUL] don't leak stream state into the
  // next sound that recycles this context.
  split_pending_ = false;
  split_starve_kicks_ = 0;
  stream_base_bits_ = 0;
  starve_run_count_ = 0;
  probe_was_streaming_ = false;
  seam_predicted_ = false;  // [NARUTO-XMA-SEAM]
  loop_input_len_bits_ = 0;
  last_seam_input_bits_ = 0;
  tentative_loop_len_bits_ = 0;
  stream_key_valid_ = false;
  decoded_output_bytes_ = 0;  // [NARUTO-XMA-LOOPEND]
  ResetFifo();                // [NARUTO-XMA-FIFO]
  ResetStock();               // [NARUTO-XMA-STOCK]
}

void XmaContext::SwapInputBuffer(XMA_CONTEXT_DATA* data) {
  // [NARUTO-XMA-CUMUL] the retired buffer's bits advance the stream base.
  if (REXCVAR_GET(apu_xma_cumulative_rdoff)) {
    stream_base_bits_ += uint64_t(data->GetCurrentInputBufferPacketCount()) * kBitsPerPacket;
  }
  if (data->current_buffer == 0) {
    data->input_buffer_0_valid = 0;
  } else {
    data->input_buffer_1_valid = 0;
  }
  data->current_buffer ^= 1;
  data->input_buffer_read_offset = kBitsPerPacketHeader;
  starve_run_count_ = 0;  // [NARUTO-XMA-STARVE]

  // [NARUTO-XMA-PROBE] Buffer swap = the streaming heartbeat (one per consumed
  // input buffer). Long sounds produce a steady run of these until the cut.
  probe_swaps_.fetch_add(1, std::memory_order_relaxed);
  if (REXCVAR_GET(apu_xma_probe)) {
    REXAPU_INFO("[nrxma] ctx={:03d} SWAP -> cur={} next_valid={} frames_decoded={}", id(),
                static_cast<uint32_t>(data->current_buffer),
                data->IsCurrentInputBufferValid() ? 1 : 0, probe_pkt_frames_);
  }
  probe_pkt_frames_ = 0;      // [NARUTO-XMA-PKT] per-buffer census resets here
  probe_pkt_hdr_logged_ = false;
}

void XmaContext::UpdateLoopStatus(XMA_CONTEXT_DATA* data) {
  if (data->loop_count == 0) {
    return;
  }

  const uint32_t loop_start = std::max(kBitsPerPacketHeader, data->loop_start);
  const uint32_t loop_end = std::max(kBitsPerPacketHeader, data->loop_end);

  // [NARUTO-XMA-LOOPEND] The DECODER-INTERNAL loop (rewind read offset to
  // loop_start) only applies to IN-MEMORY sounds whose whole loop region is
  // resident in the input buffer, so it fires on an EXACT read-offset match.
  // For Naruto's STREAMED music the game re-feeds and loop_end is an
  // output-domain marker (see apu_xma_loopend_predict) — a cumulative
  // input-bit compare here would spuriously fire mid-stream, so it is gone.
  if (data->input_buffer_read_offset != loop_end) {
    return;
  }

  data->input_buffer_read_offset = loop_start;
  loop_start_skip_pending_ = true;

  if (data->loop_count < 255) {
    data->loop_count--;
  }

  // [NARUTO-XMA-PROBE]
  if (REXCVAR_GET(apu_xma_probe)) {
    REXAPU_INFO("[nrxma] ctx={:03d} LOOP taken: -> start={} remaining_count={}", id(), loop_start,
                static_cast<uint32_t>(data->loop_count));
  }
}

int XmaContext::GetSampleRate(int id) {
  return kIdToSampleRate[std::min(id, 3)];
}

int16_t XmaContext::GetPacketNumber(size_t size, size_t bit_offset) {
  if (bit_offset < kBitsPerPacketHeader) {
    assert_always();
    return -1;
  }
  if (bit_offset >= (size << 3)) {
    assert_always();
    return -1;
  }
  size_t byte_offset = bit_offset >> 3;
  size_t packet_number = byte_offset / kBytesPerPacket;
  return static_cast<int16_t>(packet_number);
}

uint32_t XmaContext::GetCurrentInputBufferSize(XMA_CONTEXT_DATA* data) {
  return data->GetCurrentInputBufferPacketCount() * kBytesPerPacket;
}

uint8_t* XmaContext::GetCurrentInputBuffer(XMA_CONTEXT_DATA* data) {
  return memory()->TranslatePhysical(data->GetCurrentInputBufferAddress());
}

uint32_t XmaContext::GetAmountOfBitsToRead(uint32_t remaining_stream_bits, uint32_t frame_size) {
  return std::min(remaining_stream_bits, frame_size);
}

const uint8_t* XmaContext::GetNextPacket(XMA_CONTEXT_DATA* data, uint32_t next_packet_index,
                                         uint32_t current_input_packet_count) {
  if (next_packet_index < current_input_packet_count) {
    return memory()->TranslatePhysical(data->GetCurrentInputBufferAddress()) +
           next_packet_index * kBytesPerPacket;
  }

  const uint8_t next_buffer_index = data->current_buffer ^ 1;
  if (!data->IsInputBufferValid(next_buffer_index)) {
    return nullptr;
  }

  const uint32_t next_buffer_address = data->GetInputBufferAddress(next_buffer_index);
  if (!next_buffer_address) {
    REXAPU_ERROR("XmaContext {}: Buffer marked valid but has null pointer!", id());
    return nullptr;
  }

  return memory()->TranslatePhysical(next_buffer_address);
}

uint32_t XmaContext::GetNextPacketReadOffset(uint8_t* buffer, uint32_t next_packet_index,
                                             uint32_t current_input_packet_count) {
  while (next_packet_index < current_input_packet_count) {
    uint8_t* next_packet = buffer + (next_packet_index * kBytesPerPacket);
    const uint32_t packet_frame_offset = xma::GetPacketFrameOffset(next_packet);

    if (packet_frame_offset <= kMaxFrameSizeinBits) {
      return (next_packet_index * kBitsPerPacket) + packet_frame_offset;
    }
    next_packet_index++;
  }

  return kBitsPerPacketHeader;
}

memory::RingBuffer XmaContext::PrepareOutputRingBuffer(XMA_CONTEXT_DATA* data) {
  const uint32_t output_capacity = data->output_buffer_block_count * kOutputBytesPerBlock;
  const uint32_t output_read_offset = data->output_buffer_read_offset * kOutputBytesPerBlock;
  const uint32_t output_write_offset = data->output_buffer_write_offset * kOutputBytesPerBlock;

  if (output_capacity > kOutputMaxSizeBytes) {
    REXAPU_WARN(
        "XmaContext {}: Output buffer exceeds expected size! "
        "(Actual: {} Max: {})",
        id(), output_capacity, kOutputMaxSizeBytes);
  }

  uint8_t* output_buffer = memory()->TranslatePhysical(data->output_buffer_ptr);

  memory::RingBuffer output_rb(output_buffer, output_capacity);
  output_rb.set_read_offset(output_read_offset);
  output_rb.set_write_offset(output_write_offset);
  remaining_subframe_blocks_in_output_buffer_ =
      static_cast<int32_t>(output_rb.write_count()) / kOutputBytesPerBlock;

  return output_rb;
}

kPacketInfo XmaContext::GetPacketInfo(uint8_t* packet, uint32_t frame_offset) {
  kPacketInfo packet_info = {};

  const uint32_t first_frame_offset = xma::GetPacketFrameOffset(packet);
  BitStream stream(packet, kBitsPerPacket);
  stream.SetOffset(first_frame_offset);

  if (frame_offset < first_frame_offset) {
    packet_info.current_frame_ = 0;
    packet_info.current_frame_size_ = first_frame_offset - frame_offset;
  }

  while (true) {
    if (stream.BitsRemaining() < kBitsPerFrameHeader) {
      break;
    }

    const uint64_t frame_size = stream.Peek(kBitsPerFrameHeader);
    if (frame_size == 0 || frame_size == xma::kMaxFrameLength) {
      break;
    }

    if (stream.offset_bits() == frame_offset) {
      packet_info.current_frame_ = packet_info.frame_count_;
      packet_info.current_frame_size_ = static_cast<uint32_t>(frame_size);
    }

    packet_info.frame_count_++;

    if (frame_size > stream.BitsRemaining()) {
      break;
    }

    stream.Advance(frame_size - 1);

    if (stream.Read(1) == 0) {
      break;
    }
  }

  if (xma::IsPacketXma2Type(packet)) {
    const uint8_t xma2_frame_count = xma::GetPacketFrameCount(packet);
    if (xma2_frame_count > packet_info.frame_count_) {
      if (packet_info.current_frame_size_ == 0) {
        packet_info.current_frame_ = packet_info.frame_count_;
      }
      packet_info.frame_count_ = xma2_frame_count;
    }
  }
  return packet_info;
}

void XmaContext::StoreContextMerged(const XMA_CONTEXT_DATA& data,
                                    const XMA_CONTEXT_DATA& initial_data, uint8_t* context_ptr) {
  XMA_CONTEXT_DATA fresh(context_ptr);

  fresh.loop_count = data.loop_count;
  fresh.output_buffer_write_offset = data.output_buffer_write_offset;
  if (initial_data.input_buffer_0_valid && !data.input_buffer_0_valid) {
    fresh.input_buffer_0_valid = 0;
  }
  if (initial_data.input_buffer_1_valid && !data.input_buffer_1_valid) {
    fresh.input_buffer_1_valid = 0;
  }

  if (initial_data.output_buffer_valid && !data.output_buffer_valid) {
    fresh.output_buffer_valid = 0;
  }

  // [NARUTO-XMA-CUMUL] guest sees the cumulative stream offset (HW semantics).
  if (REXCVAR_GET(apu_xma_cumulative_rdoff)) {
    fresh.input_buffer_read_offset =
        static_cast<uint32_t>(stream_base_bits_ + data.input_buffer_read_offset);
  } else {
    fresh.input_buffer_read_offset = data.input_buffer_read_offset;
  }
  fresh.error_status = data.error_status;
  fresh.current_buffer = data.current_buffer;
  fresh.output_buffer_read_offset = data.output_buffer_read_offset;

  fresh.Store(context_ptr);
}

void XmaContext::Consume(memory::RingBuffer* output_rb, const XMA_CONTEXT_DATA* data) {
  if (!current_frame_remaining_subframes_) {
    return;
  }

  if (loop_frame_output_limit_ > 0) {
    const uint8_t total_subframes = (kBytesPerFrameChannel / kOutputBytesPerBlock)
                                    << data->is_stereo;
    const uint8_t consumed = total_subframes - current_frame_remaining_subframes_;
    if (consumed >= loop_frame_output_limit_) {
      remaining_subframe_blocks_in_output_buffer_ -= data->output_buffer_padding;
      current_frame_remaining_subframes_ = 0;
      loop_frame_output_limit_ = 0;
      return;
    }
  }

  const uint8_t effective_sdc = std::max(static_cast<uint32_t>(1), data->subframe_decode_count);
  int8_t subframes_to_write = std::min(static_cast<int8_t>(current_frame_remaining_subframes_),
                                       static_cast<int8_t>(effective_sdc));

  if (loop_frame_output_limit_ > 0) {
    const uint8_t total_subframes = (kBytesPerFrameChannel / kOutputBytesPerBlock)
                                    << data->is_stereo;
    const uint8_t consumed = total_subframes - current_frame_remaining_subframes_;
    const int8_t remaining_until_limit = static_cast<int8_t>(loop_frame_output_limit_ - consumed);
    if (subframes_to_write > remaining_until_limit) {
      subframes_to_write = remaining_until_limit;
    }
  }

  const int8_t raw_frame_read_offset =
      ((kBytesPerFrameChannel / kOutputBytesPerBlock) << data->is_stereo) -
      current_frame_remaining_subframes_;

  output_rb->Write(raw_frame_.data() + (kOutputBytesPerBlock * raw_frame_read_offset),
                   subframes_to_write * kOutputBytesPerBlock);

  // [NARUTO-XMA-STOCK] Ledger real PCM delivered to the guest; injected
  // silence is deliberately excluded (it doesn't represent track progress).
  if (!current_frame_is_silence_) {
    stock_bytes_ += double(subframes_to_write) * kOutputBytesPerBlock;
  }

  const int8_t headroom = (current_frame_remaining_subframes_ - subframes_to_write == 0)
                              ? data->output_buffer_padding
                              : 0;

  remaining_subframe_blocks_in_output_buffer_ -= subframes_to_write + headroom;
  current_frame_remaining_subframes_ -= subframes_to_write;
}

int XmaContext::PrepareDecoder(int sample_rate, bool is_two_channel) {
  sample_rate = GetSampleRate(sample_rate);

  uint32_t channels = is_two_channel ? 2 : 1;
  if (av_context_->sample_rate != sample_rate ||
      av_context_->channels != static_cast<int>(channels)) {
    REXAPU_NOISY_DEBUG("XmaContext {}: Codec reinit: rate {} -> {}, channels {} -> {}", id(),
                       av_context_->sample_rate, sample_rate, av_context_->channels, channels);
    avcodec_free_context(&av_context_);
    av_context_ = avcodec_alloc_context3(av_codec_);

    av_context_->sample_rate = sample_rate;
    av_context_->channels = channels;
    av_context_->flags2 |= AV_CODEC_FLAG2_SKIP_MANUAL;

    if (avcodec_open2(av_context_, av_codec_, NULL) < 0) {
      REXAPU_ERROR("XmaContext: Failed to reopen FFmpeg context");
      return -1;
    }
    return 1;
  }
  return 0;
}

void XmaContext::PreparePacket(uint32_t frame_size, uint32_t frame_padding) {
  av_packet_->data = xma_frame_.data();
  av_packet_->size = static_cast<int>(1 + ((frame_padding + frame_size) / 8) +
                                      (((frame_padding + frame_size) % 8) ? 1 : 0));

  auto padding_end = av_packet_->size * 8 - (8 + frame_padding + frame_size);
  assert_true(padding_end < 8);
  xma_frame_[0] = ((frame_padding & 7) << 5) | ((padding_end & 7) << 2);
}

bool XmaContext::DecodePacket(AVCodecContext* av_context, const AVPacket* av_packet,
                              AVFrame* av_frame) {
  auto ret = avcodec_send_packet(av_context, av_packet);
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errbuf, sizeof(errbuf));
    REXAPU_ERROR("XmaContext {}: Error sending packet for decoding: {} ({})", id(), errbuf, ret);
    return false;
  }
  ret = avcodec_receive_frame(av_context, av_frame);

  if (ret == AVERROR(EAGAIN)) {
    return false;
  }
  if (ret < 0) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, errbuf, sizeof(errbuf));
    REXAPU_ERROR("XmaContext {}: Error during decoding: {} ({})", id(), errbuf, ret);
    return false;
  }
  return true;
}

void XmaContext::Decode(XMA_CONTEXT_DATA* data) {
  SCOPE_profile_cpu_f("apu");

  if (!data->IsAnyInputBufferValid()) {
    // [NARUTO-XMA-SILENCE] Starved at a cached split frame: after well beyond
    // any healthy refill latency (measured <=121 kicks), emit silent frames
    // (one per 16 kicks) so the guest's rendered-samples watermark advances
    // past the truncated packet and its feeder produces the loop-wrap packet.
    // Real hardware renders these tail frames too (as stale-bit garbage).
    if (split_pending_ && current_frame_remaining_subframes_ == 0 &&
        REXCVAR_GET(apu_xma_starved_silence)) {
      // [NARUTO-XMA-STOCK] While the guest's downstream stock still covers
      // playback, this "starvation" is the feeder's BY-DESIGN idle (its
      // +2-entry lookahead - pre-charged by the authored 20-39-frame packets -
      // playing out downstream; run-239 anatomy). Injecting silence here is
      // what CAUSED the audible mid-track gap: the game slurps it behind the
      // real stock and plays it ~300ms later. Hold the entire timeout machine
      // (kick counter, seam learning, emission) until the stock estimate
      // drains to the floor; a TRUE starvation (end-of-stream / loop park)
      // then proceeds exactly as before. Predicted seams are exempt: their
      // instant injection is the proven gapless-loop mechanism.
      if (!seam_predicted_ && REXCVAR_GET(apu_xma_stock_starve)) {
        const double rate_bytes = double(GetSampleRate(data->sample_rate)) *
                                  double(kBytesPerSample) * (data->is_stereo ? 2.0 : 1.0);
        const double floor_bytes =
            double(std::max(0, REXCVAR_GET(apu_xma_stock_floor_ms))) * 0.001 * rate_bytes;
        if (stock_bytes_ > floor_bytes) {
          if (REXCVAR_GET(apu_xma_probe) && stock_suppressed_++ == 0) {
            REXAPU_INFO(
                "[nrxma] ctx={:03d} STARVE SUPPRESSED (stock~{}ms covers playback; by-design "
                "feeder idle, no silence)",
                id(), static_cast<uint32_t>(stock_bytes_ * 1000.0 / rate_bytes));
          }
          return;
        }
        if (REXCVAR_GET(apu_xma_probe) && stock_suppressed_ != 0) {
          REXAPU_INFO(
              "[nrxma] ctx={:03d} STOCK EXHAUSTED after {} suppressed kicks - starve machine "
              "resumes",
              id(), stock_suppressed_);
          stock_suppressed_ = 0;
        }
      }
      split_starve_kicks_++;
      const uint32_t starve_threshold =
          std::max(8, REXCVAR_GET(apu_xma_starve_kicks));  // [NARUTO-XMA-SILENCE]
      // [NARUTO-XMA-SEAM] a starvation surviving the full timeout is a real
      // seam: learn/refresh the loop length so later seams predict instantly.
      if (!seam_predicted_ && split_starve_kicks_ == starve_threshold && data->loop_count > 0) {
        // A starve timeout is EITHER a real loop seam OR a long mid-track feed
        // stall — indistinguishable here (2026-07-15: a stall one packet into a
        // track learned a 14,556-bit "loop" => silence after every packet =
        // fully stuttery music, persisted via the cache). Guards (context.h):
        // length floor + only trust a length observed twice in a row.
        uint64_t candidate = 0;
        if (last_seam_input_bits_ != 0 && episode_input_bits_ > last_seam_input_bits_) {
          candidate = episode_input_bits_ - last_seam_input_bits_;
        } else if (last_seam_input_bits_ == 0) {
          // First seam: the loop typically spans the whole streamed region.
          candidate = episode_input_bits_;
        }
        last_seam_input_bits_ = episode_input_bits_;
        if (candidate >= kSeamMinLoopInputBits) {
          const uint64_t diff = candidate > tentative_loop_len_bits_
                                    ? candidate - tentative_loop_len_bits_
                                    : tentative_loop_len_bits_ - candidate;
          if (tentative_loop_len_bits_ != 0 && diff <= kBitsPerPacket) {
            loop_input_len_bits_ = candidate;
            if (stream_key_valid_) {
              SeamCacheStore(stream_key_, loop_input_len_bits_);  // persists across sessions
            }
            if (REXCVAR_GET(apu_xma_probe)) {
              REXAPU_INFO("[nrxma] ctx={:03d} SEAM LEARNED: input_bits={} loop_len={}", id(),
                          episode_input_bits_, loop_input_len_bits_);
            }
          } else {
            tentative_loop_len_bits_ = candidate;
            if (REXCVAR_GET(apu_xma_probe)) {
              REXAPU_INFO("[nrxma] ctx={:03d} SEAM CANDIDATE (unconfirmed): input_bits={} "
                          "cand_len={}",
                          id(), episode_input_bits_, candidate);
            }
          }
        } else if (candidate != 0 && REXCVAR_GET(apu_xma_probe)) {
          REXAPU_INFO(
              "[nrxma] ctx={:03d} SEAM REJECTED (len {} below floor => stall, not a seam)",
              id(), candidate);
        }
      }
      // At a predicted loop seam, emit on every pass (the output ring itself
      // paces us); otherwise wait out possible refills.
      if (seam_predicted_ ||
          (split_starve_kicks_ >= starve_threshold && (split_starve_kicks_ & 7) == 0)) {
        raw_frame_.fill(0);
        current_frame_is_silence_ = true;  // [NARUTO-XMA-STOCK] don't ledger injected frames
        current_frame_remaining_subframes_ = 4 << data->is_stereo;
        loop_frame_output_limit_ = 0;
        if (REXCVAR_GET(apu_xma_probe)) {
          // (guard the subtraction: on the seam_predicted_ path kicks can be
          // below the threshold and the unsigned difference underflows)
          const uint32_t silent_no = split_starve_kicks_ > starve_threshold
                                         ? (split_starve_kicks_ - starve_threshold) / 8 + 1
                                         : 1;
          REXAPU_INFO(
              "[nrxma] ctx={:03d} STARVED-SILENCE: silent frame #{} (advancing guest render "
              "watermark to unblock its loop wrap)",
              id(), silent_no);
        }
      }
    }
    return;
  }

  if (current_frame_remaining_subframes_ > 0) {
    return;
  }

  // [NARUTO-XMA-PACE] Streamed contexts (single-packet 2KB input buffers, the
  // game's music/layer feed pattern) decode at the stream's real-time sample
  // rate + a small headroom, like real hardware. Without this our
  // faster-than-real-time extraction lets the game's puller run ~a packet-lump
  // ahead of its rendered-samples watermark, and its watermark-keyed producer
  // gate then starves the pipeline at every authored high-frame-count packet
  // (the ~7.7s-periodic mid-track music gap). In-memory SFX contexts
  // (multi-packet buffers) keep burst decode.
  if (REXCVAR_GET(apu_xma_realtime_pace) &&
      data->GetCurrentInputBufferPacketCount() == 1) {
    const auto now = std::chrono::steady_clock::now();
    const uint64_t rate =
        static_cast<uint64_t>(GetSampleRate(data->sample_rate));
    if (!pace_valid_) {
      pace_valid_ = true;
      pace_epoch_ = now;
      pace_frames_ = 0;
    }
    const uint64_t elapsed_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(now - pace_epoch_)
            .count());
    const uint64_t elapsed_samples = elapsed_us * rate / 1000000ull;
    const uint64_t released = pace_frames_ * kSamplesPerFrame;
    // Headroom must exceed the game's per-tick pull burst (~8-10 frames /
    // ~4800 samples measured) or every pull under-delivers = stutter + slow
    // audio (v1 used 4 => broken). It must stay below the ~17-20 frame phase
    // error that reopens the watermark gap; 12 frames = ~139ms @44.1k.
    const uint64_t headroom = kSamplesPerFrame * 12;
    if (released + kSamplesPerFrame > elapsed_samples + headroom) {
      return;  // ahead of schedule: wait for a later kick (~0.5ms cadence)
    }
    if (released + kSamplesPerFrame + headroom < elapsed_samples) {
      // Far behind schedule (first decode after idle / an input stall): re-
      // anchor instead of bursting a catch-up lump (the lump IS the bug).
      pace_epoch_ =
          now - std::chrono::microseconds((released + headroom) * 1000000ull / rate);
    }
  }

  if (!data->IsCurrentInputBufferValid()) {
    SwapInputBuffer(data);
    if (!data->IsCurrentInputBufferValid()) {
      return;
    }
  }

  uint8_t* current_input_buffer = GetCurrentInputBuffer(data);

  // [NARUTO-XMA-SEAM] Identify the track by its first packet; a previously
  // learned loop length makes even the FIRST seam predict gaplessly.
  if (!stream_key_valid_ && stream_base_bits_ == 0 &&
      REXCVAR_GET(apu_xma_starved_silence)) {
    stream_key_valid_ = true;
    stream_key_ = SeamHash(current_input_buffer, kBytesPerPacket,
                           (uint64_t(data->sample_rate) << 1) | data->is_stereo);
    if (loop_input_len_bits_ == 0) {
      auto& cache = SeamCache();
      auto it = cache.find(stream_key_);
      // (ignore below-floor entries: poisoned by pre-2026-07-15 stall-learns)
      if (it != cache.end() && it->second >= kSeamMinLoopInputBits) {
        loop_input_len_bits_ = it->second;
        last_seam_input_bits_ = 0;
        if (REXCVAR_GET(apu_xma_probe)) {
          REXAPU_INFO("[nrxma] ctx={:03d} SEAM CACHE HIT: key={:016X} loop_len={}", id(),
                      stream_key_, loop_input_len_bits_);
        }
      }
    }
  }

  // [NARUTO-XMA-SPLIT] a frame cached at the previous buffer's end completes
  // from this buffer's first packet before any new parsing.
  if (split_pending_) {
    ResumeSplitFrame(data, current_input_buffer);
    return;
  }

  input_buffer_.fill(0);

  // Detect loop end frame before UpdateLoopStatus resets the offset.
  // [NARUTO-XMA-LOOPEND] exact-match only (in-memory loop); the streamed loop
  // seam is handled by output-domain prediction, not here.
  bool is_loop_end_frame = false;
  if (data->loop_count > 0) {
    const uint32_t loop_end = std::max(kBitsPerPacketHeader, data->loop_end);
    is_loop_end_frame = (data->input_buffer_read_offset == loop_end);
  }

  // [NARUTO-XMA-CUMUL] A streamed loop wrap retires the current buffer;
  // current_input_buffer above is stale then, so re-enter on the next pass
  // (the Work loop calls Decode again with fresh state, or waits for the
  // game to feed the loop-start packets).
  const uint8_t pre_loop_current_buffer = static_cast<uint8_t>(data->current_buffer);
  UpdateLoopStatus(data);
  if (static_cast<uint8_t>(data->current_buffer) != pre_loop_current_buffer) {
    return;
  }

  if (!data->output_buffer_block_count) {
    REXAPU_ERROR("XmaContext {}: Error - Received 0 for output_buffer_block_count!", id());
    return;
  }

  if (data->input_buffer_read_offset < kBitsPerPacketHeader) {
    data->input_buffer_read_offset = kBitsPerPacketHeader;
  }

  const uint32_t current_input_size = GetCurrentInputBufferSize(data);
  const uint32_t current_input_packet_count = current_input_size / kBytesPerPacket;

  const int16_t packet_index = GetPacketNumber(current_input_size, data->input_buffer_read_offset);

  if (packet_index == -1) {
    REXAPU_ERROR("XmaContext {}: Invalid packet index. Input read offset: {}", id(),
                 static_cast<uint32_t>(data->input_buffer_read_offset));
    return;
  }

  uint8_t* packet = current_input_buffer + (packet_index * kBytesPerPacket);
  const uint32_t packet_first_frame_offset = xma::GetPacketFrameOffset(packet);
  uint32_t relative_offset = data->input_buffer_read_offset % kBitsPerPacket;

  if (relative_offset < packet_first_frame_offset) {
    data->input_buffer_read_offset = (packet_index * kBitsPerPacket) + packet_first_frame_offset;
    relative_offset = packet_first_frame_offset;
  }

  const uint8_t skip_count = xma::GetPacketSkipCount(packet);

  // Full packet skip (0xFF) -- no new frames begin in this packet.
  if (skip_count == 0xFF) {
    uint32_t next_input_offset =
        GetNextPacketReadOffset(current_input_buffer, packet_index + 1, current_input_packet_count);
    if (next_input_offset == kBitsPerPacketHeader) {
      SwapInputBuffer(data);
    }
    data->input_buffer_read_offset = next_input_offset;
    return;
  }

  kPacketInfo packet_info = GetPacketInfo(packet, relative_offset);
  const uint32_t packet_to_skip = skip_count + 1;
  const uint32_t next_packet_index = packet_index + packet_to_skip;

  // [NARUTO-XMA-PKT] one header line per input buffer: what the packet CLAIMS
  // (XMA2 header frame count) vs what the frame-chain walk found (merged into
  // packet_info.frame_count_). The ~7.7s music gap suspect is a packet whose
  // header claims ~5x the frames we actually decode.
  if (!probe_pkt_hdr_logged_ && REXCVAR_GET(apu_xma_probe)) {
    probe_pkt_hdr_logged_ = true;
    REXAPU_INFO(
        "[nrxma] ctx={:03d} PKT xma2={} hdr_frames={} merged_frames={} first_off={} skip={} "
        "cur_frame={} rdoff={}",
        id(), xma::IsPacketXma2Type(packet) ? 1 : 0,
        static_cast<uint32_t>(xma::GetPacketFrameCount(packet)),
        static_cast<uint32_t>(packet_info.frame_count_), packet_first_frame_offset,
        static_cast<uint32_t>(skip_count), static_cast<uint32_t>(packet_info.current_frame_),
        static_cast<uint32_t>(data->input_buffer_read_offset));
  }

  // Frame header split across packet boundary.
  if (packet_info.current_frame_size_ == 0) {
    const uint8_t* next_packet = GetNextPacket(data, next_packet_index, current_input_packet_count);
    if (!next_packet) {
      SwapInputBuffer(data);
      return;
    }
    std::memcpy(input_buffer_.data(), packet + kBytesPerPacketHeader, kBytesPerPacketData);
    std::memcpy(input_buffer_.data() + kBytesPerPacketData, next_packet + kBytesPerPacketHeader,
                kBytesPerPacketData);

    BitStream combined(input_buffer_.data(), (kBitsPerPacket - kBitsPerPacketHeader) * 2);
    combined.SetOffset(relative_offset - kBitsPerPacketHeader);

    uint64_t frame_size = combined.Peek(kBitsPerFrameHeader);
    if (frame_size == xma::kMaxFrameLength) {
      data->error_status = 4;
      // [NARUTO-XMA-PROBE]
      if (REXCVAR_GET(apu_xma_probe)) {
        REXAPU_INFO("[nrxma] ctx={:03d} ERROR_STATUS=4 (split-frame end marker) rdoff={}", id(),
                    static_cast<uint32_t>(data->input_buffer_read_offset));
      }
      return;
    }
    packet_info.current_frame_size_ = static_cast<uint32_t>(frame_size);
  }

  BitStream stream(current_input_buffer, (packet_index + 1) * kBitsPerPacket);
  stream.SetOffset(data->input_buffer_read_offset);

  const uint64_t bits_to_copy = GetAmountOfBitsToRead(static_cast<uint32_t>(stream.BitsRemaining()),
                                                      packet_info.current_frame_size_);

  if (bits_to_copy == 0) {
    REXAPU_ERROR("XmaContext {}: There are no bits to copy!", id());
    SwapInputBuffer(data);
    return;
  }

  if (packet_info.isLastFrameInPacket()) {
    if (stream.BitsRemaining() < packet_info.current_frame_size_) {
      const uint8_t* next_packet =
          GetNextPacket(data, next_packet_index, current_input_packet_count);
      if (!next_packet) {
        // [NARUTO-XMA-SPLIT] Hardware behavior: the exhausted buffer is
        // consumed NOW (cache the partial frame, release the buffer so the
        // game's packet-completion chain keeps flowing); the frame completes
        // when the next buffer arrives. Holding the buffer + error_status=4
        // (post-rewrite Xenia, also current upstream) stalls the game's
        // feeder at a stream's final loop-end-truncated packet forever.
        if (REXCVAR_GET(apu_xma_split_frame_cache)) {
          split_pending_ = true;
          split_starve_kicks_ = 0;  // [NARUTO-XMA-SILENCE]
          stock_suppressed_ = 0;    // [NARUTO-XMA-STOCK] new episode: log afresh
          // [NARUTO-XMA-SEAM] A loop seam repeats at the same cumulative input
          // position every iteration. The first seam is detected by the
          // starvation timeout and its position learned; later seams are
          // predicted exactly and silence emission starts immediately (same
          // Work pass), so the guest's render pipeline never drains: gapless.
          episode_input_bits_ = stream_base_bits_ + data->input_buffer_read_offset;
          // [NARUTO-XMA-LOOPEND] PRIMARY (first-ever encounter, table-free):
          // the game's loop_end is the loop point in OUTPUT PCM bytes; when our
          // cumulative decoded output reaches it, this starvation is the seam.
          const uint64_t per_frame_out_bytes =
              uint64_t(kSamplesPerFrame) * (data->is_stereo ? 2u : 1u) * kBytesPerSample;
          const uint64_t loop_end_out_bytes = data->loop_end;
          const bool out_domain_seam =
              REXCVAR_GET(apu_xma_loopend_predict) && loop_end_out_bytes > 0x1000 &&
              decoded_output_bytes_ + per_frame_out_bytes * 8 >= loop_end_out_bytes;
          // FALLBACK: learned/cached input-domain length (loop_end unset/zero).
          // (floor re-checked so stale poisoned cache entries can never fire)
          const bool learned_seam =
              loop_input_len_bits_ >= kSeamMinLoopInputBits &&
              episode_input_bits_ + kBitsPerPacket >= last_seam_input_bits_ + loop_input_len_bits_;
          seam_predicted_ =
              REXCVAR_GET(apu_xma_starved_silence) && (out_domain_seam || learned_seam);
          if (seam_predicted_ && REXCVAR_GET(apu_xma_probe)) {
            REXAPU_INFO(
                "[nrxma] ctx={:03d} SEAM PREDICTED ({}) out_bytes={} loop_end={} input_bits={} - "
                "instant silence",
                id(), out_domain_seam ? "loop_end" : "learned", decoded_output_bytes_,
                loop_end_out_bytes, episode_input_bits_);
          }
          split_frame_size_bits_ = packet_info.current_frame_size_;
          split_frame_offset_bits_ = relative_offset;
          std::memcpy(split_packet_payload_.data(), packet + kBytesPerPacketHeader,
                      kBytesPerPacketData);
          if (REXCVAR_GET(apu_xma_probe)) {
            REXAPU_INFO(
                "[nrxma] ctx={:03d} SPLIT-CACHE: frame at rdoff={} (size={} bits) spans past "
                "buffer end; buffer released, frame completes on next buffer",
                id(), static_cast<uint32_t>(data->input_buffer_read_offset),
                split_frame_size_bits_);
          }
          SwapInputBuffer(data);
          return;
        }
        // [NARUTO-XMA-STARVE] The current frame spans into a buffer the game
        // has not supplied. Normally the game supplies it within ~100ms of
        // kicks and we decode the split frame then. But at END OF STREAM no
        // next buffer will EVER come, while the game waits for this buffer to
        // be consumed before looping the track — a deadlock that permanently
        // kills long streamed sounds (minigame music). After far more stuck
        // kicks than any healthy refill wait (measured <=121, threshold 512),
        // consume the exhausted buffer and clear the error so the game's
        // loop-restart logic proceeds. The dropped partial frame is the
        // track's final ~10ms, which real hardware also never rendered.
        if (REXCVAR_GET(apu_xma_starved_swap)) {
          const uint32_t rdoff = data->input_buffer_read_offset;
          const uint8_t cur = static_cast<uint8_t>(data->current_buffer);
          if (starve_run_count_ && starve_run_rdoff_ == rdoff && starve_run_cur_ == cur) {
            starve_run_count_++;
          } else {
            starve_run_count_ = 1;
            starve_run_rdoff_ = rdoff;
            starve_run_cur_ = cur;
          }
          if (starve_run_count_ >= 512) {
            REXAPU_INFO(
                "[nrxma] ctx={:03d} STARVED-SWAP: end-of-stream deadlock broken after {} stuck "
                "kicks (rdoff={} cur={}) - consuming exhausted buffer so the game can loop",
                id(), starve_run_count_, rdoff, static_cast<uint32_t>(cur));
            starve_run_count_ = 0;
            data->error_status = 0;
            SwapInputBuffer(data);
            return;
          }
        }
        data->error_status = 4;
        // [NARUTO-XMA-PROBE]
        if (REXCVAR_GET(apu_xma_probe)) {
          REXAPU_INFO("[nrxma] ctx={:03d} ERROR_STATUS=4 (frame spans into missing next buffer) "
                      "rdoff={} cur={}",
                      id(), static_cast<uint32_t>(data->input_buffer_read_offset),
                      static_cast<uint32_t>(data->current_buffer));
        }
        return;
      }
      std::memcpy(input_buffer_.data() + kBytesPerPacketData, next_packet + kBytesPerPacketHeader,
                  kBytesPerPacketData);
    }
  }

  std::memcpy(input_buffer_.data(), packet + kBytesPerPacketHeader, kBytesPerPacketData);

  stream = BitStream(input_buffer_.data(), (kBitsPerPacket - kBitsPerPacketHeader) * 2);
  stream.SetOffset(relative_offset - kBitsPerPacketHeader);

  xma_frame_.fill(0);

  const uint32_t padding_start =
      static_cast<uint8_t>(stream.Copy(xma_frame_.data() + 1, packet_info.current_frame_size_));

  raw_frame_.fill(0);

  PrepareDecoder(data->sample_rate, bool(data->is_stereo));
  PreparePacket(packet_info.current_frame_size_, padding_start);
  if (DecodePacket(av_context_, av_packet_, av_frame_)) {
    // [NARUTO-XMA-PROBE]
    probe_decodes_.fetch_add(1, std::memory_order_relaxed);
    probe_pkt_frames_++;  // [NARUTO-XMA-PKT]
    pace_frames_++;       // [NARUTO-XMA-PACE]
    probe_was_streaming_ = true;
    starve_run_count_ = 0;  // [NARUTO-XMA-STARVE] stream is progressing
    decoded_output_bytes_ +=  // [NARUTO-XMA-LOOPEND]
        uint64_t(kSamplesPerFrame) * (data->is_stereo ? 2u : 1u) * kBytesPerSample;

    ConvertFrame(reinterpret_cast<const uint8_t**>(&av_frame_->data), bool(data->is_stereo),
                 raw_frame_.data());
    current_frame_is_silence_ = false;  // [NARUTO-XMA-STOCK] real PCM: ledger it
    current_frame_remaining_subframes_ = 4 << data->is_stereo;

    // Loop end: limit output to subframes 0..loop_subframe_end.
    if (is_loop_end_frame) {
      loop_frame_output_limit_ = (data->loop_subframe_end + 1) << data->is_stereo;
    } else {
      loop_frame_output_limit_ = 0;
    }

    // Loop start: skip leading subframes per loop_subframe_skip.
    if (loop_start_skip_pending_) {
      const uint8_t skip = data->loop_subframe_skip << data->is_stereo;
      if (skip < current_frame_remaining_subframes_) {
        current_frame_remaining_subframes_ -= skip;
      }
      loop_start_skip_pending_ = false;
    }
  }

  // Compute where to go next.
  if (!packet_info.isLastFrameInPacket()) {
    const uint32_t next_frame_offset =
        (data->input_buffer_read_offset + bits_to_copy) % kBitsPerPacket;
    data->input_buffer_read_offset = (packet_index * kBitsPerPacket) + next_frame_offset;
    return;
  }

  uint32_t next_input_offset =
      GetNextPacketReadOffset(current_input_buffer, next_packet_index, current_input_packet_count);

  if (next_input_offset == kBitsPerPacketHeader) {
    SwapInputBuffer(data);
    if (data->IsAnyInputBufferValid()) {
      next_input_offset = xma::GetPacketFrameOffset(
          memory()->TranslatePhysical(data->GetCurrentInputBufferAddress()));

      if (next_input_offset > kMaxFrameSizeinBits) {
        SwapInputBuffer(data);
        return;
      }
    }
  }
  data->input_buffer_read_offset = next_input_offset;
}

// [NARUTO-XMA-SPLIT] Complete a frame whose head was cached when its buffer
// was retired: rebuild the same combined bitstream the normal split path uses
// ([old packet payload][new first packet payload]) and decode across the seam.
void XmaContext::ResumeSplitFrame(XMA_CONTEXT_DATA* data, uint8_t* current_input_buffer) {
  split_pending_ = false;
  split_starve_kicks_ = 0;  // [NARUTO-XMA-SILENCE]
  stock_suppressed_ = 0;    // [NARUTO-XMA-STOCK]

  // [NARUTO-XMA-SEAM] At a predicted loop seam the arriving buffer is the
  // loop-START data, not the cached frame's continuation - decoding the
  // splice would produce a garbage frame (silence already stood in for it).
  // Skip straight to the new buffer's first frame and restart the loop-
  // iteration sample counter.
  if (seam_predicted_) {
    seam_predicted_ = false;
    last_seam_input_bits_ = episode_input_bits_;  // keep loop phase for the next prediction
    decoded_output_bytes_ = 0;  // [NARUTO-XMA-LOOPEND] new loop iteration
    if (REXCVAR_GET(apu_xma_probe)) {
      REXAPU_INFO("[nrxma] ctx={:03d} SEAM WRAP COMPLETE: loop data arrived, resuming cleanly",
                  id());
    }
    uint32_t first_offset = xma::GetPacketFrameOffset(current_input_buffer);
    if (first_offset > kMaxFrameSizeinBits) {
      first_offset = GetNextPacketReadOffset(current_input_buffer, 1,
                                             data->GetCurrentInputBufferPacketCount());
      if (first_offset == kBitsPerPacketHeader) {
        SwapInputBuffer(data);
        return;
      }
    }
    data->input_buffer_read_offset = first_offset;
    return;
  }

  std::memcpy(input_buffer_.data(), split_packet_payload_.data(), kBytesPerPacketData);
  std::memcpy(input_buffer_.data() + kBytesPerPacketData,
              current_input_buffer + kBytesPerPacketHeader, kBytesPerPacketData);

  BitStream stream(input_buffer_.data(), (kBitsPerPacket - kBitsPerPacketHeader) * 2);
  stream.SetOffset(split_frame_offset_bits_ - kBitsPerPacketHeader);

  xma_frame_.fill(0);
  const uint32_t padding_start =
      static_cast<uint8_t>(stream.Copy(xma_frame_.data() + 1, split_frame_size_bits_));
  raw_frame_.fill(0);

  PrepareDecoder(data->sample_rate, bool(data->is_stereo));
  PreparePacket(split_frame_size_bits_, padding_start);
  if (DecodePacket(av_context_, av_packet_, av_frame_)) {
    probe_decodes_.fetch_add(1, std::memory_order_relaxed);
    probe_pkt_frames_++;  // [NARUTO-XMA-PKT] (split resume counts toward the NEW buffer)
    pace_frames_++;       // [NARUTO-XMA-PACE]
    probe_was_streaming_ = true;
    starve_run_count_ = 0;
    decoded_output_bytes_ +=  // [NARUTO-XMA-LOOPEND]
        uint64_t(kSamplesPerFrame) * (data->is_stereo ? 2u : 1u) * kBytesPerSample;
    ConvertFrame(reinterpret_cast<const uint8_t**>(&av_frame_->data), bool(data->is_stereo),
                 raw_frame_.data());
    current_frame_is_silence_ = false;  // [NARUTO-XMA-STOCK] real PCM: ledger it
    current_frame_remaining_subframes_ = 4 << data->is_stereo;
    loop_frame_output_limit_ = 0;
    if (loop_start_skip_pending_) {
      const uint8_t skip = data->loop_subframe_skip << data->is_stereo;
      if (skip < current_frame_remaining_subframes_) {
        current_frame_remaining_subframes_ -= skip;
      }
      loop_start_skip_pending_ = false;
    }
  }

  // Continue at the first NEW frame of this buffer's first packet (the bits
  // before it were the cached frame's continuation).
  uint32_t next_input_offset = xma::GetPacketFrameOffset(current_input_buffer);
  if (next_input_offset > kMaxFrameSizeinBits) {
    // No new frame begins in packet 0; look at the following packets.
    next_input_offset = GetNextPacketReadOffset(current_input_buffer, 1,
                                                data->GetCurrentInputBufferPacketCount());
    if (next_input_offset == kBitsPerPacketHeader) {
      SwapInputBuffer(data);
      return;
    }
  }
  data->input_buffer_read_offset = next_input_offset;
}

void XmaContext::ConvertFrame(const uint8_t** samples, bool is_two_channel,
                              uint8_t* output_buffer) {
  // Loop through every sample, convert and drop it into the output array.
  // If more than one channel, we need to interleave the samples from each
  // channel next to each other. Always saturate because FFmpeg output is
  // not limited to [-1, 1] (for example 1.095 as seen in 5454082B).
  constexpr float scale = (1 << 15) - 1;
  auto out = reinterpret_cast<int16_t*>(output_buffer);

  // For testing of vectorized versions, stereo audio is common in 4D5307E6,
  // since the first menu frame; the intro cutscene also has more than 2
  // channels.
#if REX_ARCH_AMD64
  static_assert(kSamplesPerFrame % 8 == 0);
  const auto in_channel_0 = reinterpret_cast<const float*>(samples[0]);
  const __m128 scale_mm = _mm_set1_ps(scale);
  if (is_two_channel) {
    const auto in_channel_1 = reinterpret_cast<const float*>(samples[1]);
    const __m128i shufmask = _mm_set_epi8(14, 15, 6, 7, 12, 13, 4, 5, 10, 11, 2, 3, 8, 9, 0, 1);
    for (uint32_t i = 0; i < kSamplesPerFrame; i += 4) {
      // Load 8 samples, 4 for each channel.
      __m128 in_mm0 = _mm_loadu_ps(&in_channel_0[i]);
      __m128 in_mm1 = _mm_loadu_ps(&in_channel_1[i]);
      // Rescale.
      in_mm0 = _mm_mul_ps(in_mm0, scale_mm);
      in_mm1 = _mm_mul_ps(in_mm1, scale_mm);
      // Cast to int32.
      __m128i out_mm0 = _mm_cvtps_epi32(in_mm0);
      __m128i out_mm1 = _mm_cvtps_epi32(in_mm1);
      // Saturated cast and pack to int16.
      __m128i out_mm = _mm_packs_epi32(out_mm0, out_mm1);
      // Interleave channels and byte swap.
      out_mm = _mm_shuffle_epi8(out_mm, shufmask);
      // Store, as [out + i * 4] movdqu.
      _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i * 2]), out_mm);
    }
  } else {
    const __m128i shufmask = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    for (uint32_t i = 0; i < kSamplesPerFrame; i += 8) {
      // Load 8 samples, as [in_channel_0 + i * 4] and
      // [in_channel_0 + i * 4 + 16] movups.
      __m128 in_mm0 = _mm_loadu_ps(&in_channel_0[i]);
      __m128 in_mm1 = _mm_loadu_ps(&in_channel_0[i + 4]);
      // Rescale.
      in_mm0 = _mm_mul_ps(in_mm0, scale_mm);
      in_mm1 = _mm_mul_ps(in_mm1, scale_mm);
      // Cast to int32.
      __m128i out_mm0 = _mm_cvtps_epi32(in_mm0);
      __m128i out_mm1 = _mm_cvtps_epi32(in_mm1);
      // Saturated cast and pack to int16.
      __m128i out_mm = _mm_packs_epi32(out_mm0, out_mm1);
      // Byte swap.
      out_mm = _mm_shuffle_epi8(out_mm, shufmask);
      // Store, as [out + i * 2] movdqu.
      _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), out_mm);
    }
  }
#else
  uint32_t o = 0;
  for (uint32_t i = 0; i < kSamplesPerFrame; i++) {
    for (uint32_t j = 0; j <= uint32_t(is_two_channel); j++) {
      // Select the appropriate array based on the current channel.
      auto in = reinterpret_cast<const float*>(samples[j]);

      // Raw samples sometimes aren't within [-1, 1]
      float scaled_sample = rex::clamp_float(in[i], -1.0f, 1.0f) * scale;

      // Convert the sample and output it in big endian.
      auto sample = static_cast<int16_t>(scaled_sample);
      out[o++] = rex::byte_swap(sample);
    }
  }
#endif
}

}  // namespace rex::audio
