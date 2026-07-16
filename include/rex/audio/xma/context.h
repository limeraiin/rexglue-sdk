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

#pragma once

#include <array>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>

#include <rex/kernel.h>
#include <rex/memory.h>
#include <rex/thread.h>

// XMA audio format:
// From research, XMA appears to be based on WMA Pro with
// a few (very slight) modifications.
// XMA2 is fully backwards-compatible with XMA1.

// Helpful resources:
// https://github.com/koolkdev/libertyv/blob/master/libav_wrapper/xma2dec.c
// https://hcs64.com/mboard/forum.php?showthread=14818
// https://github.com/hrydgard/minidx9/blob/master/Include/xma2defs.h

// Forward declarations
struct AVCodec;
struct AVCodecParserContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;

namespace rex::audio {

// This is stored in guest space in big-endian order.
// We load and swap the whole thing to splat here so that we can
// use bitfields.
// This could be important:
// https://www.fmod.org/questions/question/forum-15859
// Appears to be dumped in order (for the most part)

struct XMA_CONTEXT_DATA {
  // DWORD 0
  uint32_t input_buffer_0_packet_count : 12;  // XMASetInputBuffer0, number of
                                              // 2KB packets. Max 4095 packets.
                                              // These packets form a block.
  uint32_t loop_count : 8;                    // +12bit, XMASetLoopData NumLoops
  uint32_t input_buffer_0_valid : 1;          // +20bit, XMAIsInputBuffer0Valid
  uint32_t input_buffer_1_valid : 1;          // +21bit, XMAIsInputBuffer1Valid
  uint32_t output_buffer_block_count : 5;     // +22bit SizeWrite 256byte blocks
  uint32_t output_buffer_write_offset : 5;    // +27bit
                                              // XMAGetOutputBufferWriteOffset
                                              // AKA OffsetWrite

  // DWORD 1
  uint32_t input_buffer_1_packet_count : 12;  // XMASetInputBuffer1, number of
                                              // 2KB packets. Max 4095 packets.
                                              // These packets form a block.
  uint32_t loop_subframe_start : 2;           // +12bit, XMASetLoopData
  uint32_t loop_subframe_end : 3;             // +14bit, XMASetLoopData
  uint32_t loop_subframe_skip : 3;            // +17bit, XMASetLoopData might be
                                              // subframe_decode_count
  uint32_t subframe_decode_count : 4;         // +20bit
  uint32_t output_buffer_padding : 3;         // +24bit, extra output buffer blocks
                                              // reserved per decoded frame
  uint32_t sample_rate : 2;                   // +27bit enum of sample rates
  uint32_t is_stereo : 1;                     // +29bit
  uint32_t unk_dword_1_c : 1;                 // +30bit
  uint32_t output_buffer_valid : 1;           // +31bit, XMAIsOutputBufferValid

  // DWORD 2
  uint32_t input_buffer_read_offset : 26;  // XMAGetInputBufferReadOffset
  uint32_t error_status : 5;               // ErrorStatus
  uint32_t error_set : 1;                  // ErrorSet

  // DWORD 3
  uint32_t loop_start : 26;          // XMASetLoopData LoopStartOffset
                                     // frame offset in bits
  uint32_t parser_error_status : 5;  // ParserErrorStatus
  uint32_t parser_error_set : 1;     // ParserErrorSet

  // DWORD 4
  uint32_t loop_end : 26;        // XMASetLoopData LoopEndOffset
                                 // frame offset in bits
  uint32_t packet_metadata : 5;  // XMAGetPacketMetadata
  uint32_t current_buffer : 1;   // ?

  // DWORD 5
  uint32_t input_buffer_0_ptr;  // physical address
  // DWORD 6
  uint32_t input_buffer_1_ptr;  // physical address
  // DWORD 7
  uint32_t output_buffer_ptr;  // physical address
  // DWORD 8
  uint32_t work_buffer_ptr;  // PtrOverlapAdd(?)

  // DWORD 9
  // +0bit, XMAGetOutputBufferReadOffset AKA WriteBufferOffsetRead
  uint32_t output_buffer_read_offset : 5;
  uint32_t : 25;
  uint32_t stop_when_done : 1;       // +30bit
  uint32_t interrupt_when_done : 1;  // +31bit

  // DWORD 10-15
  uint32_t unk_dwords_10_15[6];  // reserved?

  explicit XMA_CONTEXT_DATA(const void* ptr) {
    memory::copy_and_swap(reinterpret_cast<uint32_t*>(this), reinterpret_cast<const uint32_t*>(ptr),
                          sizeof(XMA_CONTEXT_DATA) / 4);
  }

  void Store(void* ptr) {
    memory::copy_and_swap(reinterpret_cast<uint32_t*>(ptr), reinterpret_cast<const uint32_t*>(this),
                          sizeof(XMA_CONTEXT_DATA) / 4);
  }

  bool IsInputBufferValid(uint8_t buffer_index) const {
    return buffer_index == 0 ? input_buffer_0_valid : input_buffer_1_valid;
  }

  bool IsCurrentInputBufferValid() const { return IsInputBufferValid(current_buffer); }

  bool IsAnyInputBufferValid() const { return input_buffer_0_valid || input_buffer_1_valid; }

  uint32_t GetInputBufferAddress(uint8_t buffer_index) const {
    return buffer_index == 0 ? input_buffer_0_ptr : input_buffer_1_ptr;
  }

  uint32_t GetCurrentInputBufferAddress() const { return GetInputBufferAddress(current_buffer); }

  uint32_t GetInputBufferPacketCount(uint8_t buffer_index) const {
    return buffer_index == 0 ? input_buffer_0_packet_count : input_buffer_1_packet_count;
  }

  uint32_t GetCurrentInputBufferPacketCount() const {
    return GetInputBufferPacketCount(current_buffer);
  }

  bool IsConsumeOnlyContext() const {
    return (input_buffer_0_packet_count | input_buffer_1_packet_count) == 0;
  }
};
static_assert_size(XMA_CONTEXT_DATA, 64);

#pragma pack(push, 1)
// XMA2WAVEFORMATEX
struct Xma2ExtraData {
  uint8_t raw[34];
};
static_assert_size(Xma2ExtraData, 34);
#pragma pack(pop)

struct kPacketInfo {
  uint8_t frame_count_ = 0;
  uint8_t current_frame_ = 0;
  uint32_t current_frame_size_ = 0;

  bool isLastFrameInPacket() const {
    return frame_count_ == 0 || current_frame_ == frame_count_ - 1;
  }
};

static constexpr int kIdToSampleRate[4] = {24000, 32000, 44100, 48000};

class XmaContext {
 public:
  static const uint32_t kBytesPerPacket = 2048;
  static const uint32_t kBitsPerPacket = kBytesPerPacket * 8;
  static const uint32_t kBitsPerPacketHeader = 32;
  static const uint32_t kBitsPerFrameHeader = 15;
  static const uint32_t kBytesPerPacketHeader = 4;
  static const uint32_t kBytesPerPacketData = kBytesPerPacket - kBytesPerPacketHeader;

  static const uint32_t kBytesPerSample = 2;
  static const uint32_t kSamplesPerFrame = 512;
  static const uint32_t kSamplesPerSubframe = 128;
  static const uint32_t kBytesPerFrameChannel = kSamplesPerFrame * kBytesPerSample;
  static const uint32_t kBytesPerSubframeChannel = kSamplesPerSubframe * kBytesPerSample;

  static const uint32_t kOutputBytesPerBlock = 256;
  static const uint32_t kOutputMaxSizeBytes = 31 * kOutputBytesPerBlock;
  static const uint32_t kMaxFrameSizeinBits = 0x4000 - kBitsPerPacketHeader;

  // [NARUTO-XMA-FIFO] host elastic decode-ahead buffer size (per context): 1536
  // output blocks = ~1s stereo @44.1k, comfortably above the producer's ~6-packet
  // lookahead incl. one 39-frame quiet packet (312 stereo blocks). The producer
  // gate caps how far ahead it feeds, so the FIFO fills only to the lookahead;
  // if it ever fills, the decode-ahead simply stalls (no corruption).
  static const uint32_t kFifoBytes = 1536 * kOutputBytesPerBlock;

  explicit XmaContext();
  ~XmaContext();

  int Setup(uint32_t id, memory::Memory* memory, uint32_t guest_ptr);
  bool Work();

  void Enable();
  bool Block(bool poll);
  void Clear();
  void Disable();
  void Release();

  memory::Memory* memory() const { return memory_; }

  uint32_t id() { return id_; }
  uint32_t guest_ptr() { return guest_ptr_; }
  bool is_allocated() { return is_allocated_.load(std::memory_order_acquire); }
  bool is_enabled() { return is_enabled_.load(std::memory_order_acquire); }

  void set_is_allocated(bool is_allocated) {
    is_allocated_.store(is_allocated, std::memory_order_release);
  }
  void set_is_enabled(bool is_enabled) { is_enabled_.store(is_enabled, std::memory_order_release); }

  void SignalWorkDone() {
    if (work_completion_event_) {
      work_completion_event_->Set();
    }
  }
  void WaitForWorkDone() {
    if (work_completion_event_) {
      rex::thread::Wait(work_completion_event_.get(), false);
    }
  }

  // [NARUTO-XMA-PROBE] Serializes guest-side context read-modify-store against
  // Work() (apu_xma_locked_ops) — this is the same mutex Work() holds.
  std::mutex& work_lock() { return lock_; }

  // [NARUTO-XMA-CUMUL] Called when the guest (re)initializes the context
  // (XMAInitializeContext); the new stream starts at cumulative offset 0.
  // Lock-free on purpose: the caller may already hold work_lock(), and INIT
  // only happens on a disabled/blocked context.
  void ResetStreamState() {
    stream_base_bits_ = 0;
    starve_run_count_ = 0;
    split_pending_ = false;     // [NARUTO-XMA-SPLIT]
    seam_predicted_ = false;    // [NARUTO-XMA-SEAM]
    loop_input_len_bits_ = 0;
    last_seam_input_bits_ = 0;
    tentative_loop_len_bits_ = 0;
    stream_key_valid_ = false;
    decoded_output_bytes_ = 0;  // [NARUTO-XMA-LOOPEND]
    ResetFifo();                // [NARUTO-XMA-FIFO] new stream: drop buffered PCM
    ResetStock();               // [NARUTO-XMA-STOCK] new stream: prior stock is gone
  }

  // [NARUTO-XMA-CUMUL] probe accessor.
  uint64_t stream_base_bits() const { return stream_base_bits_; }

  // [NARUTO-XMA-PROBE] Per-context counters, drained once/sec by
  // XmaDecoder::ProbeDumpStatus when apu_xma_probe is on.
  std::atomic<uint32_t> probe_decodes_{0};
  std::atomic<uint32_t> probe_swaps_{0};
  std::atomic<uint32_t> probe_starves_{0};
  std::atomic<uint32_t> probe_out_stalls_{0};
  // Guest polling visibility: what the game reads/writes while it waits.
  std::atomic<uint32_t> probe_poll_in0_{0};      // XMAIsInputBuffer0Valid
  std::atomic<uint32_t> probe_poll_in1_{0};      // XMAIsInputBuffer1Valid
  std::atomic<uint32_t> probe_poll_outwr_{0};    // XMAGetOutputBufferWriteOffset
  std::atomic<uint32_t> probe_poll_outrd_{0};    // XMAGetOutputBufferReadOffset
  std::atomic<uint32_t> probe_poll_outval_{0};   // XMAIsOutputBufferValid
  std::atomic<uint32_t> probe_poll_inrd_{0};     // XMAGetInputBufferReadOffset
  std::atomic<uint32_t> probe_set_outrd_{0};     // XMASetOutputBufferReadOffset
  std::atomic<uint32_t> probe_set_outval_{0};    // XMASetOutputBufferValid
  std::atomic<uint32_t> probe_kicks_{0};         // HW kick register writes (incl. MMIO-direct)

 private:
  void SwapInputBuffer(XMA_CONTEXT_DATA* data);
  static int GetSampleRate(int id);
  static int16_t GetPacketNumber(size_t size, size_t bit_offset);
  static uint32_t GetCurrentInputBufferSize(XMA_CONTEXT_DATA* data);

  kPacketInfo GetPacketInfo(uint8_t* packet, uint32_t frame_offset);
  uint32_t GetAmountOfBitsToRead(uint32_t remaining_stream_bits, uint32_t frame_size);
  const uint8_t* GetNextPacket(XMA_CONTEXT_DATA* data, uint32_t next_packet_index,
                               uint32_t current_input_packet_count);
  uint32_t GetNextPacketReadOffset(uint8_t* buffer, uint32_t next_packet_index,
                                   uint32_t current_input_packet_count);
  uint8_t* GetCurrentInputBuffer(XMA_CONTEXT_DATA* data);

  void Decode(XMA_CONTEXT_DATA* data);
  void Consume(memory::RingBuffer* output_rb, const XMA_CONTEXT_DATA* data);
  void UpdateLoopStatus(XMA_CONTEXT_DATA* data);
  void ClearLocked(XMA_CONTEXT_DATA* data);

  // [NARUTO-XMA-FIFO] Decode+Consume frames into `rb` until it is full or input
  // runs out (the body factored out of Work()). `skip_when_dry` breaks BEFORE
  // decoding an empty stream (so the elastic FIFO is never filled with silence);
  // false = the legacy path where a dry stream drips one silence/seam frame.
  void DecodeLoopInto(memory::RingBuffer* rb, XMA_CONTEXT_DATA* data, int32_t minimum_blocks,
                      bool skip_when_dry);
  // [NARUTO-XMA-FIFO] elastic decode-ahead: decode the producer's full lookahead
  // into fifo_, meter fifo_ -> the guest ring, then (only if fifo_ empty AND
  // input dry) drip via the legacy silence/seam path.
  void WorkElasticFifo(XMA_CONTEXT_DATA* data, memory::RingBuffer* output_rb, uint32_t effective_sdc);
  bool FifoEmpty() const { return fifo_read_ == fifo_write_; }
  void ResetFifo() { fifo_read_ = fifo_write_ = 0; }
  // [NARUTO-XMA-STOCK]
  void ResetStock() {
    stock_bytes_ = 0.0;
    stock_clock_valid_ = false;
    stock_suppressed_ = 0;
    current_frame_is_silence_ = false;
  }

  memory::RingBuffer PrepareOutputRingBuffer(XMA_CONTEXT_DATA* data);
  int PrepareDecoder(int sample_rate, bool is_two_channel);
  void PreparePacket(uint32_t frame_size, uint32_t frame_padding);
  bool DecodePacket(AVCodecContext* av_context, const AVPacket* av_packet, AVFrame* av_frame);

  void StoreContextMerged(const XMA_CONTEXT_DATA& data, const XMA_CONTEXT_DATA& initial_data,
                          uint8_t* context_ptr);

  static void ConvertFrame(const uint8_t** samples, bool is_two_channel, uint8_t* output_buffer);

  memory::Memory* memory_ = nullptr;
  std::unique_ptr<rex::thread::Event> work_completion_event_;

  uint32_t id_ = 0;
  uint32_t guest_ptr_ = 0;
  std::mutex lock_;
  std::atomic<bool> is_allocated_ = false;
  std::atomic<bool> is_enabled_ = false;

  // ffmpeg structures
  AVPacket* av_packet_ = nullptr;
  AVCodec* av_codec_ = nullptr;
  AVCodecContext* av_context_ = nullptr;
  AVFrame* av_frame_ = nullptr;

  // Packet data buffer (two packets worth for split frame handling)
  std::array<uint8_t, kBytesPerPacketData * 2> input_buffer_;
  // First byte contains bit offset information
  std::array<uint8_t, 1 + 4096> xma_frame_;
  // Conversion buffer for up to 2-channel frame
  std::array<uint8_t, kBytesPerFrameChannel * 2> raw_frame_;

  // Output buffer tracking
  int32_t remaining_subframe_blocks_in_output_buffer_ = 0;
  uint8_t current_frame_remaining_subframes_ = 0;

  // Loop subframe precision state
  uint8_t loop_frame_output_limit_ = 0;
  bool loop_start_skip_pending_ = false;

  // [NARUTO-XMA-PROBE] true once a decode succeeded; cleared when the stream
  // runs dry so each dry-out logs exactly once.
  bool probe_was_streaming_ = false;

  // [NARUTO-XMA-PKT] per-input-buffer decode census (mid-track music-gap
  // forensics): frames decoded from the current buffer + whether its first
  // packet's header was logged. Reset at SwapInputBuffer.
  uint32_t probe_pkt_frames_ = 0;
  bool probe_pkt_hdr_logged_ = false;

  // [NARUTO-XMA-PACE] real-time output pacing for streamed contexts: frames
  // released since pace_epoch_. NOT reset at loop wraps (output is continuous
  // across the seam); re-anchored when the stream falls behind schedule.
  bool pace_valid_ = false;
  std::chrono::steady_clock::time_point pace_epoch_;
  uint64_t pace_frames_ = 0;

  // [NARUTO-XMA-RINGTRACE] apu_xma_ring_trace throttle: last-emit timestamp per
  // context (us). High-rate ring occupancy trajectory (used/free blocks) for the
  // mid-track-gap stockpile-vs-jitter disambiguation (XMA-AUDIO-HANDOFF.md s3).
  int64_t ring_trace_last_us_ = 0;

  // [NARUTO-XMA-FIFO] host elastic decode-ahead buffer (apu_xma_elastic_fifo).
  // Lazily allocated to kFifoBytes on first streamed use; read/write offsets
  // persist across Work() calls. Reset on context reuse (Clear/Release/INIT),
  // NOT on loop wrap (the buffered PCM is continuous across the seam).
  std::vector<uint8_t> fifo_;
  uint32_t fifo_read_ = 0;
  uint32_t fifo_write_ = 0;

  // [NARUTO-XMA-STOCK] guest downstream-stock estimate (apu_xma_stock_starve):
  // real (non-injected) PCM bytes delivered to the output ring, drained at the
  // stream's render rate against wall time. While > floor, a starvation is the
  // feeder's BY-DESIGN idle (the +2-entry lookahead playing out downstream) and
  // silence injection would pollute the guest's stock => suppressed. Estimate
  // errors are safe in both directions: too low => today's behavior; too high
  // => drains to the floor within (error/rate) and injection resumes.
  bool current_frame_is_silence_ = false;
  double stock_bytes_ = 0.0;
  bool stock_clock_valid_ = false;
  std::chrono::steady_clock::time_point stock_last_;
  uint32_t stock_suppressed_ = 0;

  // [NARUTO-XMA-STARVE] consecutive decode attempts stuck at the same
  // spanning-frame-with-no-next-buffer state (see apu_xma_starved_swap).
  uint32_t starve_run_count_ = 0;
  uint32_t starve_run_rdoff_ = 0;
  uint8_t starve_run_cur_ = 0;

  // [NARUTO-XMA-CUMUL] bits of all fully-retired input buffers; guest-visible
  // read offset = stream_base_bits_ + intra-buffer offset (HW semantics).
  uint64_t stream_base_bits_ = 0;

  // [NARUTO-XMA-SPLIT] partial frame cached when a buffer was retired with
  // its last frame incomplete; completed from the next buffer's first packet.
  void ResumeSplitFrame(XMA_CONTEXT_DATA* data, uint8_t* current_input_buffer);
  bool split_pending_ = false;
  uint32_t split_frame_size_bits_ = 0;
  uint32_t split_frame_offset_bits_ = 0;  // frame start, bit offset within the saved packet
  std::array<uint8_t, kBytesPerPacketData> split_packet_payload_{};

  // [NARUTO-XMA-SILENCE] kicks spent starved at a cached split frame; drives
  // the silent-frame emission that unblocks the guest's render watermark.
  uint32_t split_starve_kicks_ = 0;

  // [NARUTO-XMA-SEAM] gapless loop-seam prediction: a loop seam repeats at
  // the same cumulative input position each iteration. The first seam is
  // detected by the starvation timeout and learned; later seams predict
  // exactly and silence emission starts immediately, so the guest's render
  // pipeline never drains.
  bool seam_predicted_ = false;
  uint64_t episode_input_bits_ = 0;
  uint64_t loop_input_len_bits_ = 0;
  uint64_t last_seam_input_bits_ = 0;
  // [NARUTO-XMA-SEAM] 2026-07-15 poisoning guards: a mid-track FEED STALL that
  // outlives the starve timeout is indistinguishable from a loop seam by the
  // timeout alone (a stall one packet into a track once "learned" a 14,556-bit
  // "loop" => a false SEAM PREDICTED + silence injection after EVERY packet =
  // fully stuttery music, poisoning xma_loop_cache.txt persistently). Guards:
  // (1) floor — no real streamed loop is shorter than kSeamMinLoopInputBits
  // (ground truth loop = 10,433,563 bits); (2) consistency — a length is only
  // trusted (used for prediction + persisted) after being observed twice in a
  // row, since a genuine loop repeats at the same period and random stalls
  // don't. Candidate lengths wait in tentative_loop_len_bits_.
  static constexpr uint64_t kSeamMinLoopInputBits = 1'000'000;
  uint64_t tentative_loop_len_bits_ = 0;
  // [NARUTO-XMA-LOOPEND] cumulative decoded OUTPUT bytes since the current loop
  // iteration began; compared against the game's loop_end (output-byte domain)
  // to predict the seam on the first-ever encounter. Reset to 0 at each wrap.
  uint64_t decoded_output_bytes_ = 0;
  // Persistent per-track loop-length cache: a track is identified by a hash
  // of its first input packet, so a learned loop length survives voice
  // teardown/re-entry and game restarts — first loops become gapless too.
  bool stream_key_valid_ = false;
  uint64_t stream_key_ = 0;
};

}  // namespace rex::audio
