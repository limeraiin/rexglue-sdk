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

// Disable warnings about unused parameters for kernel functions
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <chrono>
#include <cstring>
#include <mutex>

#include <rex/assert.h>
#include <rex/audio/audio_system.h>
#include <rex/audio/flags.h>
#include <rex/audio/xma/decoder.h>
#include <rex/kernel/xboxkrnl/private.h>
#include <rex/logging.h>
#include <rex/hook.h>
#include <rex/types.h>
#include <rex/runtime.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xtypes.h>

namespace rex::kernel::xboxkrnl {
using namespace rex::system;

using rex::audio::XMA_CONTEXT_DATA;

// See audio_system.cc for implementation details.
//
// XMA details:
// https://devel.nuclex.org/external/svn/directx/trunk/include/xma2defs.h
// https://github.com/gdawg/fsbext/blob/master/src/xma_header.h
//
// XMA is undocumented, but the methods are pretty simple.
// Games do this sequence to decode (now):
//   (not sure we are setting buffer validity/offsets right)
// d> XMACreateContext(20656800)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMADisableContext(000103E0, 0)
// d> XMABlockWhileInUse(000103E0)
// d> XMAInitializeContext(000103E0, 20008810)
// d> XMASetOutputBufferValid(000103E0)
// d> XMASetInputBuffer0Valid(000103E0)
// d> XMAEnableContext(000103E0)
// d> XMAGetOutputBufferWriteOffset(000103E0)
// d> XMAGetOutputBufferReadOffset(000103E0)
// d> XMAIsOutputBufferValid(000103E0)
// d> XMAGetOutputBufferReadOffset(000103E0)
// d> XMAGetOutputBufferWriteOffset(000103E0)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMAIsInputBuffer0Valid(000103E0)
// d> XMAIsInputBuffer1Valid(000103E0)
// d> XMAReleaseContext(000103E0)
//
// XAudio2 uses XMA under the covers, and seems to map with the same
// restrictions of frame/subframe/etc:
// https://msdn.microsoft.com/en-us/library/windows/desktop/microsoft.directx_sdk.xaudio2.xaudio2_buffer(v=vs.85).aspx

// [NARUTO-XMA-PROBE] helpers ------------------------------------------------
static audio::XmaDecoder* NrXmaDecoder() {
  return static_cast<audio::AudioSystem*>(REX_KERNEL_STATE()->emulator()->audio_system())
      ->xma_decoder();
}

static int NrXmaId(uint32_t context_guest_ptr) {
  return NrXmaDecoder()->GetContextId(context_guest_ptr);
}

// When apu_xma_locked_ops is on, serializes the guest's whole-context
// read-modify-store against the decode worker's Work() (which holds the same
// mutex for its whole snapshot->decode->merged-store span). Without this, a
// guest XMASet* landing mid-decode can be overwritten by the worker's store
// (or overwrite the worker's just-advanced offsets with its stale snapshot) —
// a lost update whose probability grows with sound length.
class NrXmaCtxGuard {
 public:
  explicit NrXmaCtxGuard(uint32_t context_guest_ptr) {
    if (REXCVAR_GET(apu_xma_locked_ops)) {
      auto* decoder = NrXmaDecoder();
      auto* context = decoder->GetContextById(decoder->GetContextId(context_guest_ptr));
      if (context) {
        lock_ = std::unique_lock<std::mutex>(context->work_lock());
      }
    }
  }

 private:
  std::unique_lock<std::mutex> lock_;
};

#define NRXMA_LOG(...)                 \
  do {                                 \
    if (REXCVAR_GET(apu_xma_probe)) {  \
      REXAPU_INFO("[nrxma] " __VA_ARGS__); \
    }                                  \
  } while (0)

// Guest-poll visibility: count what the game reads/writes per context.
#define NRXMA_COUNT(guest_ptr, field)                                              \
  do {                                                                             \
    if (REXCVAR_GET(apu_xma_probe)) {                                              \
      auto* nrxma_dec = NrXmaDecoder();                                            \
      auto* nrxma_ctx = nrxma_dec->GetContextById(nrxma_dec->GetContextId(guest_ptr)); \
      if (nrxma_ctx) {                                                             \
        nrxma_ctx->field.fetch_add(1, std::memory_order_relaxed);                  \
      }                                                                            \
    }                                                                              \
  } while (0)
// ---------------------------------------------------------------------------

u32 XMACreateContext_entry(mapped_u32 context_out_ptr) {
  REXKRNL_NOISY_DEBUG("XMACreateContext called!");
  auto xma_decoder =
      static_cast<audio::AudioSystem*>(REX_KERNEL_STATE()->emulator()->audio_system())
          ->xma_decoder();
  uint32_t context_ptr = xma_decoder->AllocateContext();
  *context_out_ptr = context_ptr;
  if (!context_ptr) {
    NRXMA_LOG("CreateContext FAILED (pool exhausted)");
    return X_STATUS_NO_MEMORY;
  }
  NRXMA_LOG("ctx={:03d} CREATE", NrXmaId(context_ptr));
  return X_STATUS_SUCCESS;
}

u32 XMAReleaseContext_entry(mapped_void context_ptr) {
  auto xma_decoder =
      static_cast<audio::AudioSystem*>(REX_KERNEL_STATE()->emulator()->audio_system())
          ->xma_decoder();
  NRXMA_LOG("ctx={:03d} RELEASE", NrXmaId(context_ptr.guest_address()));
  xma_decoder->ReleaseContext(context_ptr.guest_address());
  return 0;
}

void StoreXmaContextIndexedRegister(system::KernelState* kernel_state, uint32_t base_reg,
                                    uint32_t context_ptr) {
  uint32_t context_physical_address = REX_KERNEL_MEMORY()->GetPhysicalAddress(context_ptr);
  assert_true(context_physical_address != UINT32_MAX);
  auto xma_decoder =
      static_cast<audio::AudioSystem*>(kernel_state->emulator()->audio_system())->xma_decoder();
  uint32_t hw_index =
      (context_physical_address - xma_decoder->context_array_ptr()) / sizeof(XMA_CONTEXT_DATA);
  uint32_t reg_num = base_reg + (hw_index >> 5) * 4;
  uint32_t reg_value = 1 << (hw_index & 0x1F);
  xma_decoder->WriteRegister(reg_num, rex::byte_swap(reg_value));
}

struct XMA_LOOP_DATA {
  rex::be<uint32_t> loop_start;
  rex::be<uint32_t> loop_end;
  uint8_t loop_count;
  uint8_t loop_subframe_end;
  uint8_t loop_subframe_skip;
};
static_assert_size(XMA_LOOP_DATA, 12);

struct XMA_CONTEXT_INIT {
  rex::be<uint32_t> input_buffer_0_ptr;
  rex::be<uint32_t> input_buffer_0_packet_count;
  rex::be<uint32_t> input_buffer_1_ptr;
  rex::be<uint32_t> input_buffer_1_packet_count;
  rex::be<uint32_t> input_buffer_read_offset;
  rex::be<uint32_t> output_buffer_ptr;
  rex::be<uint32_t> output_buffer_block_count;
  rex::be<uint32_t> work_buffer;
  rex::be<uint32_t> subframe_decode_count;
  rex::be<uint32_t> channel_count;
  rex::be<uint32_t> sample_rate;
  XMA_LOOP_DATA loop_data;
};
static_assert_size(XMA_CONTEXT_INIT, 56);

u32 XMAInitializeContext_entry(mapped_void context_ptr, ppc_ptr_t<XMA_CONTEXT_INIT> context_init) {
  // Input buffers may be null (buffer 1 in 415607D4).
  // Convert to host endianness.
  uint32_t input_buffer_0_guest_ptr = context_init->input_buffer_0_ptr;
  uint32_t input_buffer_0_physical_address = 0;
  if (input_buffer_0_guest_ptr) {
    input_buffer_0_physical_address =
        REX_KERNEL_MEMORY()->GetPhysicalAddress(input_buffer_0_guest_ptr);
    // Xenia-specific safety check.
    assert_true(input_buffer_0_physical_address != UINT32_MAX);
    if (input_buffer_0_physical_address == UINT32_MAX) {
      REXKRNL_ERROR("XMAInitializeContext: Invalid input buffer 0 virtual address {:08X}",
                    input_buffer_0_guest_ptr);
      return X_E_FALSE;
    }
  }
  uint32_t input_buffer_1_guest_ptr = context_init->input_buffer_1_ptr;
  uint32_t input_buffer_1_physical_address = 0;
  if (input_buffer_1_guest_ptr) {
    input_buffer_1_physical_address =
        REX_KERNEL_MEMORY()->GetPhysicalAddress(input_buffer_1_guest_ptr);
    assert_true(input_buffer_1_physical_address != UINT32_MAX);
    if (input_buffer_1_physical_address == UINT32_MAX) {
      REXKRNL_ERROR("XMAInitializeContext: Invalid input buffer 1 virtual address {:08X}",
                    input_buffer_1_guest_ptr);
      return X_E_FALSE;
    }
  }
  uint32_t output_buffer_guest_ptr = context_init->output_buffer_ptr;
  assert_not_zero(output_buffer_guest_ptr);
  uint32_t output_buffer_physical_address =
      REX_KERNEL_MEMORY()->GetPhysicalAddress(output_buffer_guest_ptr);
  assert_true(output_buffer_physical_address != UINT32_MAX);
  if (output_buffer_physical_address == UINT32_MAX) {
    REXKRNL_ERROR("XMAInitializeContext: Invalid output buffer virtual address {:08X}",
                  output_buffer_guest_ptr);
    return X_E_FALSE;
  }

  NrXmaCtxGuard guard(context_ptr.guest_address());  // [NARUTO-XMA-PROBE]
  std::memset(context_ptr, 0, sizeof(XMA_CONTEXT_DATA));

  XMA_CONTEXT_DATA context(context_ptr);

  context.input_buffer_0_ptr = input_buffer_0_physical_address;
  context.input_buffer_0_packet_count = context_init->input_buffer_0_packet_count;
  context.input_buffer_1_ptr = input_buffer_1_physical_address;
  context.input_buffer_1_packet_count = context_init->input_buffer_1_packet_count;
  context.input_buffer_read_offset = context_init->input_buffer_read_offset;
  context.output_buffer_ptr = output_buffer_physical_address;
  context.output_buffer_block_count = context_init->output_buffer_block_count;

  // context.work_buffer = context_init->work_buffer;  // ?
  context.subframe_decode_count = context_init->subframe_decode_count;
  context.is_stereo = context_init->channel_count >= 1;
  context.sample_rate = context_init->sample_rate;

  context.loop_start = context_init->loop_data.loop_start;
  context.loop_end = context_init->loop_data.loop_end;
  context.loop_count = context_init->loop_data.loop_count;
  context.loop_subframe_end = context_init->loop_data.loop_subframe_end;
  context.loop_subframe_skip = context_init->loop_data.loop_subframe_skip;

  context.Store(context_ptr);

  // [NARUTO-XMA-CUMUL] a (re)initialized context starts a new stream.
  {
    auto* decoder = NrXmaDecoder();
    auto* xma_context = decoder->GetContextById(decoder->GetContextId(context_ptr.guest_address()));
    if (xma_context) {
      xma_context->ResetStreamState();
    }
  }

  NRXMA_LOG(
      "ctx={:03d} INIT in0={:08X}({} pkts) in1={:08X}({} pkts) rdoff={} outblk={} sdc={} "
      "stereo={} rate={} loop(s={} e={} n={})",
      NrXmaId(context_ptr.guest_address()), static_cast<uint32_t>(context.input_buffer_0_ptr),
      static_cast<uint32_t>(context.input_buffer_0_packet_count),
      static_cast<uint32_t>(context.input_buffer_1_ptr),
      static_cast<uint32_t>(context.input_buffer_1_packet_count),
      static_cast<uint32_t>(context.input_buffer_read_offset),
      static_cast<uint32_t>(context.output_buffer_block_count),
      static_cast<uint32_t>(context.subframe_decode_count),
      static_cast<uint32_t>(context.is_stereo), static_cast<uint32_t>(context.sample_rate),
      static_cast<uint32_t>(context.loop_start), static_cast<uint32_t>(context.loop_end),
      static_cast<uint32_t>(context.loop_count));

  StoreXmaContextIndexedRegister(REX_KERNEL_STATE(), 0x1A80, context_ptr.guest_address());

  return 0;
}

u32 XMASetLoopData_entry(mapped_void context_ptr, ppc_ptr_t<XMA_CONTEXT_DATA> loop_data) {
  NrXmaCtxGuard guard(context_ptr.guest_address());  // [NARUTO-XMA-PROBE]
  XMA_CONTEXT_DATA context(context_ptr);

  context.loop_start = loop_data->loop_start;
  context.loop_end = loop_data->loop_end;
  context.loop_count = loop_data->loop_count;
  context.loop_subframe_end = loop_data->loop_subframe_end;
  context.loop_subframe_skip = loop_data->loop_subframe_skip;

  context.Store(context_ptr);

  NRXMA_LOG("ctx={:03d} SET_LOOP s={} e={} n={}", NrXmaId(context_ptr.guest_address()),
            static_cast<uint32_t>(context.loop_start), static_cast<uint32_t>(context.loop_end),
            static_cast<uint32_t>(context.loop_count));

  return 0;
}

u32 XMAGetInputBufferReadOffset_entry(mapped_void context_ptr) {
  NRXMA_COUNT(context_ptr.guest_address(), probe_poll_inrd_);
  XMA_CONTEXT_DATA context(context_ptr);
  return context.input_buffer_read_offset;
}

u32 XMASetInputBufferReadOffset_entry(mapped_void context_ptr, u32 value) {
  NrXmaCtxGuard guard(context_ptr.guest_address());  // [NARUTO-XMA-PROBE]
  XMA_CONTEXT_DATA context(context_ptr);
  context.input_buffer_read_offset = value;
  context.Store(context_ptr);

  NRXMA_LOG("ctx={:03d} SET_RDOFF {}", NrXmaId(context_ptr.guest_address()), value);

  return 0;
}

u32 XMASetInputBuffer0_entry(mapped_void context_ptr, mapped_void buffer, u32 packet_count) {
  uint32_t buffer_physical_address =
      REX_KERNEL_MEMORY()->GetPhysicalAddress(buffer.guest_address());
  assert_true(buffer_physical_address != UINT32_MAX);
  if (buffer_physical_address == UINT32_MAX) {
    // Xenia-specific safety check.
    REXKRNL_ERROR("XMASetInputBuffer0: Invalid buffer virtual address {:08X}",
                  buffer.guest_address());
    return X_E_FALSE;
  }

  NrXmaCtxGuard guard(context_ptr.guest_address());  // [NARUTO-XMA-PROBE]
  XMA_CONTEXT_DATA context(context_ptr);

  context.input_buffer_0_ptr = buffer_physical_address;
  context.input_buffer_0_packet_count = packet_count;

  context.Store(context_ptr);

  NRXMA_LOG("ctx={:03d} SET_IN0 buf={:08X} pkts={}", NrXmaId(context_ptr.guest_address()),
            buffer_physical_address, packet_count);

  return 0;
}

u32 XMAIsInputBuffer0Valid_entry(mapped_void context_ptr) {
  NRXMA_COUNT(context_ptr.guest_address(), probe_poll_in0_);
  XMA_CONTEXT_DATA context(context_ptr);
  return context.input_buffer_0_valid;
}

u32 XMASetInputBuffer0Valid_entry(mapped_void context_ptr) {
  NrXmaCtxGuard guard(context_ptr.guest_address());  // [NARUTO-XMA-PROBE]
  XMA_CONTEXT_DATA context(context_ptr);
  context.input_buffer_0_valid = 1;
  context.Store(context_ptr);

  NRXMA_LOG("ctx={:03d} SET_IN0_VALID (in1={} cur={} rdoff={})",
            NrXmaId(context_ptr.guest_address()),
            static_cast<uint32_t>(context.input_buffer_1_valid),
            static_cast<uint32_t>(context.current_buffer),
            static_cast<uint32_t>(context.input_buffer_read_offset));

  return 0;
}

u32 XMASetInputBuffer1_entry(mapped_void context_ptr, mapped_void buffer, u32 packet_count) {
  uint32_t buffer_physical_address =
      REX_KERNEL_MEMORY()->GetPhysicalAddress(buffer.guest_address());
  assert_true(buffer_physical_address != UINT32_MAX);
  if (buffer_physical_address == UINT32_MAX) {
    // Xenia-specific safety check.
    REXKRNL_ERROR("XMASetInputBuffer1: Invalid buffer virtual address {:08X}",
                  buffer.guest_address());
    return X_E_FALSE;
  }

  NrXmaCtxGuard guard(context_ptr.guest_address());  // [NARUTO-XMA-PROBE]
  XMA_CONTEXT_DATA context(context_ptr);

  context.input_buffer_1_ptr = buffer_physical_address;
  context.input_buffer_1_packet_count = packet_count;

  context.Store(context_ptr);

  NRXMA_LOG("ctx={:03d} SET_IN1 buf={:08X} pkts={}", NrXmaId(context_ptr.guest_address()),
            buffer_physical_address, packet_count);

  return 0;
}

u32 XMAIsInputBuffer1Valid_entry(mapped_void context_ptr) {
  NRXMA_COUNT(context_ptr.guest_address(), probe_poll_in1_);
  XMA_CONTEXT_DATA context(context_ptr);
  return context.input_buffer_1_valid;
}

u32 XMASetInputBuffer1Valid_entry(mapped_void context_ptr) {
  NrXmaCtxGuard guard(context_ptr.guest_address());  // [NARUTO-XMA-PROBE]
  XMA_CONTEXT_DATA context(context_ptr);
  context.input_buffer_1_valid = 1;
  context.Store(context_ptr);

  NRXMA_LOG("ctx={:03d} SET_IN1_VALID (in0={} cur={} rdoff={})",
            NrXmaId(context_ptr.guest_address()),
            static_cast<uint32_t>(context.input_buffer_0_valid),
            static_cast<uint32_t>(context.current_buffer),
            static_cast<uint32_t>(context.input_buffer_read_offset));

  return 0;
}

u32 XMAIsOutputBufferValid_entry(mapped_void context_ptr) {
  NRXMA_COUNT(context_ptr.guest_address(), probe_poll_outval_);
  XMA_CONTEXT_DATA context(context_ptr);
  return context.output_buffer_valid;
}

u32 XMASetOutputBufferValid_entry(mapped_void context_ptr) {
  NRXMA_COUNT(context_ptr.guest_address(), probe_set_outval_);
  NrXmaCtxGuard guard(context_ptr.guest_address());  // [NARUTO-XMA-PROBE]
  XMA_CONTEXT_DATA context(context_ptr);
  context.output_buffer_valid = 1;
  context.Store(context_ptr);

  return 0;
}

u32 XMAGetOutputBufferReadOffset_entry(mapped_void context_ptr) {
  NRXMA_COUNT(context_ptr.guest_address(), probe_poll_outrd_);
  XMA_CONTEXT_DATA context(context_ptr);
  return context.output_buffer_read_offset;
}

u32 XMASetOutputBufferReadOffset_entry(mapped_void context_ptr, u32 value) {
  NRXMA_COUNT(context_ptr.guest_address(), probe_set_outrd_);
  // [NARUTO-XMA-PROBE] guard but no log: this is the guest's per-tick output
  // drain pointer, far too hot to log per call.
  NrXmaCtxGuard guard(context_ptr.guest_address());
  XMA_CONTEXT_DATA context(context_ptr);
  context.output_buffer_read_offset = value;
  context.Store(context_ptr);

  return 0;
}

u32 XMAGetOutputBufferWriteOffset_entry(mapped_void context_ptr) {
  NRXMA_COUNT(context_ptr.guest_address(), probe_poll_outwr_);
  XMA_CONTEXT_DATA context(context_ptr);
  return context.output_buffer_write_offset;
}

u32 XMAGetPacketMetadata_entry(mapped_void context_ptr) {
  XMA_CONTEXT_DATA context(context_ptr);
  return context.packet_metadata;
}

u32 XMAEnableContext_entry(mapped_void context_ptr) {
  NRXMA_LOG("ctx={:03d} ENABLE", NrXmaId(context_ptr.guest_address()));
  StoreXmaContextIndexedRegister(REX_KERNEL_STATE(), 0x1940, context_ptr.guest_address());
  return 0;
}

u32 XMADisableContext_entry(mapped_void context_ptr, u32 wait) {
  X_HRESULT result = X_E_SUCCESS;
  NRXMA_LOG("ctx={:03d} DISABLE wait={}", NrXmaId(context_ptr.guest_address()), wait);
  StoreXmaContextIndexedRegister(REX_KERNEL_STATE(), 0x1A40, context_ptr.guest_address());
  if (!static_cast<audio::AudioSystem*>(REX_KERNEL_STATE()->emulator()->audio_system())
           ->xma_decoder()
           ->BlockOnContext(context_ptr.guest_address(), !wait)) {
    result = X_E_FALSE;
  }
  return result;
}

u32 XMABlockWhileInUse_entry(mapped_void context_ptr) {
  // [NARUTO-XMA-PROBE] Time the wait: a guest audio thread stuck here stops
  // ALL of that thread's sound servicing.
  const auto block_start = std::chrono::steady_clock::now();
  do {
    XMA_CONTEXT_DATA context(context_ptr);
    if (!context.input_buffer_0_valid && !context.input_buffer_1_valid) {
      break;
    }
    if (!context.work_buffer_ptr) {
      break;
    }
    rex::thread::Sleep(std::chrono::milliseconds(1));
  } while (true);
  if (REXCVAR_GET(apu_xma_probe)) {
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - block_start)
                                .count();
    if (elapsed_ms > 50) {
      REXAPU_INFO("[nrxma] ctx={:03d} BlockWhileInUse took {}ms",
                  NrXmaId(context_ptr.guest_address()), elapsed_ms);
    }
  }
  return 0;
}

}  // namespace rex::kernel::xboxkrnl

REX_EXPORT(__imp__XMACreateContext, rex::kernel::xboxkrnl::XMACreateContext_entry)
REX_EXPORT(__imp__XMAReleaseContext, rex::kernel::xboxkrnl::XMAReleaseContext_entry)
REX_EXPORT(__imp__XMAInitializeContext, rex::kernel::xboxkrnl::XMAInitializeContext_entry)
REX_EXPORT(__imp__XMASetLoopData, rex::kernel::xboxkrnl::XMASetLoopData_entry)
REX_EXPORT(__imp__XMAGetInputBufferReadOffset,
           rex::kernel::xboxkrnl::XMAGetInputBufferReadOffset_entry)
REX_EXPORT(__imp__XMASetInputBufferReadOffset,
           rex::kernel::xboxkrnl::XMASetInputBufferReadOffset_entry)
REX_EXPORT(__imp__XMASetInputBuffer0, rex::kernel::xboxkrnl::XMASetInputBuffer0_entry)
REX_EXPORT(__imp__XMAIsInputBuffer0Valid, rex::kernel::xboxkrnl::XMAIsInputBuffer0Valid_entry)
REX_EXPORT(__imp__XMASetInputBuffer0Valid, rex::kernel::xboxkrnl::XMASetInputBuffer0Valid_entry)
REX_EXPORT(__imp__XMASetInputBuffer1, rex::kernel::xboxkrnl::XMASetInputBuffer1_entry)
REX_EXPORT(__imp__XMAIsInputBuffer1Valid, rex::kernel::xboxkrnl::XMAIsInputBuffer1Valid_entry)
REX_EXPORT(__imp__XMASetInputBuffer1Valid, rex::kernel::xboxkrnl::XMASetInputBuffer1Valid_entry)
REX_EXPORT(__imp__XMAIsOutputBufferValid, rex::kernel::xboxkrnl::XMAIsOutputBufferValid_entry)
REX_EXPORT(__imp__XMASetOutputBufferValid, rex::kernel::xboxkrnl::XMASetOutputBufferValid_entry)
REX_EXPORT(__imp__XMAGetOutputBufferReadOffset,
           rex::kernel::xboxkrnl::XMAGetOutputBufferReadOffset_entry)
REX_EXPORT(__imp__XMASetOutputBufferReadOffset,
           rex::kernel::xboxkrnl::XMASetOutputBufferReadOffset_entry)
REX_EXPORT(__imp__XMAGetOutputBufferWriteOffset,
           rex::kernel::xboxkrnl::XMAGetOutputBufferWriteOffset_entry)
REX_EXPORT(__imp__XMAGetPacketMetadata, rex::kernel::xboxkrnl::XMAGetPacketMetadata_entry)
REX_EXPORT(__imp__XMAEnableContext, rex::kernel::xboxkrnl::XMAEnableContext_entry)
REX_EXPORT(__imp__XMADisableContext, rex::kernel::xboxkrnl::XMADisableContext_entry)
REX_EXPORT(__imp__XMABlockWhileInUse, rex::kernel::xboxkrnl::XMABlockWhileInUse_entry)
