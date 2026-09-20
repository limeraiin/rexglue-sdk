// Experimental native command ownership primitive. Not wired to guest hooks.
// A buffer owner retains a Lease even while dormant. At submission it retains
// another Lease into the queue BEFORE publishing the submission. Re-recording
// builds a new generation and replaces only the owner's lease; queued leases
// continue to own the old generation. Retirement drops the owner's lease.
//
// No address lookup, eviction, timeout, heap allocation or mutex is involved.
// Begin/build/commit require one producer. Distinct leases may be retained/read/
// released on different threads; transfer a lease through a synchronized queue.
// A single Lease object must not itself be concurrently modified. The carrier
// must outlive all builders and leases. Replay is synchronous; callbacks must
// finish using the supplied spans before returning.
//
// This primitive does NOT establish guest buffer ownership or patch coverage.
// Never suppress emission merely because Commit succeeded: the entire owning
// buffer, its patchers, submission publication and retirement must be hooked.
#ifndef REX_GRAPHICS_NR_COMMAND_CARRIER_H_
#define REX_GRAPHICS_NR_COMMAND_CARRIER_H_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

namespace rex::graphics::nr {

enum class NativeBoundary : uint32_t {
  kEvent, kFence, kWait, kLowRegister, kMirrorState, kShader,
  kNestedBuffer, kUnsupported,
};

// Fixed storage is provisioned outside guest hooks. Capacities cover complete
// admitted generations, not an evictable cache. Exhaustion refuses admission.
template <size_t Slots, size_t MaxCommands, size_t MaxWords>
class NativeCommandCarrier {
  static_assert(Slots && MaxCommands && MaxWords);
  static_assert(MaxCommands <= UINT32_MAX && MaxWords <= UINT32_MAX);
  static_assert(std::atomic<uint32_t>::is_always_lock_free);
  enum class Kind { kInline, kReference, kDraw, kBoundary };
  struct Command {
    Kind kind;
    uint32_t a, b, c, offset, count;
  };
  struct Slot {
    std::atomic<uint32_t> refs{0};
    uint64_t generation = 0;
    uint32_t commands = 0, words = 0;
    std::array<Command, MaxCommands> ops;
    std::array<uint32_t, MaxWords> data;
  };

 public:
  class Lease {
   public:
    Lease() = default;
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    // Explicit, fallible retain: check the result BEFORE exposing a submission.
    Lease Retain() const {
      if (!slot_) return {};
      auto refs = slot_->refs.load(std::memory_order_relaxed);
      do {
        if (refs == UINT32_MAX) return {};
      } while (!slot_->refs.compare_exchange_weak(
          refs, refs + 1, std::memory_order_relaxed));
      return Lease(slot_);
    }
    Lease(Lease&& other) noexcept
        : slot_(std::exchange(other.slot_, nullptr)) {}
    Lease& operator=(Lease&& other) noexcept {
      if (this != &other) {
        Reset();
        slot_ = std::exchange(other.slot_, nullptr);
      }
      return *this;
    }
    ~Lease() { Reset(); }
    void Reset() {
      if (slot_) {
        slot_->refs.fetch_sub(1, std::memory_order_acq_rel);
        slot_ = nullptr;
      }
    }
    explicit operator bool() const { return slot_ != nullptr; }
    uint64_t generation() const { return slot_ ? slot_->generation : 0; }

    // Sink owns register/memory semantics. References are resolved by the sink
    // at this precise point on EVERY replay. Boundaries are explicit and never
    // crossed by this carrier. A nested-buffer boundary must itself retain its
    // exact child generation in the submission integration; an address is not
    // sufficient. All callbacks return void; there is no partial-replay fallback.
    template <typename Sink>
    void Replay(Sink& sink) const {
      if (!slot_) return;
      for (uint32_t i = 0; i < slot_->commands; ++i) {
        const auto& op = slot_->ops[i];
        switch (op.kind) {
          case Kind::kInline:
            sink.Inline(op.a, std::span<const uint32_t>(
                slot_->data.data() + op.offset, op.count));
            break;
          case Kind::kReference: sink.Reference(op.a, op.b, op.c); break;
          case Kind::kDraw: sink.Draw(op.a, op.b, op.c); break;
          case Kind::kBoundary:
            sink.Boundary(static_cast<NativeBoundary>(op.a),
                          std::span<const uint32_t>(
                              slot_->data.data() + op.offset, op.count));
            break;
        }
      }
    }

   private:
    friend class NativeCommandCarrier;
    explicit Lease(Slot* slot) : slot_(slot) {}
    Slot* slot_ = nullptr;
  };

  class Builder {
   public:
    Builder() = default;
    Builder(const Builder&) = delete;
    Builder& operator=(const Builder&) = delete;
    Builder(Builder&& other) noexcept
        : lease_(std::move(other.lease_)), expected_(other.expected_),
          failed_(other.failed_) {}
    Builder& operator=(Builder&&) = delete;
    explicit operator bool() const { return bool(lease_) && !failed_; }

    // Values are host-order captures from shadows, never a decoded PM4 plan.
    bool Inline(uint32_t destination, std::span<const uint32_t> values) {
      if (values.empty() || values.size() > UINT32_MAX - destination) {
        failed_ = true; return false;
      }
      return Append({Kind::kInline, destination, 0, 0, 0, 0}, values);
    }
    bool Reference(uint32_t address, uint32_t destination, uint32_t count) {
      if (!count || (address & 3) || count > (UINT32_MAX - address) / 4 ||
          count > UINT32_MAX - destination) {
        failed_ = true; return false;
      }
      return Append({Kind::kReference, address, destination, count, 0, 0}, {});
    }
    // Ordinal is explicit; replacement does not renumber de-tile decisions.
    // Draw semantics beyond this minimal prototype must be added before wiring.
    bool Draw(uint32_t ordinal, uint32_t first, uint32_t count) {
      return Append({Kind::kDraw, ordinal, first, count, 0, 0}, {});
    }
    bool Boundary(NativeBoundary kind, std::span<const uint32_t> operands) {
      if (kind > NativeBoundary::kUnsupported) { failed_ = true; return false; }
      return Append({Kind::kBoundary, static_cast<uint32_t>(kind), 0, 0, 0, 0},
                    operands);
    }
    // No mutable pointer escapes. Failure poisons the WHOLE candidate and
    // releases it; original emission for this candidate must still be retained.
    Lease Commit() {
      if (!*this || lease_.slot_->commands != expected_) {
        lease_.Reset(); return {};
      }
      return std::move(lease_);
    }

   private:
    friend class NativeCommandCarrier;
    Builder(Slot* slot, uint32_t expected) : lease_(slot), expected_(expected) {}
    bool Append(Command op, std::span<const uint32_t> values) {
      if (!*this) return false;
      auto& slot = *lease_.slot_;
      if (slot.commands == expected_ || values.size() > MaxWords - slot.words) {
        failed_ = true; return false;
      }
      op.offset = slot.words;
      op.count = static_cast<uint32_t>(values.size());
      for (auto value : values) slot.data[slot.words++] = value;
      slot.ops[slot.commands++] = op;
      return true;
    }
    Lease lease_;
    uint32_t expected_ = 0;
    bool failed_ = false;
  };

  NativeCommandCarrier() = default;
  NativeCommandCarrier(const NativeCommandCarrier&) = delete;
  NativeCommandCarrier& operator=(const NativeCommandCarrier&) = delete;

  Builder Begin(uint32_t command_count) {
    if (!command_count || command_count > MaxCommands ||
        generation_ == std::numeric_limits<uint64_t>::max()) return {};
    for (auto& slot : slots_) {
      uint32_t free = 0;
      // Acquire pairs with the final release, so reuse never races a reader.
      if (!slot.refs.compare_exchange_strong(free, 1, std::memory_order_acquire))
        continue;
      slot.generation = ++generation_;
      slot.commands = slot.words = 0;
      return Builder(&slot, command_count);
    }
    return {};
  }

 private:
  std::array<Slot, Slots> slots_;
  uint64_t generation_ = 0;  // single producer; never wraps/reuses a generation
};

}  // namespace rex::graphics::nr
#endif  // REX_GRAPHICS_NR_COMMAND_CARRIER_H_
