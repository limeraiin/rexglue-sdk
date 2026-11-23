/**
 * @file        core/cpu_sampler.cpp
 * @brief       In-process sampling profiler (see rex/cpu_sampler.h)
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <rex/cpu_sampler.h>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>

REXCVAR_DEFINE_INT32(cpu_sample_hz, 0, "Perf",
                     "In-process CPU sampler rate in Hz (0 = off, e.g. 500). Writes "
                     "<log stem>.cpu-samples.txt beside the log file")
    .range(0, 4000);

REXCVAR_DEFINE_INT32(cpu_sample_seconds, 0, "Perf",
                     "Seconds of CPU sampling before the sample file is written (0 = until the "
                     "runtime shuts down)")
    .range(0, 86400);

REXCVAR_DEFINE_INT32(cpu_sample_delay, 0, "Perf",
                     "Seconds after the runtime starts before CPU sampling begins")
    .range(0, 86400);

#if REX_PLATFORM_WIN32 && defined(REX_ARCH_AMD64)

#include "platform_win.h"

// PSAPI_VERSION 2 maps the module calls to the K32* entry points of kernel32
// (no psapi.lib).
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2
#endif
#include <psapi.h>
#include <tlhelp32.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <string>
#include <vector>

#include <rex/filesystem.h>
#include <rex/thread.h>

#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

namespace rex::cpu_sampler {

namespace {

constexpr uint32_t kMaxModules = 512;
constexpr uint32_t kMaxThreads = 256;   // threads tracked at one time
constexpr uint32_t kMaxProfiles = 128;  // threads that ever got a sample
constexpr uint32_t kLeafSlots = 1u << 15;
constexpr uint32_t kPairSlots = 1u << 16;
constexpr uint32_t kMaxProbe = 64;
constexpr uint32_t kScanSlots = 1024;  // stack slots read above RSP (8 KB)
constexpr uint32_t kMaxCallers = 128;  // return addresses looked at per sample
constexpr uint32_t kIdleEvals = 4;     // activity checks without CPU before a thread is skipped
constexpr uint32_t kBlindEvals = 8;    // activity checks without any CPU before sampling everything
constexpr uint32_t kPairsWritten = 4096;
constexpr uint32_t kNamedThreads = 512;

// A code address is (module index + 1) in bits 48..62 and the RVA below. An
// address in no module keeps kNoModule and the address itself. Bit 63 marks
// the caller of an app-caller pair.
constexpr uint32_t kNoModule = 0x7FFE;
constexpr uint64_t kOffsetMask = (uint64_t(1) << 48) - 1;
constexpr uint64_t kAppCallerBit = uint64_t(1) << 63;

constexpr int32_t kNoProfile = -1;
constexpr int32_t kProfileDenied = -2;

uint64_t MakeKey(uint32_t module, uint64_t offset) {
  return (uint64_t(module + 1) << 48) | (offset & kOffsetMask);
}

uint32_t KeyModule(uint64_t key) {
  return uint32_t((key & ~kAppCallerBit) >> 48) - 1;
}

uint64_t KeyOffset(uint64_t key) {
  return key & kOffsetMask;
}

uint64_t Mix(uint64_t value) {
  value ^= value >> 33;
  value *= UINT64_C(0xFF51AFD7ED558CCD);
  value ^= value >> 33;
  value *= UINT64_C(0xC4CEB9FE1A85EC53);
  value ^= value >> 33;
  return value;
}

// ---------------------------------------------------------------------------
// Names given through the runtime's thread naming path, by system thread id.
// GetThreadDescription is asked first; this covers hosts where it is missing.
// ---------------------------------------------------------------------------

struct NamedThread {
  uint32_t id;
  char name[60];
};

SRWLOCK g_names_lock = SRWLOCK_INIT;
NamedThread g_names[kNamedThreads];
uint32_t g_names_next = 0;

bool LookupThreadName(uint32_t id, char* out, size_t capacity) {
  bool found = false;
  AcquireSRWLockShared(&g_names_lock);
  for (const NamedThread& named : g_names) {
    if (named.id == id) {
      std::snprintf(out, capacity, "%s", named.name);
      found = named.name[0] != 0;
      break;
    }
  }
  ReleaseSRWLockShared(&g_names_lock);
  return found;
}

// ---------------------------------------------------------------------------
// Reads that must not fault the sampler.
// ---------------------------------------------------------------------------

// The read is in a callee of the __try frame: clang unwinds SEH from calls
// only, not from a faulting instruction of the guarded function itself.
__declspec(noinline) void RawCopy(void* dst, const void* src, size_t size) {
  std::memcpy(dst, src, size);
}

bool GuardedCopy(void* dst, const void* src, size_t size) {
  __try {
    RawCopy(dst, src, size);
  } __except (EXCEPTION_EXECUTE_HANDLER) {  // NOLINT
    return false;
  }
  return true;
}

// p is at an FF opcode `length` bytes before a return address: true when it
// is a call r/m64 of exactly that length.
bool IsIndirectCall(const uint8_t* p, uint32_t length) {
  if (p[0] != 0xFF || ((p[1] >> 3) & 7) != 2) {
    return false;
  }
  const uint8_t mod = p[1] >> 6;
  const uint8_t rm = p[1] & 7;
  uint32_t expected = 2;
  if (mod != 3) {
    if (rm == 4) {
      ++expected;  // SIB
      if (mod == 0 && (p[2] & 7) == 5) {
        expected += 4;
      }
    }
    if (mod == 1) {
      expected += 1;
    } else if (mod == 2 || (mod == 0 && rm == 5)) {
      expected += 4;
    }
  }
  return expected == length;
}

size_t DirectoryLength(const wchar_t* path) {
  size_t length = 0;
  for (size_t i = 0; path[i]; ++i) {
    if (path[i] == L'\\' || path[i] == L'/') {
      length = i + 1;
    }
  }
  return length;
}

const char* BaseName(const char* path) {
  const char* name = path;
  for (const char* p = path; *p; ++p) {
    if (*p == '\\' || *p == '/') {
      name = p + 1;
    }
  }
  return name;
}

struct Module {
  uint64_t base = 0;
  uint64_t size = 0;
  uint64_t exec_lo = 0;  // RVAs spanning the executable sections
  uint64_t exec_hi = 0;
  bool app = false;  // in the folder of the executable or of the runtime
  bool live = false;
  char path[480] = {};
};

struct LeafEntry {
  uint64_t key;
  uint64_t count;
};

struct PairEntry {
  uint64_t leaf;
  uint64_t caller;
  uint64_t count;
};

constexpr size_t kProfileBytes = kLeafSlots * sizeof(LeafEntry) + kPairSlots * sizeof(PairEntry);

// Counts of one thread. Stays after the thread exits.
struct Profile {
  uint32_t thread_id = 0;
  char name[64] = {};
  uint64_t samples = 0;
  uint64_t chains = 0;    // samples with a return address above the leaf
  uint64_t no_stack = 0;  // samples whose stack could not be read
  uint64_t dropped_leaves = 0;
  uint64_t dropped_pairs = 0;
  uint64_t cpu_100ns = 0;  // CPU time over the window
  uint32_t leaf_used = 0;
  uint32_t pair_used = 0;
  LeafEntry* leaves = nullptr;
  PairEntry* pairs = nullptr;
};

struct Tracked {
  uint32_t id = 0;
  HANDLE handle = nullptr;  // null = free slot
  void* teb = nullptr;
  uint64_t cpu_start = 0;
  uint64_t cpu_last = 0;
  int32_t profile = kNoProfile;
  uint8_t idle_evals = 0;
  bool active = false;
  bool seen = false;
};

struct ThreadBasicInfo {
  LONG exit_status;
  void* teb;
  void* unique_process;
  void* unique_thread;
  ULONG_PTR affinity_mask;
  LONG priority;
  LONG base_priority;
};

using NtQueryInformationThreadFn = LONG(NTAPI*)(HANDLE, ULONG, void*, ULONG, ULONG*);
using NtDelayExecutionFn = LONG(NTAPI*)(BOOLEAN, LARGE_INTEGER*);
using GetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
using WineGetVersionFn = const char*(__cdecl*)();

class Sampler {
 public:
  bool Initialize(int hz, int seconds, int delay);
  void Shutdown();

 private:
  struct Address {
    int32_t module;  // -1 = none, the offset is the address
    uint64_t offset;
  };

  static DWORD WINAPI ThreadEntry(void* param);
  void Run();
  bool WaitForStart();
  void WaitUntil(int64_t target);
  int64_t Now() const;

  void RefreshModules();
  void AddModule(Module* module, HMODULE handle, uint64_t base, uint64_t size);
  const Module* FindModule(uint64_t address, uint32_t* index) const;
  bool IsReturnAddress(uint64_t address, const Module& module) const;
  Address Decode(uint64_t key) const;

  void RefreshThreads();
  Tracked* FindTracked(uint32_t id);
  void Track(uint32_t id);
  void Retire(Tracked* thread);
  void UpdateActivity();
  uint64_t ThreadCpu(HANDLE handle) const;
  void CloseOut(Tracked* thread);

  void SampleThreads();
  uint32_t CopyStack(const Tracked& thread, uint64_t rsp);
  void Record(Tracked* thread, uint64_t rip, uint32_t slot_count);
  Profile* ProfileFor(Tracked* thread);
  static void AddLeaf(Profile* profile, uint64_t key);
  static void AddPair(Profile* profile, uint64_t leaf, uint64_t caller);

  void Finish();
  void Dump();

  int hz_ = 0;
  int seconds_ = 0;
  int delay_ = 0;
  DWORD pid_ = 0;
  DWORD thread_id_ = 0;  // the sampler's own, set by it
  HANDLE thread_ = nullptr;
  HANDLE timer_ = nullptr;
  std::atomic<bool> stop_{false};
  std::filesystem::path path_;
  char wine_version_[32] = {};

  NtQueryInformationThreadFn nt_query_thread_ = nullptr;
  NtDelayExecutionFn nt_delay_ = nullptr;
  GetThreadDescriptionFn get_thread_description_ = nullptr;

  int64_t frequency_ = 1;
  int64_t start_ticks_ = 0;
  int64_t end_ticks_ = 0;
  uint64_t ticks_ = 0;

  wchar_t app_dirs_[2][MAX_PATH] = {};
  size_t app_dir_lengths_[2] = {};

  Module modules_[kMaxModules];
  uint32_t module_count_ = 0;
  uint16_t sorted_[kMaxModules] = {};  // live modules by base
  uint32_t sorted_count_ = 0;
  uint64_t image_lo_ = 0;
  uint64_t image_hi_ = 0;

  Tracked tracked_[kMaxThreads];
  uint32_t tracked_count_ = 0;
  uint32_t blind_evals_ = 0;
  bool sample_all_ = false;

  uint8_t* pool_ = nullptr;
  Profile profiles_[kMaxProfiles];
  uint32_t profile_count_ = 0;

  uint64_t slots_[kScanSlots] = {};

  uint64_t suspend_failed_ = 0;
  uint64_t context_failed_ = 0;
  uint64_t no_profile_ = 0;
  uint64_t threads_dropped_ = 0;
  uint64_t modules_dropped_ = 0;
  uint64_t snapshot_failed_ = 0;
};

bool Sampler::Initialize(int hz, int seconds, int delay) {
  hz_ = std::clamp(hz, 1, 4000);
  seconds_ = std::max(seconds, 0);
  delay_ = std::max(delay, 0);
  pid_ = GetCurrentProcessId();

  LARGE_INTEGER frequency;
  QueryPerformanceFrequency(&frequency);
  frequency_ = std::max<int64_t>(frequency.QuadPart, 1);

  if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll")) {
    nt_query_thread_ = reinterpret_cast<NtQueryInformationThreadFn>(
        reinterpret_cast<void*>(GetProcAddress(ntdll, "NtQueryInformationThread")));
    nt_delay_ = reinterpret_cast<NtDelayExecutionFn>(
        reinterpret_cast<void*>(GetProcAddress(ntdll, "NtDelayExecution")));
    auto wine_get_version = reinterpret_cast<WineGetVersionFn>(
        reinterpret_cast<void*>(GetProcAddress(ntdll, "wine_get_version")));
    if (wine_get_version) {
      const char* version = wine_get_version();
      std::snprintf(wine_version_, sizeof(wine_version_), "%s",
                    version && version[0] ? version : "unknown");
    }
  }
  if (HMODULE kernel = GetModuleHandleW(L"kernel32.dll")) {
    get_thread_description_ = reinterpret_cast<GetThreadDescriptionFn>(
        reinterpret_cast<void*>(GetProcAddress(kernel, "GetThreadDescription")));
  }

  // Modules in the folder of the executable or of the runtime are the app's.
  HMODULE owners[2] = {GetModuleHandleW(nullptr), nullptr};
  GetModuleHandleExW(
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCWSTR>(&g_names_lock), &owners[1]);
  for (int i = 0; i < 2; ++i) {
    if (!owners[i]) {
      continue;
    }
    DWORD length = GetModuleFileNameW(owners[i], app_dirs_[i], MAX_PATH);
    if (!length || length >= MAX_PATH) {
      app_dirs_[i][0] = 0;
      continue;
    }
    app_dir_lengths_[i] = DirectoryLength(app_dirs_[i]);
  }

  std::filesystem::path log = rex::CurrentLogFilePath();
  path_ = log.empty() ? std::filesystem::current_path() / fmt::format("cpu-samples-{}.txt", pid_)
                      : log.parent_path() / (log.stem().string() + ".cpu-samples.txt");

  // Reserved here, committed per thread at its first sample: a page commit,
  // not a heap allocation, and never while a thread is suspended.
  pool_ = static_cast<uint8_t*>(
      VirtualAlloc(nullptr, kMaxProfiles * kProfileBytes, MEM_RESERVE, PAGE_READWRITE));
  if (!pool_) {
    REXLOG_WARN("[cpu-sample] off: cannot reserve the sample tables (error {})", GetLastError());
    return false;
  }

  // Wine serves waitable timers from its server at millisecond precision;
  // its NtDelayExecution sleeps in the thread instead.
  if (!wine_version_[0]) {
    timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                    TIMER_ALL_ACCESS);
  }

  thread_ = CreateThread(nullptr, 512 * 1024, ThreadEntry, this, STACK_SIZE_PARAM_IS_A_RESERVATION,
                         nullptr);
  if (!thread_) {
    REXLOG_WARN("[cpu-sample] off: cannot create the sampler thread (error {})", GetLastError());
    Shutdown();
    return false;
  }
  REXLOG_INFO("[cpu-sample] on: {} Hz, starts in {} s, runs {}, file {}{}{}", hz_, delay_,
              seconds_ ? fmt::format("{} s", seconds_) : std::string("until shutdown"),
              rex::path_to_utf8(path_), wine_version_[0] ? ", wine " : "", wine_version_);
  return true;
}

void Sampler::Shutdown() {
  stop_.store(true, std::memory_order_relaxed);
  if (thread_) {
    WaitForSingleObject(thread_, INFINITE);
    CloseHandle(thread_);
    thread_ = nullptr;
  }
  if (timer_) {
    CloseHandle(timer_);
    timer_ = nullptr;
  }
  if (pool_) {
    VirtualFree(pool_, 0, MEM_RELEASE);
    pool_ = nullptr;
  }
}

DWORD WINAPI Sampler::ThreadEntry(void* param) {
  static_cast<Sampler*>(param)->Run();
  return 0;
}

int64_t Sampler::Now() const {
  LARGE_INTEGER now;
  QueryPerformanceCounter(&now);
  return now.QuadPart;
}

bool Sampler::WaitForStart() {
  const int64_t target = Now() + int64_t(delay_) * frequency_;
  while (Now() < target) {
    if (stop_.load(std::memory_order_relaxed)) {
      return false;
    }
    ::Sleep(50);
  }
  return !stop_.load(std::memory_order_relaxed);
}

void Sampler::WaitUntil(int64_t target) {
  while (!stop_.load(std::memory_order_relaxed)) {
    const int64_t remaining = (target - Now()) * 10000000 / frequency_;  // 100 ns units
    if (remaining < 500) {
      return;
    }
    LARGE_INTEGER interval;
    interval.QuadPart = -remaining;
    if (timer_ && SetWaitableTimer(timer_, &interval, 0, nullptr, nullptr, FALSE)) {
      WaitForSingleObject(timer_, 1000);
    } else if (nt_delay_) {
      nt_delay_(FALSE, &interval);
    } else {
      ::Sleep(remaining >= 10000 ? DWORD(remaining / 10000) : 1);
    }
  }
}

void Sampler::Run() {
  thread_id_ = GetCurrentThreadId();
  rex::thread::set_current_thread_name("CPU Sampler");
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
  if (!WaitForStart()) {
    REXLOG_INFO("[cpu-sample] stopped before sampling began, no file written");
    return;
  }

  RefreshModules();
  RefreshThreads();
  start_ticks_ = Now();
  const int64_t period = std::max<int64_t>(frequency_ / hz_, 1);
  const int64_t limit = seconds_ > 0 ? start_ticks_ + int64_t(seconds_) * frequency_ : INT64_MAX;
  int64_t next_tick = start_ticks_ + period;
  int64_t next_activity = start_ticks_ + frequency_ / 4;
  int64_t next_refresh = start_ticks_ + frequency_;
  uint32_t refreshes = 0;

  while (!stop_.load(std::memory_order_relaxed)) {
    WaitUntil(next_tick);
    int64_t now = Now();
    if (stop_.load(std::memory_order_relaxed) || now >= limit) {
      break;
    }
    if (now >= next_refresh) {
      RefreshThreads();
      if ((++refreshes & 3) == 0) {
        RefreshModules();
      }
      next_refresh = Now() + frequency_;
    }
    if (now >= next_activity) {
      UpdateActivity();
      next_activity = Now() + frequency_ / 4;
    }
    SampleThreads();
    ++ticks_;
    // A late tick is not made up for, and the threads always get a quarter
    // period to themselves.
    next_tick = std::max(next_tick + period, Now() + period / 4);
  }
  end_ticks_ = Now();

  Finish();
  Dump();
}

// ---------------------------------------------------------------------------
// Modules
// ---------------------------------------------------------------------------

void Sampler::RefreshModules() {
  HMODULE handles[kMaxModules];
  DWORD needed = 0;
  const HANDLE process = GetCurrentProcess();
  if (!EnumProcessModules(process, handles, sizeof(handles), &needed)) {
    return;
  }
  const uint32_t count = std::min<uint32_t>(needed / sizeof(HMODULE), kMaxModules);
  for (uint32_t i = 0; i < module_count_; ++i) {
    modules_[i].live = false;
  }
  for (uint32_t i = 0; i < count; ++i) {
    MODULEINFO info;
    if (!GetModuleInformation(process, handles[i], &info, sizeof(info))) {
      continue;
    }
    const uint64_t base = reinterpret_cast<uintptr_t>(info.lpBaseOfDll);
    const uint64_t size = info.SizeOfImage;
    if (!base || !size) {
      continue;
    }
    Module* module = nullptr;
    for (uint32_t j = 0; j < module_count_; ++j) {
      if (modules_[j].base == base && modules_[j].size == size) {
        module = &modules_[j];
        break;
      }
    }
    if (!module) {
      if (module_count_ == kMaxModules) {
        ++modules_dropped_;
        continue;
      }
      module = &modules_[module_count_++];
      AddModule(module, handles[i], base, size);
    }
    module->live = true;
  }

  sorted_count_ = 0;
  image_lo_ = 0;
  image_hi_ = 0;
  for (uint32_t i = 0; i < module_count_; ++i) {
    if (modules_[i].live) {
      sorted_[sorted_count_++] = uint16_t(i);
      image_hi_ = std::max(image_hi_, modules_[i].base + modules_[i].size);
    }
  }
  std::sort(sorted_, sorted_ + sorted_count_,
            [this](uint16_t a, uint16_t b) { return modules_[a].base < modules_[b].base; });
  if (sorted_count_) {
    image_lo_ = modules_[sorted_[0]].base;
  }
}

void Sampler::AddModule(Module* module, HMODULE handle, uint64_t base, uint64_t size) {
  module->base = base;
  module->size = size;
  module->exec_lo = 0;
  module->exec_hi = size;
  module->app = false;

  wchar_t wide[1024];
  DWORD length = GetModuleFileNameW(handle, wide, DWORD(std::size(wide)));
  if (length >= std::size(wide)) {
    length = 0;
  }
  wide[length] = 0;
  if (!length || !WideCharToMultiByte(CP_UTF8, 0, wide, -1, module->path, sizeof(module->path),
                                      nullptr, nullptr)) {
    std::snprintf(module->path, sizeof(module->path), "module_%llx",
                  static_cast<unsigned long long>(base));
  }
  const size_t directory = DirectoryLength(wide);
  for (int i = 0; i < 2; ++i) {
    if (directory && directory == app_dir_lengths_[i] &&
        !_wcsnicmp(wide, app_dirs_[i], directory)) {
      module->app = true;
    }
  }

  // Return addresses are looked for in the executable sections only: the
  // stack is full of pointers into an image's data.
  const uint8_t* image = reinterpret_cast<const uint8_t*>(base);
  IMAGE_DOS_HEADER dos;
  if (!GuardedCopy(&dos, image, sizeof(dos)) || dos.e_magic != IMAGE_DOS_SIGNATURE ||
      dos.e_lfanew <= 0 || uint64_t(dos.e_lfanew) + sizeof(IMAGE_NT_HEADERS64) > size) {
    return;
  }
  IMAGE_NT_HEADERS64 nt;
  if (!GuardedCopy(&nt, image + dos.e_lfanew, sizeof(nt)) || nt.Signature != IMAGE_NT_SIGNATURE) {
    return;
  }
  const uint64_t table = uint64_t(dos.e_lfanew) + offsetof(IMAGE_NT_HEADERS64, OptionalHeader) +
                         nt.FileHeader.SizeOfOptionalHeader;
  const uint32_t sections = std::min<uint32_t>(nt.FileHeader.NumberOfSections, 96);
  uint64_t lo = UINT64_MAX;
  uint64_t hi = 0;
  for (uint32_t i = 0; i < sections; ++i) {
    IMAGE_SECTION_HEADER section;
    const uint64_t offset = table + uint64_t(i) * sizeof(section);
    if (offset + sizeof(section) > size ||
        !GuardedCopy(&section, image + offset, sizeof(section))) {
      break;
    }
    if (!(section.Characteristics & IMAGE_SCN_MEM_EXECUTE)) {
      continue;
    }
    const uint64_t begin = section.VirtualAddress;
    const uint64_t end =
        begin + std::max<uint32_t>(section.Misc.VirtualSize, section.SizeOfRawData);
    lo = std::min(lo, begin);
    hi = std::max(hi, std::min(end, size));
  }
  if (lo < hi) {
    module->exec_lo = lo;
    module->exec_hi = hi;
  }
}

const Module* Sampler::FindModule(uint64_t address, uint32_t* index) const {
  if (address < image_lo_ || address >= image_hi_) {
    return nullptr;
  }
  uint32_t lo = 0;
  uint32_t hi = sorted_count_;
  while (lo < hi) {
    const uint32_t mid = (lo + hi) / 2;
    if (modules_[sorted_[mid]].base <= address) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (!lo) {
    return nullptr;
  }
  const Module& module = modules_[sorted_[lo - 1]];
  if (address - module.base >= module.size) {
    return nullptr;
  }
  *index = sorted_[lo - 1];
  return &module;
}

// True when the address is in the module's code and the bytes before it are
// a call, which stale data and most function pointers are not.
bool Sampler::IsReturnAddress(uint64_t address, const Module& module) const {
  const uint64_t rva = address - module.base;
  if (rva < module.exec_lo + 8 || rva >= module.exec_hi) {
    return false;
  }
  uint8_t bytes[12] = {};  // bytes[8 - n] is the byte n before the address
  if (!GuardedCopy(bytes, reinterpret_cast<const void*>(address - 8), 8)) {
    return false;
  }
  if (bytes[3] == 0xE8) {  // call rel32
    return true;
  }
  for (uint32_t length : {2u, 3u, 4u, 6u, 7u}) {
    if (IsIndirectCall(bytes + 8 - length, length)) {
      return true;
    }
  }
  return false;
}

Sampler::Address Sampler::Decode(uint64_t key) const {
  const uint32_t module = KeyModule(key);
  const uint64_t offset = KeyOffset(key);
  if (module != kNoModule) {
    return {int32_t(module), offset};
  }
  // A module loaded after the sample may own the address by now.
  uint32_t index = 0;
  if (const Module* found = FindModule(offset, &index)) {
    return {int32_t(index), offset - found->base};
  }
  return {-1, offset};
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

Tracked* Sampler::FindTracked(uint32_t id) {
  for (uint32_t i = 0; i < tracked_count_; ++i) {
    if (tracked_[i].handle && tracked_[i].id == id) {
      return &tracked_[i];
    }
  }
  return nullptr;
}

uint64_t Sampler::ThreadCpu(HANDLE handle) const {
  FILETIME created, exited, kernel, user;
  if (!GetThreadTimes(handle, &created, &exited, &kernel, &user)) {
    return 0;
  }
  return ((uint64_t(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime) +
         ((uint64_t(user.dwHighDateTime) << 32) | user.dwLowDateTime);
}

void Sampler::RefreshThreads() {
  uint32_t fresh[kMaxThreads];
  uint32_t fresh_count = 0;
  bool listed = false;

  for (uint32_t i = 0; i < tracked_count_; ++i) {
    tracked_[i].seen = false;
  }
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snapshot != INVALID_HANDLE_VALUE) {
    THREADENTRY32 entry;
    entry.dwSize = sizeof(entry);
    if (Thread32First(snapshot, &entry)) {
      listed = true;
      do {
        if (entry.th32OwnerProcessID != pid_ || entry.th32ThreadID == thread_id_) {
          continue;
        }
        if (Tracked* thread = FindTracked(entry.th32ThreadID)) {
          thread->seen = true;
        } else if (fresh_count < kMaxThreads) {
          fresh[fresh_count++] = entry.th32ThreadID;
        }
      } while (Thread32Next(snapshot, &entry));
    }
    CloseHandle(snapshot);
  }
  if (!listed) {
    ++snapshot_failed_;
  }

  // Exited threads first: an id can already belong to a new thread.
  for (uint32_t i = 0; i < tracked_count_; ++i) {
    Tracked& thread = tracked_[i];
    if (!thread.handle) {
      continue;
    }
    DWORD exit_code = STILL_ACTIVE;
    const bool exited = GetExitCodeThread(thread.handle, &exit_code) && exit_code != STILL_ACTIVE;
    if (exited || (listed && !thread.seen)) {
      const uint32_t id = thread.id;
      const bool reused = exited && thread.seen;
      Retire(&thread);
      if (reused && fresh_count < kMaxThreads) {
        fresh[fresh_count++] = id;
      }
    }
  }
  for (uint32_t i = 0; i < fresh_count; ++i) {
    Track(fresh[i]);
  }
}

void Sampler::Track(uint32_t id) {
  Tracked* slot = nullptr;
  for (uint32_t i = 0; i < tracked_count_; ++i) {
    if (!tracked_[i].handle) {
      slot = &tracked_[i];
      break;
    }
  }
  if (!slot) {
    if (tracked_count_ == kMaxThreads) {
      ++threads_dropped_;
      return;
    }
    slot = &tracked_[tracked_count_++];
  }
  HANDLE handle =
      OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, id);
  if (!handle) {
    if (slot == &tracked_[tracked_count_ - 1]) {
      --tracked_count_;
    }
    return;
  }
  *slot = Tracked{};
  slot->id = id;
  slot->handle = handle;
  if (nt_query_thread_) {
    ThreadBasicInfo info = {};
    if (nt_query_thread_(handle, 0 /* ThreadBasicInformation */, &info, sizeof(info), nullptr) >=
        0) {
      slot->teb = info.teb;
    }
  }
  slot->cpu_start = slot->cpu_last = ThreadCpu(handle);
  // Sampled from its first CPU time on (UpdateActivity).
  slot->idle_evals = uint8_t(kIdleEvals);
  slot->active = sample_all_;
}

// Final CPU time and name of a thread that has a profile.
void Sampler::CloseOut(Tracked* thread) {
  const uint64_t cpu = ThreadCpu(thread->handle);
  if (cpu > thread->cpu_last) {
    thread->cpu_last = cpu;
  }
  if (thread->profile < 0) {
    return;
  }
  Profile& profile = profiles_[thread->profile];
  profile.cpu_100ns = thread->cpu_last - thread->cpu_start;

  char name[256] = {};
  if (get_thread_description_) {
    PWSTR wide = nullptr;
    if (SUCCEEDED(get_thread_description_(thread->handle, &wide)) && wide) {
      if (!WideCharToMultiByte(CP_UTF8, 0, wide, -1, name, sizeof(name) - 1, nullptr, nullptr)) {
        name[0] = 0;
      }
      LocalFree(wide);
    }
  }
  if (!name[0]) {
    LookupThreadName(thread->id, name, sizeof(name));
  }
  for (char* p = name; *p; ++p) {
    if (static_cast<unsigned char>(*p) < 0x20) {
      *p = ' ';
    }
  }
  std::snprintf(profile.name, sizeof(profile.name), "%s", name);
}

void Sampler::Retire(Tracked* thread) {
  CloseOut(thread);
  CloseHandle(thread->handle);
  *thread = Tracked{};
}

// A thread is sampled while it has used CPU within the last kIdleEvals
// checks. Thread times advance in 10 ms steps under Wine, so the check is
// over quarter seconds and not per tick. Where the times never advance at
// all, every thread is sampled.
void Sampler::UpdateActivity() {
  bool any = false;
  for (uint32_t i = 0; i < tracked_count_; ++i) {
    Tracked& thread = tracked_[i];
    if (!thread.handle) {
      continue;
    }
    const uint64_t cpu = ThreadCpu(thread.handle);
    if (cpu > thread.cpu_last) {
      thread.cpu_last = cpu;
      thread.idle_evals = 0;
      any = true;
    } else if (thread.idle_evals < kIdleEvals) {
      ++thread.idle_evals;
    }
  }
  if (any) {
    blind_evals_ = 0;
    sample_all_ = false;
  } else if (++blind_evals_ >= kBlindEvals) {
    sample_all_ = true;
  }
  for (uint32_t i = 0; i < tracked_count_; ++i) {
    Tracked& thread = tracked_[i];
    thread.active = thread.handle && (sample_all_ || thread.idle_evals < kIdleEvals);
  }
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

void Sampler::SampleThreads() {
  CONTEXT context;
  for (uint32_t i = 0; i < tracked_count_; ++i) {
    Tracked& thread = tracked_[i];
    if (!thread.handle || !thread.active) {
      continue;
    }
    // Between the suspend and the resume: no allocation, no logging, no
    // lock. The thread may hold the heap or the logger.
    if (SuspendThread(thread.handle) == DWORD(-1)) {
      ++suspend_failed_;
      continue;
    }
    context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
    const bool have_context = GetThreadContext(thread.handle, &context) != 0;
    uint32_t slot_count = 0;
    if (have_context) {
      slot_count = CopyStack(thread, context.Rsp);
    }
    ResumeThread(thread.handle);
    if (!have_context) {
      ++context_failed_;
      continue;
    }
    Record(&thread, context.Rip, slot_count);
  }
}

// Copies up to kScanSlots stack slots from RSP towards the stack base of a
// suspended thread. The range is checked first, against the committed stack
// of the thread's TEB or else against the region RSP is in, so a wild RSP is
// not read and a guard page is not touched.
uint32_t Sampler::CopyStack(const Tracked& thread, uint64_t rsp) {
  if (!rsp || (rsp & 7)) {
    return 0;
  }
  uint64_t end = 0;
  if (thread.teb) {
    uint64_t bounds[2] = {};  // NT_TIB StackBase, StackLimit
    if (GuardedCopy(bounds, static_cast<const uint8_t*>(thread.teb) + offsetof(NT_TIB, StackBase),
                    sizeof(bounds)) &&
        rsp >= bounds[1] && rsp < bounds[0]) {
      end = bounds[0];
    }
  }
  if (!end) {
    MEMORY_BASIC_INFORMATION info;
    if (!VirtualQuery(reinterpret_cast<const void*>(rsp), &info, sizeof(info)) ||
        info.State != MEM_COMMIT || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) ||
        !(info.Protect & (PAGE_READWRITE | PAGE_READONLY | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                          PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY))) {
      return 0;
    }
    end = reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize;
  }
  const uint32_t count = uint32_t(std::min<uint64_t>((end - rsp) / 8, kScanSlots));
  if (!count || !GuardedCopy(slots_, reinterpret_cast<const void*>(rsp), count * 8)) {
    return 0;
  }
  return count;
}

Profile* Sampler::ProfileFor(Tracked* thread) {
  if (thread->profile >= 0) {
    return &profiles_[thread->profile];
  }
  if (thread->profile == kProfileDenied) {
    return nullptr;
  }
  thread->profile = kProfileDenied;
  if (profile_count_ == kMaxProfiles) {
    return nullptr;
  }
  uint8_t* memory = pool_ + size_t(profile_count_) * kProfileBytes;
  if (!VirtualAlloc(memory, kProfileBytes, MEM_COMMIT, PAGE_READWRITE)) {
    return nullptr;
  }
  thread->profile = int32_t(profile_count_);
  Profile& profile = profiles_[profile_count_++];
  profile.thread_id = thread->id;
  profile.leaves = reinterpret_cast<LeafEntry*>(memory);
  profile.pairs = reinterpret_cast<PairEntry*>(memory + kLeafSlots * sizeof(LeafEntry));
  return &profile;
}

void Sampler::AddLeaf(Profile* profile, uint64_t key) {
  uint32_t index = uint32_t(Mix(key)) & (kLeafSlots - 1);
  for (uint32_t probe = 0; probe < kMaxProbe; ++probe, index = (index + 1) & (kLeafSlots - 1)) {
    LeafEntry& entry = profile->leaves[index];
    if (entry.key == key) {
      ++entry.count;
      return;
    }
    if (!entry.key) {
      if (profile->leaf_used >= kLeafSlots / 8 * 7) {
        break;
      }
      entry.key = key;
      entry.count = 1;
      ++profile->leaf_used;
      return;
    }
  }
  ++profile->dropped_leaves;
}

void Sampler::AddPair(Profile* profile, uint64_t leaf, uint64_t caller) {
  uint32_t index = uint32_t(Mix(leaf ^ Mix(caller))) & (kPairSlots - 1);
  for (uint32_t probe = 0; probe < kMaxProbe; ++probe, index = (index + 1) & (kPairSlots - 1)) {
    PairEntry& entry = profile->pairs[index];
    if (entry.leaf == leaf && entry.caller == caller) {
      ++entry.count;
      return;
    }
    if (!entry.leaf) {
      if (profile->pair_used >= kPairSlots / 8 * 7) {
        break;
      }
      entry.leaf = leaf;
      entry.caller = caller;
      entry.count = 1;
      ++profile->pair_used;
      return;
    }
  }
  ++profile->dropped_pairs;
}

// Counts the leaf, the first return address above it and, for a leaf outside
// the app's modules, the first return address inside them: the function of
// the runtime or the game that the time in a system module belongs to.
void Sampler::Record(Tracked* thread, uint64_t rip, uint32_t slot_count) {
  Profile* profile = ProfileFor(thread);
  if (!profile) {
    ++no_profile_;
    return;
  }
  ++profile->samples;
  uint32_t leaf_index = 0;
  const Module* leaf_module = FindModule(rip, &leaf_index);
  const uint64_t leaf =
      leaf_module ? MakeKey(leaf_index, rip - leaf_module->base) : MakeKey(kNoModule, rip);
  AddLeaf(profile, leaf);
  if (!slot_count) {
    ++profile->no_stack;
    return;
  }

  const bool want_app = !leaf_module || !leaf_module->app;
  uint64_t near_key = 0;
  uint64_t app_key = 0;
  uint32_t callers = 0;
  for (uint32_t i = 0; i < slot_count && callers < kMaxCallers; ++i) {
    const uint64_t value = slots_[i];
    uint32_t index = 0;
    const Module* module = FindModule(value, &index);
    if (!module || !IsReturnAddress(value, *module)) {
      continue;
    }
    ++callers;
    const uint64_t key = MakeKey(index, value - module->base);
    if (!near_key) {
      near_key = key;
    }
    if (!want_app) {
      break;
    }
    if (module->app) {
      app_key = key;
      break;
    }
  }
  if (near_key) {
    ++profile->chains;
    AddPair(profile, leaf, near_key);
  }
  if (app_key) {
    AddPair(profile, leaf, app_key | kAppCallerBit);
  }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

void Sampler::Finish() {
  for (uint32_t i = 0; i < tracked_count_; ++i) {
    if (tracked_[i].handle) {
      Retire(&tracked_[i]);
    }
  }
  tracked_count_ = 0;
  RefreshModules();
}

void Sampler::Dump() {
  std::FILE* file = rex::filesystem::OpenFile(path_, "w");
  if (!file) {
    REXLOG_ERROR("[cpu-sample] cannot write {}", rex::path_to_utf8(path_));
    return;
  }
  using ull = unsigned long long;
  const double seconds = double(end_ticks_ - start_ticks_) / double(frequency_);

  std::fprintf(file, "# rex cpu samples v1\n");
  std::fprintf(file, "# module <index> <base> <size> <app> <path>\n");
  std::fprintf(file,
               "# thread <id> <samples> <cpu_ms> <with_caller> <no_stack> <dropped_leaves> "
               "<dropped_callers> <name>\n");
  std::fprintf(file, "# leaf <module> <rva> <count>\n");
  std::fprintf(file, "# caller <leaf_module> <leaf_rva> <caller_module> <caller_rva> <count>\n");
  std::fprintf(file, "# appcaller <leaf_module> <leaf_rva> <caller_module> <caller_rva> <count>\n");
  std::fprintf(file, "# caller: first return address above the leaf. appcaller: first one in\n");
  std::fprintf(file, "# an app module, for a leaf outside them. module -1: no module, the rva\n");
  std::fprintf(file,
               "# is the address. Samples are wall-clock; cpu_ms is the thread's CPU time.\n");
  std::fprintf(file, "hz\t%d\n", hz_);
  std::fprintf(file, "ticks\t%llu\n", ull(ticks_));
  std::fprintf(file, "seconds\t%.3f\n", seconds);
  std::fprintf(file, "pid\t%lu\n", static_cast<unsigned long>(pid_));
  std::fprintf(file, "wine\t%s\n", wine_version_[0] ? wine_version_ : "-");
  std::fprintf(file, "scan_slots\t%u\n", kScanSlots);
  std::fprintf(file,
               "stats\tsuspend_failed\t%llu\tcontext_failed\t%llu\tno_profile\t%llu\t"
               "threads_dropped\t%llu\tmodules_dropped\t%llu\tsnapshot_failed\t%llu\t"
               "sample_all\t%d\n",
               ull(suspend_failed_), ull(context_failed_), ull(no_profile_), ull(threads_dropped_),
               ull(modules_dropped_), ull(snapshot_failed_), sample_all_ ? 1 : 0);
  for (uint32_t i = 0; i < module_count_; ++i) {
    const Module& module = modules_[i];
    std::fprintf(file, "module\t%u\t0x%llx\t0x%llx\t%d\t%s\n", i, ull(module.base),
                 ull(module.size), module.app ? 1 : 0, module.path);
  }

  std::vector<uint32_t> order;
  for (uint32_t i = 0; i < profile_count_; ++i) {
    order.push_back(i);
  }
  std::sort(order.begin(), order.end(),
            [this](uint32_t a, uint32_t b) { return profiles_[a].samples > profiles_[b].samples; });

  std::vector<LeafEntry> leaves;
  std::vector<PairEntry> pairs[2];
  for (uint32_t profile_index : order) {
    const Profile& profile = profiles_[profile_index];
    const double cpu_ms = double(profile.cpu_100ns) / 10000.0;
    std::fprintf(file, "thread\t%lu\t%llu\t%.1f\t%llu\t%llu\t%llu\t%llu\t%s\n",
                 static_cast<unsigned long>(profile.thread_id), ull(profile.samples), cpu_ms,
                 ull(profile.chains), ull(profile.no_stack), ull(profile.dropped_leaves),
                 ull(profile.dropped_pairs), profile.name[0] ? profile.name : "-");

    leaves.clear();
    for (uint32_t i = 0; i < kLeafSlots; ++i) {
      if (profile.leaves[i].key) {
        leaves.push_back(profile.leaves[i]);
      }
    }
    std::sort(leaves.begin(), leaves.end(),
              [](const LeafEntry& a, const LeafEntry& b) { return a.count > b.count; });
    for (const LeafEntry& entry : leaves) {
      const Address leaf = Decode(entry.key);
      std::fprintf(file, "leaf\t%d\t0x%llx\t%llu\n", leaf.module, ull(leaf.offset),
                   ull(entry.count));
    }

    pairs[0].clear();
    pairs[1].clear();
    for (uint32_t i = 0; i < kPairSlots; ++i) {
      if (profile.pairs[i].leaf) {
        pairs[(profile.pairs[i].caller & kAppCallerBit) ? 1 : 0].push_back(profile.pairs[i]);
      }
    }
    for (int kind = 0; kind < 2; ++kind) {
      std::sort(pairs[kind].begin(), pairs[kind].end(),
                [](const PairEntry& a, const PairEntry& b) { return a.count > b.count; });
      if (pairs[kind].size() > kPairsWritten) {
        pairs[kind].resize(kPairsWritten);
      }
      for (const PairEntry& entry : pairs[kind]) {
        const Address leaf = Decode(entry.leaf);
        const Address caller = Decode(entry.caller);
        std::fprintf(file, "%s\t%d\t0x%llx\t%d\t0x%llx\t%llu\n", kind ? "appcaller" : "caller",
                     leaf.module, ull(leaf.offset), caller.module, ull(caller.offset),
                     ull(entry.count));
      }
    }

    if (leaves.empty() || !profile.samples) {
      continue;
    }
    const Address top = Decode(leaves[0].key);
    REXLOG_INFO("[cpu-sample] thread {} '{}': {} samples, {:.0f} ms cpu, top {}+0x{:x} {:.1f}%",
                profile.thread_id, profile.name, profile.samples, cpu_ms,
                top.module >= 0 ? BaseName(modules_[top.module].path) : "?", top.offset,
                100.0 * double(leaves[0].count) / double(profile.samples));
  }
  std::fclose(file);

  REXLOG_INFO("[cpu-sample] {} ticks in {:.1f} s, {} threads, {} modules, wrote {}", ticks_,
              seconds, profile_count_, module_count_, rex::path_to_utf8(path_));
  if (suspend_failed_ || context_failed_ || no_profile_ || threads_dropped_ || sample_all_) {
    REXLOG_WARN(
        "[cpu-sample] suspend failed {}, context failed {}, samples without a table {}, threads "
        "not tracked {}, thread times {}",
        suspend_failed_, context_failed_, no_profile_, threads_dropped_,
        sample_all_ ? "never advanced (every thread sampled)" : "ok");
  }
}

std::atomic<Sampler*> g_sampler{nullptr};
std::atomic<bool> g_started{false};

}  // namespace

void Start() {
  const int hz = REXCVAR_GET(cpu_sample_hz);
  if (hz <= 0 || g_started.exchange(true)) {
    return;
  }
  auto* sampler = new Sampler();
  if (!sampler->Initialize(hz, REXCVAR_GET(cpu_sample_seconds), REXCVAR_GET(cpu_sample_delay))) {
    delete sampler;
    return;
  }
  g_sampler.store(sampler);
}

void Stop() {
  Sampler* sampler = g_sampler.exchange(nullptr);
  if (!sampler) {
    return;
  }
  sampler->Shutdown();
  delete sampler;
}

void NoteThreadName(uint32_t thread_id, std::string_view name) {
  if (!thread_id) {
    return;
  }
  AcquireSRWLockExclusive(&g_names_lock);
  NamedThread* slot = nullptr;
  for (NamedThread& named : g_names) {
    if (named.id == thread_id) {
      slot = &named;
      break;
    }
  }
  if (!slot) {
    slot = &g_names[g_names_next++ % kNamedThreads];
  }
  slot->id = thread_id;
  const size_t length = std::min(name.size(), sizeof(slot->name) - 1);
  std::memcpy(slot->name, name.data(), length);
  slot->name[length] = 0;
  ReleaseSRWLockExclusive(&g_names_lock);
}

}  // namespace rex::cpu_sampler

#else  // REX_PLATFORM_WIN32 && REX_ARCH_AMD64

namespace rex::cpu_sampler {

void Start() {
  if (REXCVAR_GET(cpu_sample_hz) > 0) {
    REXLOG_WARN("[cpu-sample] cpu_sample_hz is implemented on Windows x64 only");
  }
}

void Stop() {}

void NoteThreadName(uint32_t, std::string_view) {}

}  // namespace rex::cpu_sampler

#endif  // REX_PLATFORM_WIN32 && REX_ARCH_AMD64
