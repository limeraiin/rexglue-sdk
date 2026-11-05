// [ftrace] runtime glue for include/rex/graphics/frame_trace.h: the global
// ring, the trace start stamps, the 1 Hz line and the shutdown/crash CSV dump.
// The ring is written only by the command-processor thread (XE_SWAP); the
// crash entry reads it best-effort from whatever thread faulted.
#include <rex/graphics/frame_trace.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <string>

#include <rex/chrono/clock.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/ui/graphics_upload_buffer_pool.h>  // [ftrace] v3 upalloc timing flag

namespace rex::graphics::ftrace {

// v3 exec split: the gate and the ledger (CP thread writes, swap reads).
bool g_exec_split = false;
SplitTotals g_split;
// v4: the trace is on at all (processor samples at the draw site).
bool g_on = false;

namespace {

int g_mode = 0;
FrameTraceRing g_ring;
uint64_t g_start_ticks = 0;
uint64_t g_freq = 0;
int64_t g_start_wall_ms = 0;
uint64_t g_report_seq = 0;  // first sequence not yet covered by a 1 Hz line
std::atomic<int> g_dumped{0};

// Writes the ring to `path`; the summary line names the path.
void Dump(const std::filesystem::path& path, const char* how) {
  if (!g_ring.allocated()) {
    REXGPU_INFO("[ftrace] nothing to dump ({}: no frames)", how);
    return;
  }
  CsvInfo info;
  info.freq = g_freq;
  info.start_wall_ms = g_start_wall_ms;
  bool ok = false;
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ok = out.is_open() && WriteCsv(out, g_ring, info);
  }
  const std::string path_utf8 = rex::path_to_utf8(path);
  if (!ok) {
    REXGPU_WARN("[ftrace] CSV write failed ({}): {}", how, path_utf8);
  }
  const FrameSummary s = Summarize(g_ring, g_ring.oldest_seq(), g_ring.pushed());
  char line[1024];
  FormatShutdownLine(line, sizeof(line), s, path_utf8.c_str());
  REXGPU_INFO("[ftrace] {}", line);
}

std::filesystem::path CsvPath(const char* suffix) {
  std::filesystem::path log = rex::CurrentLogFilePath();
  std::filesystem::path dir = log.empty() ? std::filesystem::current_path() : log.parent_path();
  std::string stem = log.empty() ? std::string("frames") : log.stem().string();
  return dir / (stem + suffix);
}

}  // namespace

int Mode() { return g_mode; }

void Begin(int mode) {
  g_mode = mode;
  // Mode 3: every split bracket and the upload pools' page-create timer.
  g_exec_split = ExecSplitForMode(mode);
  rex::ui::g_upload_pool_trace.timed = g_exec_split;
  g_on = mode > 0;
  if (mode <= 0) return;
  g_freq = rex::chrono::Clock::QueryHostTickFrequency();
  g_start_ticks = rex::chrono::Clock::QueryHostTickCount();
  g_start_wall_ms = int64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch())
                                .count());
  g_report_seq = g_ring.pushed();
  REXGPU_INFO("[ftrace] on: mode {} ({}), ring {} frames, slow > {:.0f} ms, CSV at shutdown",
              mode,
              mode >= 3   ? "cpu + gpu timestamps + exec split"
              : mode >= 2 ? "cpu + gpu timestamps"
                          : "cpu",
              FrameTraceRing::kCapacity, kSlowFrameMs);
}

FrameTraceRing* Ring() { return g_mode > 0 ? &g_ring : nullptr; }

uint64_t StartTicks() { return g_start_ticks; }

uint64_t TickFrequency() { return g_freq; }

void Report1Hz() {
  if (g_mode <= 0) return;
  const uint64_t end = g_ring.pushed();
  if (end == g_report_seq) return;
  const FrameSummary s = Summarize(g_ring, g_report_seq, end);
  g_report_seq = end;
  if (!s.count) return;
  char line[512];
  FormatReport1Hz(line, sizeof(line), s, g_mode >= 3);
  REXGPU_INFO("[ftrace] {}", line);
}

void DumpAtShutdown() {
  if (g_mode <= 0) return;
  int expected = 0;
  if (!g_dumped.compare_exchange_strong(expected, 1)) return;
  Dump(CsvPath(".frames.csv"), "shutdown");
}

}  // namespace rex::graphics::ftrace

extern "C" void rex_frame_trace_dump() {
  using namespace rex::graphics::ftrace;
  if (g_mode <= 0) return;
  int expected = 0;
  if (!g_dumped.compare_exchange_strong(expected, 1)) return;
  Dump(CsvPath(".frames.crash.csv"), "crash");
}
