/**
 * @file        ui/shader_progress_win.cpp
 * @brief       Win32 "preparing shaders" progress window (own thread).
 *
 * @license     BSD 3-Clause License
 */

#include <rex/ui/shader_progress.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

// clang-format off
#include <windows.h>
#include <commctrl.h>
// clang-format on

namespace rex::ui {

namespace {

constexpr int kShowDelayMs = 900;   // quick boots never see the window
constexpr int kPollMs = 100;

std::wstring Widen(const std::string& s) {
  if (s.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
  return w;
}

class Win32ShaderProgressWindow : public ShaderProgressWindow {
 public:
  Win32ShaderProgressWindow(std::string title, ShaderProgressPoll poll)
      : title_(std::move(title)), poll_(std::move(poll)) {
    thread_ = std::thread([this]() { Run(); });
  }

  ~Win32ShaderProgressWindow() override {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
  }

 private:
  static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    if (msg == WM_CLOSE) return 0;  // not closable by the user
    return DefWindowProcW(hwnd, msg, wparam, lparam);
  }

  void Create() {
    INITCOMMONCONTROLSEX icc{sizeof(icc), ICC_PROGRESS_CLASS};
    InitCommonControlsEx(&icc);
    HINSTANCE hinstance = GetModuleHandleW(nullptr);
    const wchar_t* kClassName = L"ShaderProgressWindow";
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hinstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    wc.lpszClassName = kClassName;
    wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
    RegisterClassExW(&wc);

    NONCLIENTMETRICSW ncm{};
    ncm.cbSize = sizeof(ncm);
    if (SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof(ncm), &ncm, 0)) {
      font_ = CreateFontIndirectW(&ncm.lfMessageFont);
    }

    const DWORD style = WS_OVERLAPPED | WS_CAPTION;
    RECT rc{0, 0, 460, 132};
    AdjustWindowRectEx(&rc, style, FALSE, 0);
    int w = rc.right - rc.left, h = rc.bottom - rc.top;
    int x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
    int y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;
    std::wstring title = Widen(title_) + L"  -  Preparing shaders";
    hwnd_ = CreateWindowExW(WS_EX_TOPMOST, kClassName, title.c_str(), style, x, y, w, h, nullptr,
                            nullptr, hinstance, nullptr);
    if (!hwnd_) return;

    auto make = [&](const wchar_t* cls, const wchar_t* text, DWORD st, int cx, int cy, int cw,
                    int ch) {
      HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | st, cx, cy, cw, ch, hwnd_,
                               nullptr, hinstance, nullptr);
      if (c && font_) SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font_), TRUE);
      return c;
    };
    make(L"STATIC",
         L"First launch on this PC: the game's shaders are being prepared. This "
         L"happens once and can take a few minutes.",
         SS_LEFT, 16, 14, 428, 40);
    label_ = make(L"STATIC", L"Loading shaders...", SS_LEFT, 16, 62, 428, 20);
    bar_ = make(PROGRESS_CLASSW, nullptr, PBS_MARQUEE, 16, 90, 428, 22);
    SendMessageW(bar_, PBM_SETMARQUEE, TRUE, 30);
    marquee_ = true;

    ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
    UpdateWindow(hwnd_);
  }

  void Update(uint32_t phase, uint32_t done, uint32_t total) {
    if (!hwnd_) return;
    if (phase == 2 && total > 0) {
      if (marquee_) {
        SetWindowLongPtrW(bar_, GWL_STYLE, GetWindowLongPtrW(bar_, GWL_STYLE) & ~PBS_MARQUEE);
        SendMessageW(bar_, PBM_SETMARQUEE, FALSE, 0);
        SendMessageW(bar_, PBM_SETRANGE32, 0, LPARAM(total));
        marquee_ = false;
      }
      SendMessageW(bar_, PBM_SETPOS, WPARAM(done), 0);
      wchar_t text[96];
      swprintf_s(text, L"Building shader pipelines  %u / %u", unsigned(done), unsigned(total));
      SetWindowTextW(label_, text);
    } else {
      SetWindowTextW(label_, L"Loading shaders...");
    }
  }

  void Pump() {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
  }

  void Run() {
    auto start = std::chrono::steady_clock::now();
    while (!stop_.load(std::memory_order_acquire)) {
      uint32_t phase = 0, done = 0, total = 0;
      bool running = poll_ && poll_(&phase, &done, &total);
      if (!hwnd_ && running) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
        if (elapsed >= kShowDelayMs) Create();
      }
      if (hwnd_) {
        if (running) Update(phase, done, total);
        Pump();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
    }
    if (hwnd_) {
      DestroyWindow(hwnd_);
      Pump();
      hwnd_ = nullptr;
    }
    if (font_) {
      DeleteObject(font_);
      font_ = nullptr;
    }
  }

  std::string title_;
  ShaderProgressPoll poll_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  HWND hwnd_ = nullptr;
  HWND label_ = nullptr;
  HWND bar_ = nullptr;
  HFONT font_ = nullptr;
  bool marquee_ = true;
};

}  // namespace

std::unique_ptr<ShaderProgressWindow> StartShaderProgressWindow(std::string title,
                                                                ShaderProgressPoll poll) {
  return std::make_unique<Win32ShaderProgressWindow>(std::move(title), std::move(poll));
}

}  // namespace rex::ui
