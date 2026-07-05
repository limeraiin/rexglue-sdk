/**
 * @file        ui/startup_config_dialog_win.cpp
 * @brief       Native Win32 pre-launch configuration dialog.
 *
 * Built as a plain top-level window with standard controls (no .rc resource),
 * driven by a local modal message loop. Reads/writes cvars by name and persists
 * to the consumer's config TOML. Shown before the game window/graphics exist.
 *
 * @copyright   Copyright (c) 2026 Tom Clay
 * @license     BSD 3-Clause License
 */

#include <rex/ui/startup_config_dialog.h>

#include <cstdlib>
#include <iterator>
#include <string>

#include <rex/cvar.h>

// clang-format off
#include <windows.h>
#include <shlobj.h>   // SHBrowseForFolderW, SHGetPathFromIDListW
// clang-format on

namespace rex::ui {

namespace {

constexpr int IDC_RENDERER = 1001;
constexpr int IDC_RES = 1002;
// (IDC_VSYNC 1003 retired: vsync is always on, no dialog control.)
constexpr int IDC_DISPLAY_MODE = 1004;
constexpr int IDC_DATA = 1005;
constexpr int IDC_BROWSE = 1006;
constexpr int IDC_INTERNAL_RES = 1007;
constexpr int IDC_EXP_60FPS = 1008;

struct GpuOption {
  const wchar_t* label;
  const char* value;
};
const GpuOption kGpuOptions[] = {
    {L"Auto (Direct3D 12)", "any"},
    {L"Direct3D 12", "d3d12"},
    {L"Vulkan", "vulkan"},
};

struct ResPreset {
  int width;
  int height;
  const wchar_t* label;
};
const ResPreset kResPresets[] = {
    {1280, 720, L"1280 x 720 (720p)"},   {1600, 900, L"1600 x 900"},
    {1920, 1080, L"1920 x 1080 (1080p)"}, {2560, 1440, L"2560 x 1440 (1440p)"},
    {3840, 2160, L"3840 x 2160 (4K)"},
};

// Internal render-resolution scale (Xenia draw_resolution_scale). The guest
// renders at a fixed 720p; this supersamples its render targets by an integer
// factor (2x -> 1440p, 3x -> 2160p), which is the only lever that changes the
// actual rendered resolution. Maps to the `resolution_scale` cvar.
struct InternalResOption {
  int scale;
  const wchar_t* label;
};
const InternalResOption kInternalResOptions[] = {
    {1, L"Native (720p)"},
    {2, L"1440p (2x supersampling)"},
    {3, L"2160p / 4K (3x supersampling)"},
};

// Display mode. The runtime implements "fullscreen" as borderless fullscreen
// (no DXGI exclusive mode); maps to the bool `fullscreen` cvar.
struct DisplayModeOption {
  bool fullscreen;
  const wchar_t* label;
};
const DisplayModeOption kDisplayModes[] = {
    {false, L"Windowed"},
    {true, L"Borderless Fullscreen"},
};

struct DialogState {
  bool done = false;
  bool play = false;
  HWND combo_renderer = nullptr;
  HWND combo_res = nullptr;
  HWND combo_internal_res = nullptr;
  HWND check_exp_60fps = nullptr;
  HWND combo_display = nullptr;
  HWND edit_data = nullptr;
  HFONT font = nullptr;
  HFONT font_small = nullptr;  // smaller font for the bottom-left credits line
};

std::wstring Widen(const std::string& s) {
  if (s.empty()) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
  return w;
}

std::string Narrow(const std::wstring& w) {
  if (w.empty()) return {};
  int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(size_t(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

std::wstring GetEditText(HWND edit) {
  int len = GetWindowTextLengthW(edit);
  if (len <= 0) return {};
  std::wstring text(size_t(len) + 1, L'\0');
  GetWindowTextW(edit, text.data(), len + 1);
  text.resize(size_t(len));
  return text;
}

int CALLBACK BrowseCallback(HWND hwnd, UINT msg, LPARAM /*lp*/, LPARAM data) {
  if (msg == BFFM_INITIALIZED && data) {
    // Preselect the current folder (data is a wchar_t*).
    SendMessageW(hwnd, BFFM_SETSELECTIONW, TRUE, data);
  }
  return 0;
}

std::wstring BrowseForFolder(HWND owner, const std::wstring& initial) {
  std::wstring result;
  BROWSEINFOW bi{};
  bi.hwndOwner = owner;
  bi.lpszTitle = L"Select the game data folder";
  bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_EDITBOX;
  if (!initial.empty()) {
    bi.lpfn = BrowseCallback;
    bi.lParam = reinterpret_cast<LPARAM>(initial.c_str());
  }
  LPITEMIDLIST pidl = SHBrowseForFolderW(&bi);
  if (pidl) {
    wchar_t path[MAX_PATH] = {};
    if (SHGetPathFromIDListW(pidl, path)) {
      result = path;
    }
    CoTaskMemFree(pidl);
  }
  return result;
}

HWND MakeControl(HWND parent, const wchar_t* cls, const wchar_t* text, DWORD style, int x, int y,
                 int w, int h, int id, HFONT font) {
  HWND ctl = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style, x, y, w, h, parent,
                             reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                             reinterpret_cast<HINSTANCE>(GetWindowLongPtrW(parent, GWLP_HINSTANCE)),
                             nullptr);
  if (ctl && font) {
    SendMessageW(ctl, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
  }
  return ctl;
}

void BuildControls(HWND hwnd, DialogState* st) {
  HFONT font = st->font;

  MakeControl(hwnd, L"STATIC", L"Graphics renderer:", SS_LEFT, 16, 18, 150, 20, -1, font);
  st->combo_renderer = MakeControl(hwnd, L"COMBOBOX", nullptr,
                                   CBS_DROPDOWNLIST | WS_TABSTOP | WS_VSCROLL, 172, 16, 260, 220,
                                   IDC_RENDERER, font);

  MakeControl(hwnd, L"STATIC", L"Window resolution:", SS_LEFT, 16, 52, 150, 20, -1, font);
  st->combo_res = MakeControl(hwnd, L"COMBOBOX", nullptr, CBS_DROPDOWNLIST | WS_TABSTOP | WS_VSCROLL,
                              172, 50, 260, 220, IDC_RES, font);

  MakeControl(hwnd, L"STATIC", L"Internal resolution:", SS_LEFT, 16, 86, 150, 20, -1, font);
  st->combo_internal_res =
      MakeControl(hwnd, L"COMBOBOX", nullptr, CBS_DROPDOWNLIST | WS_TABSTOP | WS_VSCROLL, 172, 84,
                  260, 220, IDC_INTERNAL_RES, font);

  MakeControl(hwnd, L"STATIC", L"Display mode:", SS_LEFT, 16, 120, 150, 20, -1, font);
  st->combo_display =
      MakeControl(hwnd, L"COMBOBOX", nullptr, CBS_DROPDOWNLIST | WS_TABSTOP | WS_VSCROLL, 172, 118,
                  260, 220, IDC_DISPLAY_MODE, font);

  // Vsync is always on (no dialog control; forced true in ApplyAndSave).
  st->check_exp_60fps =
      MakeControl(hwnd, L"BUTTON", L"60 FPS overworld (EXPERIMENTAL)", BS_AUTOCHECKBOX | WS_TABSTOP,
                  172, 154, 260, 22, IDC_EXP_60FPS, font);

  MakeControl(hwnd, L"STATIC", L"Game data folder:", SS_LEFT, 16, 222, 200, 20, -1, font);
  st->edit_data = MakeControl(hwnd, L"EDIT", nullptr,
                              WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, 16, 244, 330, 24, IDC_DATA,
                              font);
  MakeControl(hwnd, L"BUTTON", L"Browse...", WS_TABSTOP, 352, 243, 80, 26, IDC_BROWSE, font);

  MakeControl(hwnd, L"BUTTON", L"Play", BS_DEFPUSHBUTTON | WS_TABSTOP, 256, 298, 84, 30, IDOK, font);
  MakeControl(hwnd, L"BUTTON", L"Quit", WS_TABSTOP, 348, 298, 84, 30, IDCANCEL, font);

  // Bottom-left credits line, drawn in a smaller font. Spans the full client
  // width so it can wrap to a second line if the system font is large.
  MakeControl(hwnd, L"STATIC",
              L"Special thanks to Simeon, Kalarot, Vexil Megga, Hailnate13x, GUARD, ctrlalt3l1t3, "
              L"Austin_Toonz, seraf5 and Mark_Rampage for their Patreon support.",
              SS_LEFT, 16, 332, 416, 50, -1, st->font_small ? st->font_small : font);

  // --- Populate from current cvar values ---
  std::string gpu = rex::cvar::GetFlagByName("gpu");
  int gpu_sel = 0;
  for (int i = 0; i < int(std::size(kGpuOptions)); ++i) {
    SendMessageW(st->combo_renderer, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>(kGpuOptions[i].label));
    if (gpu == kGpuOptions[i].value) gpu_sel = i;
  }
  SendMessageW(st->combo_renderer, CB_SETCURSEL, gpu_sel, 0);

  int cur_w = std::atoi(rex::cvar::GetFlagByName("video_mode_width").c_str());
  int cur_h = std::atoi(rex::cvar::GetFlagByName("video_mode_height").c_str());
  int res_sel = 0;
  for (int i = 0; i < int(std::size(kResPresets)); ++i) {
    SendMessageW(st->combo_res, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(kResPresets[i].label));
    if (cur_w == kResPresets[i].width && cur_h == kResPresets[i].height) res_sel = i;
  }
  SendMessageW(st->combo_res, CB_SETCURSEL, res_sel, 0);

  int cur_scale = std::atoi(rex::cvar::GetFlagByName("resolution_scale").c_str());
  int internal_sel = 0;
  for (int i = 0; i < int(std::size(kInternalResOptions)); ++i) {
    SendMessageW(st->combo_internal_res, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>(kInternalResOptions[i].label));
    if (cur_scale == kInternalResOptions[i].scale) internal_sel = i;
  }
  SendMessageW(st->combo_internal_res, CB_SETCURSEL, internal_sel, 0);

  SendMessageW(st->check_exp_60fps, BM_SETCHECK,
               rex::cvar::GetFlagByName("experimental_60fps") == "true" ? BST_CHECKED
                                                                        : BST_UNCHECKED,
               0);

  bool cur_fullscreen = rex::cvar::GetFlagByName("fullscreen") == "true";
  int display_sel = 0;
  for (int i = 0; i < int(std::size(kDisplayModes)); ++i) {
    SendMessageW(st->combo_display, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>(kDisplayModes[i].label));
    if (cur_fullscreen == kDisplayModes[i].fullscreen) display_sel = i;
  }
  SendMessageW(st->combo_display, CB_SETCURSEL, display_sel, 0);

  std::wstring data_dir = Widen(rex::cvar::GetFlagByName("game_data_root"));
  SetWindowTextW(st->edit_data, data_dir.c_str());
}

// Applies the dialog selections to cvars and persists them. Returns false (with
// a message box) if the game data folder is missing/invalid.
bool ApplyAndSave(HWND hwnd, DialogState* st, const std::filesystem::path& config_path) {
  std::wstring data = GetEditText(st->edit_data);
  while (!data.empty() && (data.back() == L'\\' || data.back() == L'/' || data.back() == L' ')) {
    data.pop_back();
  }
  if (data.empty()) {
    MessageBoxW(hwnd, L"Please choose the game data folder.", L"Game data folder required",
                MB_OK | MB_ICONWARNING);
    return false;
  }
  DWORD attr = GetFileAttributesW(data.c_str());
  if (attr == INVALID_FILE_ATTRIBUTES || !(attr & FILE_ATTRIBUTE_DIRECTORY)) {
    MessageBoxW(hwnd, L"That game data folder does not exist.", L"Invalid folder",
                MB_OK | MB_ICONWARNING);
    return false;
  }

  int gpu_sel = int(SendMessageW(st->combo_renderer, CB_GETCURSEL, 0, 0));
  if (gpu_sel < 0 || gpu_sel >= int(std::size(kGpuOptions))) gpu_sel = 0;
  rex::cvar::SetFlagByName("gpu", kGpuOptions[gpu_sel].value);

  int res_sel = int(SendMessageW(st->combo_res, CB_GETCURSEL, 0, 0));
  if (res_sel < 0 || res_sel >= int(std::size(kResPresets))) res_sel = 0;
  rex::cvar::SetFlagByName("video_mode_width", std::to_string(kResPresets[res_sel].width));
  rex::cvar::SetFlagByName("video_mode_height", std::to_string(kResPresets[res_sel].height));

  int internal_sel = int(SendMessageW(st->combo_internal_res, CB_GETCURSEL, 0, 0));
  if (internal_sel < 0 || internal_sel >= int(std::size(kInternalResOptions))) internal_sel = 0;
  rex::cvar::SetFlagByName("resolution_scale",
                           std::to_string(kInternalResOptions[internal_sel].scale));

  // Vsync is always on (no dialog control) — force it true on every save.
  rex::cvar::SetFlagByName("vsync", "true");

  rex::cvar::SetFlagByName("experimental_60fps",
                           SendMessageW(st->check_exp_60fps, BM_GETCHECK, 0, 0) == BST_CHECKED
                               ? "true"
                               : "false");

  int display_sel = int(SendMessageW(st->combo_display, CB_GETCURSEL, 0, 0));
  if (display_sel < 0 || display_sel >= int(std::size(kDisplayModes))) display_sel = 0;
  rex::cvar::SetFlagByName("fullscreen", kDisplayModes[display_sel].fullscreen ? "true" : "false");
  rex::cvar::SetFlagByName("game_data_root", Narrow(data));

  rex::cvar::SaveConfig(config_path);
  return true;
}

const std::filesystem::path* g_config_path = nullptr;  // valid for the dialog's lifetime

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
  auto* st = reinterpret_cast<DialogState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
  switch (msg) {
    case WM_CREATE: {
      auto* cs = reinterpret_cast<CREATESTRUCTW*>(lparam);
      st = reinterpret_cast<DialogState*>(cs->lpCreateParams);
      SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(st));
      BuildControls(hwnd, st);
      return 0;
    }
    case WM_COMMAND: {
      if (!st) break;
      int id = LOWORD(wparam);
      if (id == IDC_BROWSE) {
        std::wstring picked = BrowseForFolder(hwnd, GetEditText(st->edit_data));
        if (!picked.empty()) SetWindowTextW(st->edit_data, picked.c_str());
        return 0;
      }
      if (id == IDOK) {
        if (ApplyAndSave(hwnd, st, *g_config_path)) {
          st->play = true;
          st->done = true;
          DestroyWindow(hwnd);
        }
        return 0;
      }
      if (id == IDCANCEL) {
        st->play = false;
        st->done = true;
        DestroyWindow(hwnd);
        return 0;
      }
      break;
    }
    case WM_CLOSE:
      if (st) {
        st->play = false;
        st->done = true;
      }
      DestroyWindow(hwnd);
      return 0;
    // NOTE: deliberately NO WM_DESTROY/PostQuitMessage here. The modal loop
    // below exits on st->done, and posting WM_QUIT would leave it in the
    // thread's message queue (the loop often breaks before consuming it),
    // which would then make the app's main message loop quit immediately ->
    // black screen with audio still playing.
    default:
      break;
  }
  return DefWindowProcW(hwnd, msg, wparam, lparam);
}

}  // namespace

bool ShowStartupConfigDialog(std::string_view app_name, const std::filesystem::path& config_path) {
  HINSTANCE hinstance = GetModuleHandleW(nullptr);

  const wchar_t* kClassName = L"RexStartupConfigDialog";
  WNDCLASSEXW wc{};
  wc.cbSize = sizeof(wc);
  wc.lpfnWndProc = WndProc;
  wc.hInstance = hinstance;
  // Use the unsuffixed resource loaders: IDC_ARROW / IDI_APPLICATION are
  // MAKEINTRESOURCE values whose A/W flavor follows the UNICODE macro (not
  // defined here), so they must pair with the matching unsuffixed function.
  wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
  wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
  wc.lpszClassName = kClassName;
  wc.hIcon = LoadIcon(nullptr, IDI_APPLICATION);
  RegisterClassExW(&wc);  // ignore failure if already registered

  DialogState state;
  NONCLIENTMETRICSW ncm{};
  ncm.cbSize = sizeof(ncm);
  if (SystemParametersInfoW(SPI_GETNONCLIENTMETRICS, sizeof(ncm), &ncm, 0)) {
    state.font = CreateFontIndirectW(&ncm.lfMessageFont);
    // A slightly smaller flavour of the message font for the credits line.
    // NB: can't name this `small` — <rpcndr.h> does `#define small char`.
    LOGFONTW small_lf = ncm.lfMessageFont;
    LONG h = small_lf.lfHeight < 0 ? -small_lf.lfHeight : small_lf.lfHeight;
    h = (h * 5) / 6;  // ~83% of the message-font size
    small_lf.lfHeight = small_lf.lfHeight < 0 ? -h : h;
    state.font_small = CreateFontIndirectW(&small_lf);
  }

  g_config_path = &config_path;

  // Client area 448 x 384; size the window to fit it (the extra bottom strip
  // below the buttons holds the smaller-font credits line).
  const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
  RECT rc{0, 0, 448, 384};
  AdjustWindowRectEx(&rc, style, FALSE, 0);
  int win_w = rc.right - rc.left;
  int win_h = rc.bottom - rc.top;
  int x = (GetSystemMetrics(SM_CXSCREEN) - win_w) / 2;
  int y = (GetSystemMetrics(SM_CYSCREEN) - win_h) / 2;

  std::wstring title = Widen(std::string(app_name)) + L"  —  Setup";

  HWND hwnd = CreateWindowExW(0, kClassName, title.c_str(), style, x, y, win_w, win_h, nullptr,
                              nullptr, hinstance, &state);
  if (!hwnd) {
    if (state.font) DeleteObject(state.font);
    if (state.font_small) DeleteObject(state.font_small);
    g_config_path = nullptr;
    return true;  // Don't block launch if the dialog can't be created.
  }

  ShowWindow(hwnd, SW_SHOW);
  SetForegroundWindow(hwnd);
  if (state.combo_renderer) SetFocus(state.combo_renderer);

  // Local modal message loop until the user picks Play/Quit (state.done).
  // Driven by the done flag — NOT by WM_QUIT — so we never post WM_QUIT into
  // this thread's queue (which the app's main loop would later consume and quit
  // on). The `!state.done` short-circuit means once a choice is made we stop
  // before blocking in GetMessage again.
  MSG msg;
  while (!state.done && GetMessageW(&msg, nullptr, 0, 0) > 0) {
    if (!IsDialogMessageW(hwnd, &msg)) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
  }

  if (state.font) DeleteObject(state.font);
  if (state.font_small) DeleteObject(state.font_small);
  g_config_path = nullptr;
  return state.play;
}

}  // namespace rex::ui
