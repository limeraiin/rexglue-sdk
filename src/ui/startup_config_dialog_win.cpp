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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include <rex/cvar.h>
#include <rex/system/xam/content_install.h>

// clang-format off
#include <windows.h>
#include <shlobj.h>   // SHBrowseForFolderW, SHGetPathFromIDListW
#include <commdlg.h>  // GetOpenFileNameW (DLC file picker)
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
constexpr int IDC_DLC_LIST = 1009;
constexpr int IDC_DLC_INSTALL = 1010;
constexpr int IDC_SKIP_LAUNCHER = 1011;
constexpr int IDC_DLC_UNINSTALL = 1012;
constexpr int IDC_GPU_INSTANCE = 1013;

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
  HWND check_gpu_instance = nullptr;
  HWND combo_display = nullptr;
  HWND edit_data = nullptr;
  HWND list_dlc = nullptr;
  HWND check_skip_launcher = nullptr;
  HFONT font = nullptr;
  HFONT font_small = nullptr;  // smaller font for the bottom-left credits line

  // Installed-DLC panel state. `content_root` is the user data root the runtime
  // will use; empty means the panel is unavailable (the caller didn't supply
  // one), in which case the list is disabled rather than lying to the user.
  std::filesystem::path content_root;
  std::filesystem::path config_path;  // for persisting DLC changes immediately
  std::vector<rex::system::xam::InstalledContentEntry> dlc_entries;
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

// Opens a standard file-open dialog to pick a single DLC package. Xbox 360
// content packages usually have no file extension (the on-disc name is a content
// hash), so the filter defaults to all files. Returns "" if cancelled.
std::wstring BrowseForFile(HWND owner, const std::wstring& initial) {
  wchar_t path[MAX_PATH] = {};
  if (!initial.empty() && initial.size() < MAX_PATH) {
    wcscpy_s(path, initial.c_str());
  }
  OPENFILENAMEW ofn{};
  ofn.lStructSize = sizeof(ofn);
  ofn.hwndOwner = owner;
  ofn.lpstrFilter = L"DLC / content package (*.*)\0*.*\0\0";
  ofn.lpstrFile = path;
  ofn.nMaxFile = MAX_PATH;
  ofn.lpstrTitle = L"Select a DLC package file";
  ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
  if (GetOpenFileNameW(&ofn)) {
    return path;
  }
  return {};
}

// --- Installed-DLC panel -----------------------------------------------------

// Reads the title id of the game the user has selected, so DLC for a different
// game can be rejected. The runtime isn't up yet, so this comes straight from the
// XEX image in the game data folder. Returns 0 when it can't be determined, in
// which case the title check is skipped (the package's own title id still decides
// where it is installed).
uint32_t DetectGameTitleId(const std::wstring& game_data_dir) {
  if (game_data_dir.empty()) {
    return 0;
  }
  std::filesystem::path dir(game_data_dir);
  uint32_t title_id = 0;

  std::error_code ec;
  auto default_xex = dir / "default.xex";
  if (std::filesystem::is_regular_file(default_xex, ec) &&
      rex::system::xam::ReadXexTitleId(default_xex, &title_id)) {
    return title_id;
  }

  // Fall back to any .xex sitting in the folder root.
  const std::filesystem::directory_iterator end;
  for (auto it = std::filesystem::directory_iterator(dir, ec); !ec && it != end; it.increment(ec)) {
    std::error_code entry_ec;
    if (!it->is_regular_file(entry_ec) || entry_ec) {
      continue;
    }
    auto ext = it->path().extension().string();
    if (_stricmp(ext.c_str(), ".xex") == 0 &&
        rex::system::xam::ReadXexTitleId(it->path(), &title_id)) {
      return title_id;
    }
  }
  return 0;
}

// The file name of the package the `install_content` cvar points at (that cvar is
// the legacy/CLI import path: the runtime imports it once at boot). Empty if unset.
std::string PendingInstallFileName() {
  std::string pending = rex::cvar::GetFlagByName("install_content");
  if (pending.empty()) {
    return {};
  }
  return std::filesystem::path(pending).filename().string();
}

// Drops a queued boot-time import of `file_name`, if that is what install_content
// points at, and persists it right away. Saving here (rather than waiting for
// Play) matters: otherwise uninstalling and then quitting would leave the stale
// path in the config, and the next launch would silently re-install the DLC that
// was just removed.
void ClearPendingInstallFor(DialogState* st, const std::string& file_name) {
  auto pending = PendingInstallFileName();
  if (pending.empty() || _stricmp(pending.c_str(), file_name.c_str()) != 0) {
    return;
  }
  rex::cvar::SetFlagByName("install_content", "");
  if (!st->config_path.empty()) {
    rex::cvar::SaveConfig(st->config_path);
  }
}

void UpdateDlcButtons(HWND hwnd, DialogState* st) {
  bool have_root = !st->content_root.empty();
  int sel = st->list_dlc ? int(SendMessageW(st->list_dlc, LB_GETCURSEL, 0, 0)) : LB_ERR;
  bool can_uninstall =
      have_root && sel != LB_ERR && sel >= 0 && sel < int(st->dlc_entries.size());
  if (HWND btn = GetDlgItem(hwnd, IDC_DLC_UNINSTALL)) {
    EnableWindow(btn, can_uninstall ? TRUE : FALSE);
  }
  if (HWND btn = GetDlgItem(hwnd, IDC_DLC_INSTALL)) {
    EnableWindow(btn, have_root ? TRUE : FALSE);
  }
}

void RefreshDlcList(HWND hwnd, DialogState* st) {
  if (!st->list_dlc) {
    return;
  }
  SendMessageW(st->list_dlc, LB_RESETCONTENT, 0, 0);
  st->dlc_entries.clear();

  if (st->content_root.empty()) {
    SendMessageW(st->list_dlc, LB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>(L"(DLC folder unavailable)"));
    EnableWindow(st->list_dlc, FALSE);
    UpdateDlcButtons(hwnd, st);
    return;
  }

  const uint32_t game_title = DetectGameTitleId(GetEditText(st->edit_data));
  st->dlc_entries = rex::system::xam::ListInstalledContent(st->content_root);

  for (const auto& entry : st->dlc_entries) {
    std::wstring label = Widen(entry.display_name);
    uint64_t mb = (entry.size_bytes + 512 * 1024) / (1024 * 1024);
    label += L"  (" + std::to_wstring(mb) + L" MB)";
    if (entry.has_guest_module) {
      label += L"  [UNSUPPORTED]";
    } else if (game_title != 0 && entry.title_id != game_title) {
      label += L"  [other game]";
    }
    SendMessageW(st->list_dlc, LB_ADDSTRING, 0, reinterpret_cast<LPARAM>(label.c_str()));
  }

  if (st->dlc_entries.empty()) {
    SendMessageW(st->list_dlc, LB_ADDSTRING, 0, reinterpret_cast<LPARAM>(L"(none installed)"));
  }
  EnableWindow(st->list_dlc, TRUE);
  UpdateDlcButtons(hwnd, st);
}

void DoInstallDlc(HWND hwnd, DialogState* st) {
  if (st->content_root.empty()) {
    return;
  }
  std::wstring picked = BrowseForFile(hwnd, L"");
  if (picked.empty()) {
    return;
  }
  std::filesystem::path package(picked);

  rex::system::xam::ContentPackageInfo info;
  if (!rex::system::xam::InspectContentPackage(package, &info)) {
    MessageBoxW(hwnd,
                L"Not an Xbox 360 content package.\n\n"
                L"Pick the downloaded DLC file itself. It has no file extension.",
                L"Not a DLC package", MB_OK | MB_ICONWARNING);
    return;
  }

  const uint32_t game_title = DetectGameTitleId(GetEditText(st->edit_data));
  auto result = rex::system::xam::InstallContentPackage(package, st->content_root, game_title,
                                                       /*allow_guest_module=*/false);

  if (result == rex::system::xam::ContentInstallResult::kSuccess ||
      result == rex::system::xam::ContentInstallResult::kAlreadyInstalled) {
    // The launcher installs immediately, so a queued boot-time import of the same
    // package would be redundant work on every launch.
    ClearPendingInstallFor(st, info.file_name);
    RefreshDlcList(hwnd, st);
    std::wstring msg = Widen(info.display_name) + L"\n\n" +
                       Widen(rex::system::xam::ContentInstallResultToString(result));
    MessageBoxW(hwnd, msg.c_str(), L"DLC", MB_OK | MB_ICONINFORMATION);
    return;
  }

  std::wstring msg = Widen(info.display_name.empty() ? info.file_name : info.display_name) +
                     L"\n\n" + Widen(rex::system::xam::ContentInstallResultToString(result));
  if (result == rex::system::xam::ContentInstallResult::kUnsupportedGuestModule &&
      !info.guest_module_name.empty()) {
    msg += L"\n\n(It contains " + Widen(info.guest_module_name) + L".)";
  }
  if (result == rex::system::xam::ContentInstallResult::kWrongTitle) {
    wchar_t ids[128];
    swprintf_s(ids, L"\n\nPackage title id: %08X\nYour game: %08X", info.title_id, game_title);
    msg += ids;
  }
  MessageBoxW(hwnd, msg.c_str(), L"DLC not installed", MB_OK | MB_ICONWARNING);
}

void DoUninstallDlc(HWND hwnd, DialogState* st) {
  int sel = int(SendMessageW(st->list_dlc, LB_GETCURSEL, 0, 0));
  if (sel == LB_ERR || sel < 0 || sel >= int(st->dlc_entries.size())) {
    return;
  }
  const auto entry = st->dlc_entries[size_t(sel)];

  std::wstring prompt = L"Remove this DLC?\n\n" + Widen(entry.display_name) +
                        L"\n\nThe downloaded package file is not touched.";
  if (MessageBoxW(hwnd, prompt.c_str(), L"Uninstall DLC", MB_YESNO | MB_ICONQUESTION) != IDYES) {
    return;
  }

  bool removed = rex::system::xam::UninstallContent(entry);
  ClearPendingInstallFor(st, entry.file_name);
  RefreshDlcList(hwnd, st);
  if (!removed) {
    MessageBoxW(hwnd, L"Could not remove that DLC. The files may be in use or read-only.",
                L"Uninstall failed", MB_OK | MB_ICONWARNING);
  }
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
                  172, 154, 300, 22, IDC_EXP_60FPS, font);

  st->check_gpu_instance =
      MakeControl(hwnd, L"BUTTON", L"GPU draw instancing (EXPERIMENTAL)",
                  BS_AUTOCHECKBOX | WS_TABSTOP, 172, 180, 300, 22, IDC_GPU_INSTANCE, font);
  MakeControl(hwnd, L"STATIC",
              L"Direct3D 12 only. Turn this off if you get a black screen or flashing.",
              SS_LEFT, 190, 204, 282, 48, -1, st->font_small ? st->font_small : font);

  MakeControl(hwnd, L"STATIC", L"Game data folder:", SS_LEFT, 16, 258, 200, 20, -1, font);
  st->edit_data = MakeControl(hwnd, L"EDIT", nullptr,
                              WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, 16, 280, 330, 24, IDC_DATA,
                              font);
  MakeControl(hwnd, L"BUTTON", L"Browse...", WS_TABSTOP, 352, 279, 80, 26, IDC_BROWSE, font);

  // --- Right-hand panel: installed DLC ---
  MakeControl(hwnd, L"STATIC", L"Installed DLC:", SS_LEFT, 456, 18, 200, 20, -1, font);
  st->list_dlc = MakeControl(hwnd, L"LISTBOX", nullptr,
                             WS_BORDER | WS_VSCROLL | WS_TABSTOP | LBS_NOTIFY, 456, 40, 328, 216,
                             IDC_DLC_LIST, font);
  MakeControl(hwnd, L"BUTTON", L"Install DLC...", WS_TABSTOP, 456, 264, 156, 28, IDC_DLC_INSTALL,
              font);
  MakeControl(hwnd, L"BUTTON", L"Uninstall", WS_TABSTOP, 628, 264, 156, 28, IDC_DLC_UNINSTALL, font);
  MakeControl(hwnd, L"STATIC",
              L"Voice and audio packs work. Character packs are not supported.",
              SS_LEFT, 456, 298, 328, 60, -1, st->font_small ? st->font_small : font);

  // Skip this dialog on future launches (persists to the skip_config_dialog
  // cvar). Bottom-left, aligned with the Play/Quit buttons.
  st->check_skip_launcher =
      MakeControl(hwnd, L"BUTTON", L"Skip launcher next time", BS_AUTOCHECKBOX | WS_TABSTOP, 16, 338,
                  228, 22, IDC_SKIP_LAUNCHER, font);

  MakeControl(hwnd, L"BUTTON", L"Play", BS_DEFPUSHBUTTON | WS_TABSTOP, 256, 332, 84, 30, IDOK, font);
  MakeControl(hwnd, L"BUTTON", L"Quit", WS_TABSTOP, 348, 332, 84, 30, IDCANCEL, font);

  // Bottom credits line, drawn in a smaller font. Spans the full client width so
  // it can wrap if the system font is large.
  MakeControl(hwnd, L"STATIC",
              L"Special thanks to Simeon, Armin Suljovikj, ObsoleteSponge, Kalarot, Dante Smith, "
              L"Vexil Megga, cody russell, Hailnate13x, GUARD, ctrlalt3l1t3, Austin_Toonz, "
              L"Mark_Rampage, PELIODAS(Bubu), Sega The Hedgehog, Jesus Cantu, Enel, Chris Parnell "
              L"and eddie for their Patreon support.",
              SS_LEFT, 16, 374, 768, 60, -1, st->font_small ? st->font_small : font);

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

  // gpu_instance lives in the D3D12 command processor; addressing it by name means
  // the dialog doesn't care which module defines it. Persisting it from here also
  // stops SaveConfig from silently dropping a hand-written `gpu_instance = false`
  // (SerializeToTOML only writes values that differ from the default).
  SendMessageW(st->check_gpu_instance, BM_SETCHECK,
               rex::cvar::GetFlagByName("gpu_instance") == "true" ? BST_CHECKED : BST_UNCHECKED, 0);

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

  SendMessageW(st->check_skip_launcher, BM_SETCHECK,
               rex::cvar::GetFlagByName("skip_config_dialog") == "true" ? BST_CHECKED
                                                                        : BST_UNCHECKED,
               0);

  // Needs edit_data populated first: the list marks entries belonging to another
  // game, which is decided by the title id of the selected game data folder.
  RefreshDlcList(hwnd, st);
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

  rex::cvar::SetFlagByName("gpu_instance",
                           SendMessageW(st->check_gpu_instance, BM_GETCHECK, 0, 0) == BST_CHECKED
                               ? "true"
                               : "false");

  int display_sel = int(SendMessageW(st->combo_display, CB_GETCURSEL, 0, 0));
  if (display_sel < 0 || display_sel >= int(std::size(kDisplayModes))) display_sel = 0;
  rex::cvar::SetFlagByName("fullscreen", kDisplayModes[display_sel].fullscreen ? "true" : "false");
  rex::cvar::SetFlagByName("game_data_root", Narrow(data));

  // NB: `install_content` is deliberately left alone here. DLC is installed and
  // removed directly by the panel on the right, so that cvar is now only the
  // headless/CLI import path — the launcher must neither set nor wipe it.

  // Persist the "skip launcher next time" choice. If set, the next launch boots
  // straight into the game; clear skip_config_dialog in naruto.toml to get the
  // dialog back.
  rex::cvar::SetFlagByName("skip_config_dialog",
                           SendMessageW(st->check_skip_launcher, BM_GETCHECK, 0, 0) == BST_CHECKED
                               ? "true"
                               : "false");

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
        if (!picked.empty()) {
          SetWindowTextW(st->edit_data, picked.c_str());
          RefreshDlcList(hwnd, st);  // title id (and so the "other game" marks) changed
        }
        return 0;
      }
      if (id == IDC_DLC_INSTALL) {
        DoInstallDlc(hwnd, st);
        return 0;
      }
      if (id == IDC_DLC_UNINSTALL) {
        DoUninstallDlc(hwnd, st);
        return 0;
      }
      if (id == IDC_DLC_LIST) {
        if (HIWORD(wparam) == LBN_SELCHANGE) {
          UpdateDlcButtons(hwnd, st);
        }
        if (HIWORD(wparam) == LBN_DBLCLK) {
          DoUninstallDlc(hwnd, st);
        }
        return 0;
      }
      if (id == IDC_DATA && HIWORD(wparam) == EN_KILLFOCUS) {
        // A different game folder can change which entries count as "other game".
        RefreshDlcList(hwnd, st);
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

bool ShowStartupConfigDialog(std::string_view app_name, const std::filesystem::path& config_path,
                            const std::filesystem::path& content_root) {
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
  state.content_root = content_root;
  state.config_path = config_path;
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

  // Client area 800 x 444: settings column on the left (x 16..472), installed-DLC
  // panel on the right (x 456..784), credits strip along the bottom.
  const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
  RECT rc{0, 0, 800, 444};
  AdjustWindowRectEx(&rc, style, FALSE, 0);
  int win_w = rc.right - rc.left;
  int win_h = rc.bottom - rc.top;
  int x = (GetSystemMetrics(SM_CXSCREEN) - win_w) / 2;
  int y = (GetSystemMetrics(SM_CYSCREEN) - win_h) / 2;

  std::wstring title = Widen(std::string(app_name)) + L"  -  Setup";

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
