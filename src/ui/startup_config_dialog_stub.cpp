/**
 * @file        ui/startup_config_dialog_stub.cpp
 * @brief       Non-Windows fallback for the startup configuration dialog.
 *
 * No native dialog on these platforms (yet) — proceed straight to launch using
 * cvar/config values supplied on the command line or config file.
 *
 * @copyright   Copyright (c) 2026 Tom Clay
 * @license     BSD 3-Clause License
 */

#include <rex/ui/startup_config_dialog.h>

namespace rex::ui {

bool ShowStartupConfigDialog(std::string_view /*app_name*/,
                             const std::filesystem::path& /*config_path*/,
                             const std::filesystem::path& /*content_root*/) {
  return true;
}

}  // namespace rex::ui
