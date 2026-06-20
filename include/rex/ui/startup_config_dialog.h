/**
 * @file        ui/startup_config_dialog.h
 * @brief       Native pre-launch configuration dialog.
 *
 * Shows a modal, native configuration window before the game window/graphics
 * are created, letting the user pick the graphics backend, vsync, resolution,
 * fullscreen and the game data folder. Values are read from and written back to
 * the cvar registry by name (so it works regardless of which module defines a
 * given cvar) and persisted to the consumer's config file.
 *
 * @copyright   Copyright (c) 2026 Tom Clay
 * @license     BSD 3-Clause License
 */

#pragma once

#include <filesystem>
#include <string_view>

namespace rex::ui {

// Shows the modal startup configuration dialog. `app_name` is used for the
// window title; `config_path` is the TOML file chosen settings are saved to.
//
// Returns true if the user chose to launch the game (settings have been applied
// to cvars and persisted), or false if the user chose to quit. On non-Windows
// platforms this is a no-op that returns true.
bool ShowStartupConfigDialog(std::string_view app_name, const std::filesystem::path& config_path);

}  // namespace rex::ui
