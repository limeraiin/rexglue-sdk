/**
 * @file        ui/rex_app.cpp
 * @brief       ReXApp implementation — compiled as part of the consumer executable
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <rex/rex_app.h>

#include <rex/cvar.h>
#include <rex/ui/flags.h>
#include <rex/kernel/crt/heap.h>
#include <rex/filesystem.h>
#include <rex/logging/sink.h>
#include <rex/logging.h>
#include <rex/ui/overlay/console_overlay.h>
#include <rex/ui/overlay/debug_overlay.h>
#include <rex/ui/overlay/settings_overlay.h>
#include <rex/graphics/graphics_system.h>
#if REX_HAS_VULKAN
#include <rex/graphics/vulkan/graphics_system.h>
#endif
#if REX_HAS_D3D12
#include <rex/graphics/d3d12/graphics_system.h>
#endif
#include <rex/audio/audio_system.h>
#include <rex/audio/sdl/sdl_audio_system.h>
#include <rex/input/input_system.h>
#include <rex/kernel/init.h>
#include <rex/system.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xthread.h>
#include <rex/ui/graphics_provider.h>
#include <rex/ui/keybinds.h>
#include <rex/ui/startup_config_dialog.h>
#include <rex/version.h>

#include <fmt/format.h>
#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

// Graphics backend selection. Every cvar is also a launch flag, so this is
// selectable at startup as e.g. `--gpu vulkan`.
//   any    - auto-pick (prefers D3D12 on Windows, falls back to Vulkan)
//   d3d12  - force the Direct3D 12 backend
//   vulkan - force the Vulkan backend
// If the requested backend is unavailable or was not compiled into this build,
// the runtime falls back to the other one. kInitOnly: the backend is chosen
// once during startup and cannot change at runtime.
REXCVAR_DEFINE_STRING(gpu, "any", "GPU", "Graphics backend: any, d3d12, vulkan")
    .allowed({"any", "d3d12", "vulkan"})
    .lifecycle(::rex::cvar::Lifecycle::kInitOnly);

// Skip the native pre-launch configuration dialog and start the game directly
// using the current cvar/config/CLI values. Set true for automated/headless
// launches (e.g. --skip_config_dialog true).
REXCVAR_DEFINE_BOOL(skip_config_dialog, false, "UI",
                    "Skip the pre-launch configuration dialog and start immediately");

// Path to an STFS content package (DLC .LIVE file) to import once at boot. When
// set, the runtime mounts the package and extracts it into the user data content
// tree (<user_data>/0000000000000000/<title_id>/00000002/<name>/ + .header) so
// the guest's XamContentCreateEnumerator finds it like installed HDD content.
// Idempotent: an already-populated destination is skipped, so leaving the path set
// costs nothing on later boots. Packages for another title, or ones carrying guest
// code the recompiler never processed, are rejected (see InstallContentPackage).
//
// This is the headless/CLI import path. Interactive users install and remove DLC
// from the pre-launch dialog's DLC panel, which does the work immediately and
// leaves this cvar alone.
REXCVAR_DEFINE_STRING(install_content, "", "Runtime",
                      "Import an STFS DLC package at boot (path to the .LIVE file)");

namespace rex {

namespace {

// Builds the graphics backend chosen by the `gpu` cvar. Lives here (rather than
// the old compile-time #if/#elif) so a single binary built with both backends
// can switch between them at launch.
std::unique_ptr<system::IGraphicsSystem> CreateConfiguredGraphicsBackend() {
  std::string backend = REXCVAR_GET(gpu);
  std::transform(backend.begin(), backend.end(), backend.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

#if REX_HAS_D3D12 && REX_HAS_VULKAN
  const bool d3d12_ok = graphics::d3d12::D3D12GraphicsSystem::IsAvailable();
  const bool vulkan_ok = graphics::vulkan::VulkanGraphicsSystem::IsAvailable();

  if (backend == "vulkan") {
    if (vulkan_ok) {
      REXLOG_INFO("Graphics backend: Vulkan (requested via --gpu)");
      return std::make_unique<graphics::vulkan::VulkanGraphicsSystem>();
    }
    REXLOG_WARN("Vulkan backend requested but unavailable; falling back to D3D12");
    return std::make_unique<graphics::d3d12::D3D12GraphicsSystem>();
  }
  if (backend == "d3d12") {
    if (d3d12_ok) {
      REXLOG_INFO("Graphics backend: D3D12 (requested via --gpu)");
      return std::make_unique<graphics::d3d12::D3D12GraphicsSystem>();
    }
    REXLOG_WARN("D3D12 backend requested but unavailable; falling back to Vulkan");
    return std::make_unique<graphics::vulkan::VulkanGraphicsSystem>();
  }
  // "any" (default) or anything unrecognized: prefer D3D12 on Windows, then Vulkan.
  if (backend != "any") {
    REXLOG_WARN("Unknown gpu backend '{}'; using auto-selection", backend);
  }
  if (d3d12_ok) {
    REXLOG_INFO("Graphics backend: D3D12 (auto)");
    return std::make_unique<graphics::d3d12::D3D12GraphicsSystem>();
  }
  REXLOG_INFO("Graphics backend: Vulkan (auto)");
  return std::make_unique<graphics::vulkan::VulkanGraphicsSystem>();
#elif REX_HAS_D3D12
  if (backend == "vulkan") {
    REXLOG_WARN("Vulkan backend requested but not compiled into this build; using D3D12");
  }
  REXLOG_INFO("Graphics backend: D3D12 (only backend compiled in)");
  return std::make_unique<graphics::d3d12::D3D12GraphicsSystem>();
#elif REX_HAS_VULKAN
  if (backend == "d3d12") {
    REXLOG_WARN("D3D12 backend requested but not compiled into this build; using Vulkan");
  }
  REXLOG_INFO("Graphics backend: Vulkan (only backend compiled in)");
  return std::make_unique<graphics::vulkan::VulkanGraphicsSystem>();
#else
  REXLOG_ERROR("No graphics backend was compiled into this build");
  return nullptr;
#endif
}

}  // namespace

// --- ReXApp ---

ReXApp::~ReXApp() = default;

ReXApp::ReXApp(ui::WindowedAppContext& ctx, std::string_view name, PPCImageInfo ppc_info,
               std::string_view usage)
    : WindowedApp(ctx, name, usage), ppc_info_(ppc_info) {}

bool ReXApp::OnInitialize() {
  if (!SetupEnvironment())
    return false;
  if (!SetupPresentation())
    return false;

  auto paths = OnFinalizePaths(resolved_defaults_, MakeResumeCallback());
  if (!paths) {
    // Async: consumer will invoke resume when ready. OnInitialize returns
    // true so the event loop keeps pumping (wizard dialogs render).
    return true;
  }

  if (!ConstructRuntime(*paths))
    return false;
  LaunchModule();
  return true;
}

std::string ReXApp::GetDisplayName() const {
  // Default: app name + SDK build stamp. Apps override for a clean product name.
  return std::string(GetName()) + " " + REXGLUE_BUILD_TITLE;
}

bool ReXApp::SetupEnvironment() {
  auto exe_dir = rex::filesystem::GetExecutableFolder();

  // Native pre-launch configuration dialog (graphics backend, vsync, resolution,
  // fullscreen, game data folder, installed DLC). Load any persisted config first
  // so the dialog shows the user's previous choices, then let it override the
  // cvars and save them. Returning false here (Quit) exits the app before any
  // window/graphics are created. Skipped with --skip_config_dialog true
  // (automated launches).
  {
    std::filesystem::path startup_config = exe_dir / (std::string(GetName()) + ".toml");
    if (std::filesystem::exists(startup_config)) {
      rex::cvar::LoadConfig(startup_config);
    }
    if (!REXCVAR_GET(skip_config_dialog)) {
      // The dialog's DLC panel installs/removes content directly, so it needs the
      // user data root up front. Resolved the same way as below; note this is the
      // pre-OnConfigurePaths value, so an app that relocates user data in that hook
      // would need to surface it here too.
      std::string user_data_cvar = REXCVAR_GET(user_data_root);
      std::filesystem::path dialog_content_root =
          user_data_cvar.empty() ? rex::filesystem::GetUserFolder() / GetName()
                                 : std::filesystem::path(user_data_cvar);
      if (!rex::ui::ShowStartupConfigDialog(GetDisplayName(), startup_config,
                                            dialog_content_root)) {
        return false;
      }
    }
  }

  std::filesystem::path game_dir;
  std::string game_data_cvar = REXCVAR_GET(game_data_root);
  if (!game_data_cvar.empty()) {
    game_dir = game_data_cvar;
  }

  // User data: cvar override, or platform user directory
  std::filesystem::path user_dir;
  std::string user_data_cvar = REXCVAR_GET(user_data_root);
  if (!user_data_cvar.empty()) {
    user_dir = user_data_cvar;
  } else {
    user_dir = rex::filesystem::GetUserFolder() / GetName();
  }

  // Update data: cvar override, or empty (opt-in)
  std::filesystem::path update_dir;
  std::string update_data_cvar = REXCVAR_GET(update_data_root);
  if (!update_data_cvar.empty()) {
    update_dir = update_data_cvar;
  }

  // Cache: cvar override, or user_dir/cache
  std::filesystem::path cache_dir;
  std::string cache_path_cvar = REXCVAR_GET(cache_path);
  if (!cache_path_cvar.empty()) {
    cache_dir = cache_path_cvar;
  } else {
    cache_dir = user_dir / "cache";
  }

  PathConfig path_config{game_dir, user_dir, update_dir, cache_dir,
                         exe_dir / (std::string(GetName()) + ".toml")};
  OnConfigurePaths(path_config);
  game_data_root_ = path_config.game_data_root;
  user_data_root_ = path_config.user_data_root;
  update_data_root_ = path_config.update_data_root;
  cache_root_ = path_config.cache_root;
  config_path_ = path_config.config_path;
  resolved_defaults_ = std::move(path_config);

  // Load config FIRST so log cvars have final values
  if (std::filesystem::exists(config_path_))
    rex::cvar::LoadConfig(config_path_);

  // Late-phase logging
  std::string log_file_cvar = REXCVAR_GET(log_file);
  std::string log_level_str = REXCVAR_GET(log_level);
  if (REXCVAR_GET(log_verbose) && log_level_str == "info")
    log_level_str = "trace";

  auto category_levels = rex::ParseCategoryLevelsFromConfig(config_path_);
  auto log_config = rex::BuildLogConfig(log_file_cvar.empty() ? nullptr : log_file_cvar.c_str(),
                                        log_level_str, category_levels);
  if (log_file_cvar.empty()) {
    log_config.app_name = std::string(GetName());
    log_config.log_dir = (exe_dir / "logs").string();
  }

  rex::InitLogging(log_config);
  rex::RegisterLogLevelCallback();

  log_sink_ = std::make_shared<rex::LogCaptureSink>();
  rex::AddSink(log_sink_);

  OnPostInitLogging();

  if (std::filesystem::exists(config_path_))
    REXLOG_INFO("Loaded config: {}", config_path_.filename().string());

  REXLOG_INFO("{} starting", GetName());
  if (!game_data_root_.empty()) {
    REXLOG_INFO("  Game directory: {}", game_data_root_.string());
  }
  if (!user_data_root_.empty()) {
    REXLOG_INFO("  User data:      {}", user_data_root_.string());
  }
  if (!update_data_root_.empty()) {
    REXLOG_INFO("  Update data:    {}", update_data_root_.string());
  }
  REXLOG_INFO("  Cache root:     {}", cache_root_.string());

  return true;
}

bool ReXApp::ConstructRuntime(const PathConfig& paths) {
  if (paths.game_data_root.empty()) {
    auto msg = std::string("--game_data_root was not provided.");
    REXLOG_ERROR("{}", msg);
    rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error, msg);
    return false;
  }
  if (!std::filesystem::is_directory(paths.game_data_root)) {
    auto msg = fmt::format("--game_data_root does not exist: {}", paths.game_data_root.string());
    REXLOG_ERROR("{}", msg);
    rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error, msg);
    return false;
  }

  runtime_ = std::make_unique<rex::Runtime>(paths.game_data_root, paths.user_data_root,
                                            paths.update_data_root, paths.cache_root);
  runtime_->set_app_context(&app_context());

  // Window and ImGui drawer already exist from SetupPresentation; publish them
  // to the runtime before Setup so hooks and native rendering see them.
  if (window_) {
    runtime_->set_display_window(window_.get());
  }
  if (imgui_drawer_) {
    runtime_->set_imgui_drawer(imgui_drawer_.get());
  }

  auto status = runtime_->Setup(ppc_info_, std::move(config_));
  if (XFAILED(status)) {
    REXLOG_ERROR("Runtime setup failed: {:08X}", status);
    return false;
  }

  if (window_ && runtime_->input_system()) {
    static_cast<rex::input::InputSystem*>(runtime_->input_system())->AttachWindow(window_.get());
  }

  if (ppc_info_.register_modules) {
    ppc_info_.register_modules(runtime_->kernel_state());
  }

  if (imgui_drawer_) {
    auto* input_sys = static_cast<rex::input::InputSystem*>(runtime_->input_system());
    if (input_sys) {
      input_sys->SetActiveCallback([this]() {
        if (!debug_overlay_ && !console_overlay_ && !settings_overlay_)
          return true;
        return !imgui_drawer_->GetIO().WantCaptureMouse;
      });
    }
  }

  std::string xex_image = "game:\\default.xex";
  OnLoadXexImage(xex_image);

  // Mirrors the game:\ / d:\ -> game_data_root mapping in Runtime::SetupVfs.
  {
    constexpr std::string_view kGameDevice = "game:\\";
    constexpr std::string_view kDDevice = "d:\\";
    std::string_view tail = xex_image;
    if (tail.starts_with(kGameDevice)) {
      tail.remove_prefix(kGameDevice.size());
    } else if (tail.starts_with(kDDevice)) {
      tail.remove_prefix(kDDevice.size());
    }
    std::string host_tail{tail};
    std::replace(host_tail.begin(), host_tail.end(), '\\', '/');
    auto xex_host = paths.game_data_root / host_tail;
    if (!std::filesystem::is_regular_file(xex_host)) {
      auto msg = fmt::format("Entrypoint XEX not found: {}", xex_host.string());
      REXLOG_ERROR("{}", msg);
      rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error, msg);
      return false;
    }
  }

  status = runtime_->LoadXexImage(xex_image);
  if (XFAILED(status)) {
    auto msg = fmt::format("Failed to load XEX ({}): {:08X}", xex_image, status);
    REXLOG_ERROR("{}", msg);
    rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error, msg);
    return false;
  }

  OnPostLoadXexImage();

  // Optional one-shot DLC import. The XEX is loaded (so title_id() is valid) but
  // the guest has not started, so the content tree is ready before the game's
  // first XamContentCreateEnumerator call.
  {
    std::string install_pkg = REXCVAR_GET(install_content);
    if (!install_pkg.empty()) {
      auto* ks = runtime_->kernel_state();
      auto* cm = ks ? ks->content_manager() : nullptr;
      if (cm) {
        X_RESULT ir = cm->InstallContent(rex::to_path(install_pkg));
        if (ir == X_ERROR_ALREADY_EXISTS) {
          REXLOG_INFO("DLC already installed, skipping import: {}", install_pkg);
        } else if (XSUCCEEDED(ir)) {
          REXLOG_INFO("DLC import succeeded (title {:08X}): {}", ks->title_id(), install_pkg);
        } else {
          REXLOG_ERROR("DLC import failed ({:08X}): {}", ir, install_pkg);
        }
      } else {
        REXLOG_ERROR("DLC import requested but content manager unavailable");
      }
    }
  }

  if (ppc_info_.rexcrt_heap) {
    if (!rex::kernel::crt::InitHeap(REXCVAR_GET(rexcrt_heap_size_mb), runtime_->memory())) {
      REXLOG_ERROR("Failed to initialize rexcrt heap");
      return false;
    }
  }

  OnPostSetup();

  return true;
}

bool ReXApp::SetupPresentation() {
  config_.graphics = CreateConfiguredGraphicsBackend();
  config_.audio_factory = REX_AUDIO_BACKEND(rex::audio::sdl::SDLAudioSystem);
  config_.input_factory = REX_INPUT_BACKEND(rex::input::CreateDefaultInputSystem);
  config_.kernel_init = rex::kernel::InitializeKernel;

  OnPreSetup(config_);

  if (config_.graphics) {
    X_STATUS status = config_.graphics->SetupPresentation(&app_context());
    if (XFAILED(status)) {
      REXLOG_ERROR("Graphics presentation setup failed: {:08X}", status);
      return false;
    }
  }

  // Create window
  window_ = rex::ui::Window::Create(app_context(), GetName(), 1280, 720);
  if (!window_) {
    REXLOG_ERROR("Failed to create window");
    return false;
  }

  // Window caption: the app's display name (clean product name if overridden,
  // else the build-stamped app name).
  window_->SetTitle(GetDisplayName());

  window_->AddListener(this);
  window_->AddInputListener(this, 0);

  if (REXCVAR_GET(fullscreen)) {
    window_->SetFullscreen(true);
  }
  window_->Open();

  auto* graphics_system = static_cast<rex::graphics::GraphicsSystem*>(config_.graphics.get());
  if (graphics_system && graphics_system->presenter()) {
    auto* presenter = graphics_system->presenter();
    auto* provider = graphics_system->provider();
    if (provider) {
      immediate_drawer_ = provider->CreateImmediateDrawer();
      if (immediate_drawer_) {
        immediate_drawer_->SetPresenter(presenter);
        imgui_drawer_ = std::make_unique<rex::ui::ImGuiDrawer>(
            window_.get(), 64, [this](ImFontAtlas* atlas) { OnConfigureFonts(atlas); });
        imgui_drawer_->SetPresenterAndImmediateDrawer(presenter, immediate_drawer_.get());
        rex::ui::RegisterBind("bind_debug_overlay", "F3", "Toggle debug overlay", [this] {
          if (debug_overlay_) {
            debug_overlay_.reset();
          } else {
            debug_overlay_ = std::make_unique<ui::DebugOverlayDialog>(imgui_drawer_.get(),
                                                                      frame_stats_provider_);
          }
        });
        rex::ui::RegisterBind("bind_console", "Backtick", "Toggle console overlay", [this] {
          if (console_overlay_) {
            console_overlay_.reset();
          } else {
            console_overlay_ = std::make_unique<ui::ConsoleDialog>(imgui_drawer_.get(), log_sink_);
          }
        });
        rex::ui::RegisterBind("bind_settings", "F4", "Toggle settings overlay", [this] {
          if (settings_overlay_) {
            settings_overlay_.reset();
          } else {
            settings_overlay_ =
                std::make_unique<ui::SettingsDialog>(imgui_drawer_.get(), config_path_);
          }
        });

        OnCreateDialogs(imgui_drawer_.get());
      }
    }
    window_->SetPresenter(presenter);
  }

  return true;
}

void ReXApp::LaunchModule() {
  app_context().CallInUIThreadDeferred([this]() {
    OnPreLaunchModule();

    auto main_thread = runtime_->PrepareModuleLaunch();
    if (!main_thread) {
      REXLOG_ERROR("Failed to launch module");
      app_context().QuitFromUIThread();
      return;
    }

    auto* graphics_system =
        static_cast<rex::graphics::GraphicsSystem*>(runtime_->graphics_system());
    if (graphics_system && !runtime_->cache_root().empty()) {
      uint32_t title_id = runtime_->kernel_state()->title_id();
      if (title_id != 0) {
        REXLOG_INFO("Initializing shader storage for title {:08X}...", title_id);
        graphics_system->InitializeShaderStorage(runtime_->cache_root(), title_id, true);
      }
    }

    OnPostLaunchModule(main_thread.get());
    main_thread->Resume();

    module_thread_ = std::thread([this, main_thread = std::move(main_thread)]() mutable {
      main_thread->Wait(0, 0, 0, nullptr);
      OnGuestThreadExit(main_thread.get());
      REXLOG_INFO("Execution complete");
      if (!shutting_down_.load(std::memory_order_acquire)) {
        app_context().CallInUIThread([this]() { app_context().QuitFromUIThread(); });
      }
    });
  });
}

std::function<void(PathConfig)> ReXApp::MakeResumeCallback() {
  return [this](PathConfig paths) {
    if (shutting_down_.load(std::memory_order_acquire))
      return;
    if (!ConstructRuntime(std::move(paths))) {
      app_context().QuitFromUIThread();
      return;
    }
    LaunchModule();
  };
}

void ReXApp::OnKeyDown(ui::KeyEvent& e) {
  rex::ui::ProcessKeyEvent(e);
}

void ReXApp::OnClosing(ui::UIEvent& e) {
  (void)e;
  REXLOG_INFO("Window closing, shutting down...");
  shutting_down_.store(true, std::memory_order_release);
  if (runtime_ && runtime_->kernel_state()) {
    runtime_->kernel_state()->TerminateTitle();
  }
  app_context().QuitFromUIThread();
}

void ReXApp::OnDestroy() {
  // Notify subclass before cleanup
  OnShutdown();

  // Unregister overlay keybinds before destroying dialogs
  rex::ui::UnregisterBind("bind_debug_overlay");
  rex::ui::UnregisterBind("bind_console");
  rex::ui::UnregisterBind("bind_settings");

  // ImGui cleanup (reverse of setup)
  settings_overlay_.reset();
  console_overlay_.reset();
  debug_overlay_.reset();
  if (imgui_drawer_) {
    imgui_drawer_->SetPresenterAndImmediateDrawer(nullptr, nullptr);
    imgui_drawer_.reset();
  }
  if (immediate_drawer_) {
    immediate_drawer_->SetPresenter(nullptr);
    immediate_drawer_.reset();
  }
  if (runtime_) {
    runtime_->set_display_window(nullptr);
    runtime_->set_imgui_drawer(nullptr);
  }
  // Window/runtime cleanup
  if (window_) {
    window_->SetPresenter(nullptr);
  }
  if (module_thread_.joinable()) {
    module_thread_.join();
  }
  if (window_) {
    window_->RemoveInputListener(this);
    window_->RemoveListener(this);
  }
  window_.reset();
  runtime_.reset();
}

void ReXApp::SetGuestFrameStats(ui::DebugOverlayDialog::FrameStatsProvider provider) {
  frame_stats_provider_ = provider;
  if (debug_overlay_) {
    debug_overlay_->SetStatsProvider(provider);
  }
}

}  // namespace rex
