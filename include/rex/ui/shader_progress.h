/**
 * @file        ui/shader_progress.h
 * @brief       Small native "preparing shaders" progress window for the
 *              blocking shader storage boot pass (first launch on a PC).
 *
 * The window lives on its own thread so it keeps painting while the caller's
 * thread is blocked in InitializeShaderStorage. It only appears when the pass
 * takes longer than a moment, and closes when the returned handle is destroyed.
 *
 * @license     BSD 3-Clause License
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace rex::ui {

class ShaderProgressWindow {
 public:
  virtual ~ShaderProgressWindow() = default;
};

// Poll callback: fills phase (1 = loading shaders, total unknown; 2 = building
// pipelines, done/total) and returns false when no pass is running.
using ShaderProgressPoll = std::function<bool(uint32_t* phase, uint32_t* done, uint32_t* total)>;

// Returns null on platforms without an implementation.
std::unique_ptr<ShaderProgressWindow> StartShaderProgressWindow(std::string title,
                                                                ShaderProgressPoll poll);

}  // namespace rex::ui
