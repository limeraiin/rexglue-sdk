/**
 * @file        ui/shader_progress_stub.cpp
 * @brief       No progress window on platforms without a native implementation.
 *
 * @license     BSD 3-Clause License
 */

#include <rex/ui/shader_progress.h>

namespace rex::ui {

std::unique_ptr<ShaderProgressWindow> StartShaderProgressWindow(std::string /*title*/,
                                                                ShaderProgressPoll /*poll*/) {
  return nullptr;
}

}  // namespace rex::ui
