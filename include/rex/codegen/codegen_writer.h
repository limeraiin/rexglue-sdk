/**
 * @file        rex/codegen/codegen_writer.h
 * @brief       Consolidated codegen output writer
 *
 * @copyright   Copyright (c) 2026 Tom Clay <tomc@tctechstuff.com>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <fmt/core.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <rex/codegen/codegen_context.h>

namespace rex {
class Runtime;
}

namespace rex::codegen {

class CodegenWriter {
 public:
  CodegenWriter(CodegenContext& ctx, Runtime* runtime = nullptr);

  /// Run the full output pipeline: validate, clean old files, generate, flush.
  bool write(bool force);

  /**
   * Basenames of files removed by the pre-emit cleanup sweep during write().
   * Populated only after write() completes. Empty otherwise.
   */
  const std::vector<std::string>& deletedFiles() const { return deletedFiles_; }

  /**
   * Basenames of files written to disk during write() (via FlushPendingWrites).
   * Populated only after write() completes. Empty otherwise.
   */
  const std::vector<std::string>& writtenFiles() const { return writtenFiles_; }

 private:
  CodegenContext& ctx_;
  Runtime* runtime_;

  std::string out;
  size_t cppFileIndex = 0;
  std::vector<std::pair<std::string, std::string>> pendingWrites;
  std::vector<std::string> deletedFiles_;
  std::vector<std::string> writtenFiles_;

  template <class... Args>
  void print(fmt::format_string<Args...> fmt, Args&&... args) {
    fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
  }

  template <class... Args>
  void println(fmt::format_string<Args...> fmt, Args&&... args) {
    fmt::vformat_to(std::back_inserter(out), fmt.get(), fmt::make_format_args(args...));
    out += '\n';
  }

  /// The entry points of the register save/restore helpers, with their names.
  std::unordered_map<uint32_t, std::string> findRegisterHelpers() const;

  /// The functions that keep every register in the context although the
  /// *_as_local options are set: the configured ones and those found in the code.
  std::unordered_set<uint32_t> findContextRegisterFunctions(
      const std::vector<const FunctionNode*>& functions,
      const std::unordered_map<uint32_t, std::string>& registerHelpers) const;

  void SaveCurrentOutData(std::string_view name = {});
  void FlushPendingWrites();

  // Convenience accessors
  FunctionGraph& graph();
  const FunctionGraph& graph() const;
  const BinaryView& binary() const;
  RecompilerConfig& config();
  const RecompilerConfig& config() const;
  AnalysisState& analysisState();
  const AnalysisState& analysisState() const;
};

}  // namespace rex::codegen
