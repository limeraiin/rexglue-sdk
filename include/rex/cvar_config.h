#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace rex::cvar {
// Merge selected flags into the latest on-disk configuration.
bool SaveConfigValues(const std::filesystem::path& path, const std::vector<std::string>& names);
}
