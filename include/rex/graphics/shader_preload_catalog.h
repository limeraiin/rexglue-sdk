#pragma once
#include <cstdint>
#include <filesystem>

namespace rex::graphics {
struct ShaderPreloadCatalogResult {
  uint32_t shaders = 0;
  uint32_t pipelines = 0;
  uint32_t database_shaders = 0;
};
// Merge by content, not file size. Existing player discoveries are retained.
// The database comes from the player's extracted disc; the shipped catalog
// contains previously observed pipeline descriptions and their shader inputs.
ShaderPreloadCatalogResult PrepareShaderPreloadCatalog(
    const std::filesystem::path& shareable, const std::filesystem::path& catalog,
    const std::filesystem::path& database, uint32_t title_id);
}
