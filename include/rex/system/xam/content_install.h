/**
 * @file        system/xam/content_install.h
 * @brief       Kernel-free inspection / install / uninstall of STFS content.
 *
 * These helpers work purely on the host filesystem and the STFS container
 * reader, with no KernelState, no mounted VFS and no running guest. That lets
 * both the runtime (ContentManager::InstallContent, after the XEX is loaded) and
 * the pre-launch configuration dialog (which runs before anything else exists)
 * share one implementation of the DLC content tree.
 *
 * Layout produced/expected (matches ContentManager::ResolvePackagePath and
 * ResolvePackageHeaderPath for marketplace content, whose xuid is always 0):
 *
 *   <content_root>/0000000000000000/<title_id>/00000002/<package_file_name>/...
 *   <content_root>/0000000000000000/<title_id>/Headers/00000002/<name>.header
 *
 * NB: deliberately free of REXLOG calls — the dialog runs before logging is
 * configured, and results are surfaced to the user as message boxes instead.
 *
 * @copyright   Copyright (c) 2026 Tom Clay
 * @license     BSD 3-Clause License
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace rex {
namespace system {
namespace xam {

/// What a content package declares about itself, read from its STFS header.
struct ContentPackageInfo {
  bool valid = false;          ///< magic recognised and volume type is STFS
  uint32_t package_type = 0;   ///< 'LIVE' / 'PIRS' / 'CON '
  uint32_t content_type = 0;   ///< XContentType (0x2 == marketplace/DLC)
  uint32_t title_id = 0;       ///< title the content belongs to
  uint32_t license_mask = 0;   ///< OR of the header's license bits
  uint64_t content_size = 0;
  std::string title_name;      ///< e.g. "Naruto: Rise of a Ninja"
  std::string display_name;    ///< English display name, else the file name
  std::string file_name;       ///< package file name == content id
  std::vector<std::string> files;  ///< top-level entries inside the package

  /// True when the package ships PowerPC guest code (a .dll/.xex/.xexp). Such
  /// content cannot work under static recompilation: the module was never fed
  /// to the recompiler, so calling into it traps on the first instruction.
  bool has_guest_module = false;
  std::string guest_module_name;
};

/// Reads a package's header (and its file table, to fill `files`). Returns false
/// if the file is missing or is not a recognisable STFS content package.
bool InspectContentPackage(const std::filesystem::path& package_path, ContentPackageInfo* out_info);

/// True when an *installed* content directory contains PowerPC guest code, i.e.
/// content that cannot work under static recompilation. `out_module_name`
/// receives the offending file name when non-null.
bool ContentDirectoryHasGuestModule(const std::filesystem::path& directory,
                                    std::string* out_module_name = nullptr);

/// One installed DLC directory found under a content root.
struct InstalledContentEntry {
  std::filesystem::path directory;    ///< the extracted content directory
  std::filesystem::path header_file;  ///< matching .header (may not exist)
  std::string file_name;              ///< directory name == content id
  std::string display_name;           ///< from the .header, else file_name
  uint32_t title_id = 0;
  uint32_t content_type = 0;
  uint64_t size_bytes = 0;
  bool has_guest_module = false;  ///< see ContentPackageInfo::has_guest_module
  std::string guest_module_name;
};

/// Enumerates installed marketplace content (DLC) under `content_root`.
/// `title_id` of 0 lists every title found, so content belonging to a different
/// game still shows up (and can therefore be removed).
std::vector<InstalledContentEntry> ListInstalledContent(const std::filesystem::path& content_root,
                                                       uint32_t title_id = 0);

enum class ContentInstallResult {
  kSuccess,
  kAlreadyInstalled,
  kPackageNotFound,
  kNotAContentPackage,
  kUnsupportedContentType,  ///< not marketplace content (saves/themes/...)
  kWrongTitle,              ///< belongs to a different game
  kUnsupportedGuestModule,  ///< ships PPC code the recompiler never processed
  kExtractFailed,
};

/// Human-readable, user-facing explanation of an install result.
const char* ContentInstallResultToString(ContentInstallResult result);

/// Extracts `package_path` into the content tree under `content_root`.
///
/// `expected_title_id` of 0 skips the title check; otherwise a package for a
/// different title is rejected with kWrongTitle. Packages carrying guest code
/// are rejected with kUnsupportedGuestModule unless `allow_guest_module` is set.
/// Already-populated destinations return kAlreadyInstalled without re-extracting,
/// so this is cheap to call on every boot.
ContentInstallResult InstallContentPackage(const std::filesystem::path& package_path,
                                          const std::filesystem::path& content_root,
                                          uint32_t expected_title_id, bool allow_guest_module,
                                          std::filesystem::path* out_install_dir = nullptr);

/// Deletes an installed content directory and its .header. Returns true if
/// anything was removed.
bool UninstallContent(const InstalledContentEntry& entry);

/// Reads the title id from an XEX2 image's execution-info optional header.
/// Used to learn the running game's title id before the runtime exists.
bool ReadXexTitleId(const std::filesystem::path& xex_path, uint32_t* out_title_id);

}  // namespace xam
}  // namespace system
}  // namespace rex
