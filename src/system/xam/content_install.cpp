/**
 * @file        system/xam/content_install.cpp
 * @brief       Kernel-free inspection / install / uninstall of STFS content.
 *
 * @copyright   Copyright (c) 2026 Tom Clay
 * @license     BSD 3-Clause License
 */

#include <rex/system/xam/content_install.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <queue>
#include <span>

#include <fmt/format.h>

#include <rex/filesystem.h>
#include <rex/filesystem/devices/stfs_container_device.h>
#include <rex/filesystem/file.h>
#include <rex/string.h>
#include <rex/system/xam/content_device.h>
#include <rex/system/xam/content_manager.h>

namespace rex {
namespace system {
namespace xam {

namespace {

constexpr const char* kHeaderDirName = "Headers";
constexpr const char* kMarketplaceXuid = "0000000000000000";

std::string Utf8FromU16(const std::u16string& value) {
  return rex::path_to_utf8(rex::to_path(value));
}

std::string LowerAscii(std::string_view value) {
  std::string out(value);
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return out;
}

/// True for file names that are PowerPC guest modules. A DLC shipping one of
/// these replaces a module the static recompiler had to process ahead of time,
/// so the guest ends up calling into un-recompiled code and traps.
bool IsGuestModuleName(std::string_view name) {
  auto lower = LowerAscii(name);
  auto dot = lower.rfind('.');
  if (dot == std::string::npos) {
    return false;
  }
  auto ext = lower.substr(dot);
  return ext == ".dll" || ext == ".xex" || ext == ".xexp";
}

std::filesystem::path ContentDirectory(const std::filesystem::path& content_root, uint32_t title_id,
                                       uint32_t content_type, const std::string& file_name) {
  return content_root / kMarketplaceXuid / fmt::format("{:08X}", title_id) /
         fmt::format("{:08X}", content_type) / rex::to_path(file_name);
}

std::filesystem::path ContentHeaderPath(const std::filesystem::path& content_root,
                                        uint32_t title_id, uint32_t content_type,
                                        const std::string& file_name) {
  return content_root / kMarketplaceXuid / fmt::format("{:08X}", title_id) / kHeaderDirName /
         fmt::format("{:08X}", content_type) / rex::to_path(file_name + ".header");
}

uint64_t DirectorySize(const std::filesystem::path& path) {
  uint64_t total = 0;
  std::error_code ec;
  for (auto it = std::filesystem::recursive_directory_iterator(path, ec);
       !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    std::error_code size_ec;
    if (it->is_regular_file(size_ec)) {
      total += it->file_size(size_ec);
    }
  }
  return total;
}

/// Recursively copies one STFS entry out of a mounted package.
/// (Moved here from ContentManager so both install paths share it.)
X_RESULT ExtractEntry(rex::filesystem::Entry* entry, const std::filesystem::path& base_path) {
  auto dest_path = base_path / rex::to_path(rex::string::utf8_fix_path_separators(entry->path()));

  if (entry->attributes() & rex::filesystem::kFileAttributeDirectory) {
    std::error_code ec;
    std::filesystem::create_directories(dest_path, ec);
    if (ec) {
      return X_ERROR_ACCESS_DENIED;
    }
    return X_ERROR_SUCCESS;
  }

  std::error_code ec;
  std::filesystem::create_directories(dest_path.parent_path(), ec);

  rex::filesystem::File* in_file = nullptr;
  X_STATUS status = entry->Open(rex::filesystem::FileAccess::kFileReadData, &in_file);
  if (status != X_STATUS_SUCCESS) {
    return X_ERROR_ACCESS_DENIED;
  }

  auto out_file = rex::filesystem::OpenFile(dest_path, "wb");
  if (!out_file) {
    in_file->Destroy();
    return X_ERROR_ACCESS_DENIED;
  }

  constexpr size_t kBufferSize = 4 * 1024 * 1024;  // 4 MiB
  auto buffer = std::make_unique<uint8_t[]>(kBufferSize);
  size_t remaining = entry->size();
  size_t offset = 0;

  while (remaining > 0) {
    size_t bytes_read = 0;
    size_t to_read = std::min(remaining, kBufferSize);
    in_file->ReadSync(std::span<uint8_t>(buffer.get(), to_read), offset, &bytes_read);
    if (bytes_read == 0) {
      break;
    }
    fwrite(buffer.get(), 1, bytes_read, out_file);
    offset += bytes_read;
    remaining -= bytes_read;
  }

  fclose(out_file);
  in_file->Destroy();
  return X_ERROR_SUCCESS;
}

}  // namespace

bool InspectContentPackage(const std::filesystem::path& package_path,
                           ContentPackageInfo* out_info) {
  if (!out_info) {
    return false;
  }
  *out_info = ContentPackageInfo{};

  std::error_code ec;
  if (!std::filesystem::is_regular_file(package_path, ec)) {
    return false;
  }

  out_info->file_name = rex::path_to_utf8(package_path.filename());

  auto header = rex::filesystem::StfsContainerDevice::ReadPackageHeader(package_path);
  if (!header) {
    return false;
  }

  out_info->package_type = static_cast<uint32_t>(header->header.magic.get());
  out_info->content_type = static_cast<uint32_t>(header->metadata.content_type.get());
  out_info->title_id = header->metadata.execution_info.title_id;
  out_info->content_size = header->metadata.content_size;
  out_info->title_name = Utf8FromU16(header->metadata.title_name());

  auto display_name = header->metadata.display_name(XLanguage::kEnglish);
  out_info->display_name =
      display_name.empty() ? out_info->file_name : Utf8FromU16(display_name);

  for (size_t i = 0; i < 0x10; i++) {
    if (header->header.licenses[i].license_flags) {
      out_info->license_mask |= header->header.licenses[i].license_bits;
    }
  }

  // SVOD volumes are a different on-disk layout; the install path below only
  // handles STFS (which is what DLC packages use).
  if (header->metadata.volume_type != rex::filesystem::XContentVolumeType::kStfs) {
    return false;
  }
  out_info->valid = true;

  // Mount just far enough to read the file table, so callers can see what the
  // package actually contains (and spot guest modules before installing).
  auto device = std::make_unique<rex::filesystem::StfsContainerDevice>("", package_path);
  if (device->Initialize()) {
    if (auto* root = device->ResolvePath("")) {
      for (const auto& child : root->children()) {
        out_info->files.push_back(child->name());
        if (!(child->attributes() & rex::filesystem::kFileAttributeDirectory) &&
            IsGuestModuleName(child->name())) {
          out_info->has_guest_module = true;
          if (out_info->guest_module_name.empty()) {
            out_info->guest_module_name = child->name();
          }
        }
      }
    }
  }

  return true;
}

bool ContentDirectoryHasGuestModule(const std::filesystem::path& directory,
                                    std::string* out_module_name) {
  std::error_code ec;
  for (auto it = std::filesystem::recursive_directory_iterator(directory, ec);
       !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
    if (!it->is_regular_file(ec)) {
      continue;
    }
    auto name = rex::path_to_utf8(it->path().filename());
    if (IsGuestModuleName(name)) {
      if (out_module_name) {
        *out_module_name = name;
      }
      return true;
    }
  }
  return false;
}

std::vector<InstalledContentEntry> ListInstalledContent(const std::filesystem::path& content_root,
                                                       uint32_t title_id) {
  std::vector<InstalledContentEntry> result;
  if (content_root.empty()) {
    return result;
  }

  const uint32_t content_type = static_cast<uint32_t>(XContentType::kMarketplaceContent);
  const auto xuid_root = content_root / kMarketplaceXuid;

  std::error_code ec;
  if (!std::filesystem::is_directory(xuid_root, ec)) {
    return result;
  }

  // <xuid>/<title_id>/00000002/<content id>/
  // Iterated with the error_code overloads throughout: a stray unreadable
  // directory must not throw out of the launcher's list refresh.
  const std::filesystem::directory_iterator dir_end;
  for (auto title_it = std::filesystem::directory_iterator(xuid_root, ec);
       !ec && title_it != dir_end; title_it.increment(ec)) {
    const auto& title_dir = *title_it;
    std::error_code entry_ec;
    if (!title_dir.is_directory(entry_ec) || entry_ec) {
      continue;
    }
    uint32_t dir_title = 0;
    auto title_text = rex::path_to_utf8(title_dir.path().filename());
    if (title_text.size() != 8 ||
        title_text.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
      continue;
    }
    dir_title = static_cast<uint32_t>(std::strtoul(title_text.c_str(), nullptr, 16));
    if (title_id != 0 && dir_title != title_id) {
      continue;
    }

    auto type_dir = title_dir.path() / fmt::format("{:08X}", content_type);
    std::error_code type_ec;
    if (!std::filesystem::is_directory(type_dir, type_ec)) {
      continue;
    }

    for (auto content_it = std::filesystem::directory_iterator(type_dir, type_ec);
         !type_ec && content_it != dir_end; content_it.increment(type_ec)) {
      const auto& content_dir = *content_it;
      std::error_code content_ec;
      if (!content_dir.is_directory(content_ec) || content_ec) {
        continue;
      }

      InstalledContentEntry entry;
      entry.directory = content_dir.path();
      entry.file_name = rex::path_to_utf8(content_dir.path().filename());
      entry.title_id = dir_title;
      entry.content_type = content_type;
      entry.header_file = ContentHeaderPath(content_root, dir_title, content_type, entry.file_name);
      entry.size_bytes = DirectorySize(entry.directory);

      // Display name comes from the .header written at install time; fall back to
      // the content id (which is what pre-header installs show).
      entry.display_name = entry.file_name;
      std::error_code header_ec;
      if (std::filesystem::is_regular_file(entry.header_file, header_ec)) {
        if (auto* file = rex::filesystem::OpenFile(entry.header_file, "rb")) {
          XCONTENT_AGGREGATE_DATA data{};
          if (fread(&data, 1, sizeof(data), file) == sizeof(data)) {
            auto name = Utf8FromU16(data.display_name());
            if (!name.empty()) {
              entry.display_name = name;
            }
          }
          fclose(file);
        }
      }

      // Flag already-installed content that carries guest code, so the user can
      // see which entry is preventing the game from starting.
      entry.has_guest_module =
          ContentDirectoryHasGuestModule(entry.directory, &entry.guest_module_name);

      result.push_back(std::move(entry));
    }
  }

  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return LowerAscii(a.display_name) < LowerAscii(b.display_name);
  });
  return result;
}

const char* ContentInstallResultToString(ContentInstallResult result) {
  switch (result) {
    case ContentInstallResult::kSuccess:
      return "Installed.";
    case ContentInstallResult::kAlreadyInstalled:
      return "Already installed.";
    case ContentInstallResult::kPackageNotFound:
      return "Could not open that file.";
    case ContentInstallResult::kNotAContentPackage:
      return "Not an Xbox 360 content package.";
    case ContentInstallResult::kUnsupportedContentType:
      return "Not downloadable content (DLC).";
    case ContentInstallResult::kWrongTitle:
      return "This package is for a different game.";
    case ContentInstallResult::kUnsupportedGuestModule:
      return "Not supported. This DLC replaces game program code, which a native "
             "PC build cannot load, and would stop the game from starting.";
    case ContentInstallResult::kExtractFailed:
      return "Extract failed. Check free disk space.";
  }
  return "Unknown error.";
}

ContentInstallResult InstallContentPackage(const std::filesystem::path& package_path,
                                          const std::filesystem::path& content_root,
                                          uint32_t expected_title_id, bool allow_guest_module,
                                          std::filesystem::path* out_install_dir) {
  ContentPackageInfo info;
  if (!InspectContentPackage(package_path, &info)) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(package_path, ec)) {
      return ContentInstallResult::kPackageNotFound;
    }
    return ContentInstallResult::kNotAContentPackage;
  }

  if (info.content_type != static_cast<uint32_t>(XContentType::kMarketplaceContent)) {
    return ContentInstallResult::kUnsupportedContentType;
  }
  if (expected_title_id != 0 && info.title_id != expected_title_id) {
    return ContentInstallResult::kWrongTitle;
  }
  if (info.has_guest_module && !allow_guest_module) {
    return ContentInstallResult::kUnsupportedGuestModule;
  }

  auto install_dir = ContentDirectory(content_root, info.title_id, info.content_type, info.file_name);
  if (out_install_dir) {
    *out_install_dir = install_dir;
  }

  // Cheap re-check: a populated destination means this package was imported on
  // an earlier run, so nothing to do.
  std::error_code exists_ec;
  if (std::filesystem::is_directory(install_dir, exists_ec) &&
      !std::filesystem::is_empty(install_dir, exists_ec)) {
    return ContentInstallResult::kAlreadyInstalled;
  }

  auto device = std::make_unique<rex::filesystem::StfsContainerDevice>("", package_path);
  if (!device->Initialize()) {
    return ContentInstallResult::kNotAContentPackage;
  }
  auto* root = device->ResolvePath("");
  if (!root) {
    return ContentInstallResult::kNotAContentPackage;
  }

  std::error_code ec;
  std::filesystem::create_directories(install_dir, ec);
  if (ec) {
    return ContentInstallResult::kExtractFailed;
  }

  std::queue<rex::filesystem::Entry*> queue;
  queue.push(root);
  while (!queue.empty()) {
    auto* entry = queue.front();
    queue.pop();
    for (auto& child : entry->children()) {
      queue.push(child.get());
    }
    if (ExtractEntry(entry, install_dir) != X_ERROR_SUCCESS) {
      return ContentInstallResult::kExtractFailed;
    }
  }

  // Header file: the runtime's content enumerator reads display name/title/type
  // from here, so DLC shows up with its proper name.
  XCONTENT_AGGREGATE_DATA data{};
  data.device_id = static_cast<uint32_t>(DummyDeviceId::HDD);
  data.content_type = XContentType::kMarketplaceContent;
  data.title_id = info.title_id;
  data.xuid = 0;
  data.set_file_name(info.file_name);
  data.set_display_name(rex::path_to_utf16(rex::to_path(info.display_name)));

  auto header_path = ContentHeaderPath(content_root, info.title_id, info.content_type,
                                       info.file_name);
  std::error_code header_ec;
  std::filesystem::create_directories(header_path.parent_path(), header_ec);
  if (header_ec) {
    return ContentInstallResult::kExtractFailed;
  }

  auto* file = rex::filesystem::OpenFile(header_path, "wb");
  if (!file) {
    return ContentInstallResult::kExtractFailed;
  }
  fwrite(&data, 1, sizeof(data), file);
  if (info.license_mask != 0) {
    fwrite(&info.license_mask, 1, sizeof(info.license_mask), file);
  }
  fclose(file);

  return ContentInstallResult::kSuccess;
}

bool UninstallContent(const InstalledContentEntry& entry) {
  std::error_code dir_ec;
  auto removed = std::filesystem::remove_all(entry.directory, dir_ec);

  std::error_code header_ec;
  bool header_removed = std::filesystem::remove(entry.header_file, header_ec);

  return (!dir_ec && removed > 0) || header_removed;
}

bool ReadXexTitleId(const std::filesystem::path& xex_path, uint32_t* out_title_id) {
  if (!out_title_id) {
    return false;
  }

  auto* file = rex::filesystem::OpenFile(xex_path, "rb");
  if (!file) {
    return false;
  }

  auto read_be32 = [](FILE* f, long offset, uint32_t* out) {
    if (fseek(f, offset, SEEK_SET) != 0) {
      return false;
    }
    uint8_t raw[4];
    if (fread(raw, 1, sizeof(raw), f) != sizeof(raw)) {
      return false;
    }
    *out = (uint32_t(raw[0]) << 24) | (uint32_t(raw[1]) << 16) | (uint32_t(raw[2]) << 8) |
           uint32_t(raw[3]);
    return true;
  };

  bool found = false;
  uint32_t magic = 0;
  uint32_t header_count = 0;
  if (read_be32(file, 0x00, &magic) && magic == 0x58455832u /* 'XEX2' */ &&
      read_be32(file, 0x14, &header_count) && header_count < 0x1000) {
    // Optional header directory: { key, value } pairs. For keys whose low byte is
    // > 1 the value is a file offset to the payload; XEX_HEADER_EXECUTION_INFO
    // (0x00040006) points at a struct whose title id sits at +0xC.
    for (uint32_t i = 0; i < header_count && !found; i++) {
      uint32_t key = 0;
      uint32_t value = 0;
      long entry_offset = 0x18 + long(i) * 8;
      if (!read_be32(file, entry_offset, &key) || !read_be32(file, entry_offset + 4, &value)) {
        break;
      }
      if (key == 0x00040006u) {
        found = read_be32(file, long(value) + 0xC, out_title_id);
      }
    }
  }

  fclose(file);
  return found;
}

}  // namespace xam
}  // namespace system
}  // namespace rex
