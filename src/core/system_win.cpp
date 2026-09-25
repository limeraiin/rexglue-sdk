/**
 * @file        core/system_win.cpp
 * @brief       Windows implementations of rex/system.h platform helpers.
 *
 * @copyright   Copyright (c) 2026 Tom Clay
 * @license     BSD 3-Clause License
 */

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <rex/string.h>
#include <string>

#include <rex/system.h>

namespace rex {

namespace {
std::string& ProductNameStorage() {
  static std::string name = "Game";
  return name;
}
}  // namespace

void SetProductName(std::string_view name) { ProductNameStorage().assign(name); }
std::string_view ProductName() { return ProductNameStorage(); }

void ShowSimpleMessageBox(SimpleMessageBoxType type, std::string_view message) {
  UINT flags = MB_OK | MB_TOPMOST | MB_SETFOREGROUND;
  auto title_w = rex::string::to_utf16(ProductName());
  const wchar_t* title = reinterpret_cast<const wchar_t*>(title_w.c_str());
  switch (type) {
    case SimpleMessageBoxType::Help:
      flags |= MB_ICONINFORMATION;
      break;
    case SimpleMessageBoxType::Warning:
      flags |= MB_ICONWARNING;
      break;
    case SimpleMessageBoxType::Error:
      flags |= MB_ICONERROR;
      break;
  }
  auto wide = rex::string::to_utf16(message);
  ::MessageBoxW(nullptr, reinterpret_cast<const wchar_t*>(wide.c_str()), title, flags);
}

}  // namespace rex
