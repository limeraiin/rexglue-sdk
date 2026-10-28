// Controller UI ported directly from sibling SVR11 launcher_win.cpp.
#include "startup_controller_image.inc"
enum Anchor { aLT, aLB, aLS, aDP, aBK, aRT, aRB, aY, aB, aA, aX, aRS, aST, aCount };

const POINT kAnchorPreview[aCount] = {
    {275, 6},    // LT
    {285, 24},   // LB
    {205, 72},   // left stick
    {238, 180},  // d-pad
    {308, 112},  // Back
    {515, 40},   // RT
    {520, 62},   // RB
    {548, 140},  // Y
    {575, 205},  // B
    {515, 225},  // A
    {488, 170},  // X
    {400, 235},  // right stick
    {428, 150},  // Start
};

constexpr int kAnchorPreviewW = 640;

// Photo pixel size (assets/pad360.png).
constexpr int kPadSrcW = 900;
constexpr int kPadSrcH = 819;

// ---- bindings ---------------------------------------------------------------

struct BindDef {
  const char* cvar;
  const wchar_t* label;
  const char* default_key;  // nin_system.cpp defaults
  Anchor anchor;
};

// Left column (label, key box) then right column (key box, label); the layout
// draws rows in this order. Consecutive rows with the same anchor share one
// pointer line (vertical bracket + line to the photo).
const BindDef kBinds[] = {
    // left column
    {"keybind_left_trigger", L"LT", "Shift", aLT},
    {"keybind_left_shoulder", L"LB", "Q", aLB},
    {"keybind_lstick_up", L"Left stick up", "W", aLS},
    {"keybind_lstick_down", L"Left stick down", "S", aLS},
    {"keybind_lstick_left", L"Left stick left", "A", aLS},
    {"keybind_lstick_right", L"Left stick right", "D", aLS},
    {"keybind_lstick_press", L"Left stick press", "F", aLS},
    {"keybind_dpad_up", L"D-pad up", "U", aDP},
    {"keybind_dpad_down", L"D-pad down", "J", aDP},
    {"keybind_dpad_left", L"D-pad left", "H", aDP},
    {"keybind_dpad_right", L"D-pad right", "K", aDP},
    {"keybind_back", L"Back", "Backspace", aBK},
    // right column
    {"keybind_right_trigger", L"RT", "Space", aRT},
    {"keybind_right_shoulder", L"RB", "E", aRB},
    {"keybind_y", L"Y", "V", aY},
    {"keybind_b", L"B", "X", aB},
    {"keybind_a", L"A", "Z", aA},
    {"keybind_x", L"X", "C", aX},
    {"keybind_rstick_up", L"Right stick up", "Up", aRS},
    {"keybind_rstick_down", L"Right stick down", "Down", aRS},
    {"keybind_rstick_left", L"Right stick left", "Left", aRS},
    {"keybind_rstick_right", L"Right stick right", "Right", aRS},
    {"keybind_rstick_press", L"Right stick press", "G", aRS},
    {"keybind_start", L"Start", "Return", aST},
};
constexpr int kBindCount = static_cast<int>(sizeof(kBinds) / sizeof(kBinds[0]));
constexpr int kLeftCount = 12;

// ---- key names ---------------------------------------------------------------

// VK <-> name. The names are exactly the ones rex::ui::ParseVirtualKey accepts
// (reference/rexglue-sdk-yukes/src/ui/keybinds.cpp kKeyNames); letters, digits,
// F-keys and numpad digits are generated. Kept local so the launcher does not
// depend on the SDK UI headers and so the name for a VK is deterministic (the
// SDK's VirtualKeyToString walks an unordered_map with two names for VK_RETURN).
struct KeyName {
  UINT vk;
  const char* name;
};

const KeyName kKeyNames[] = {
    {VK_OEM_3, "Backtick"},
    {VK_OEM_MINUS, "Minus"},
    {VK_OEM_PLUS, "Plus"},
    {VK_OEM_COMMA, "Comma"},
    {VK_OEM_PERIOD, "Period"},
    {VK_OEM_1, "Semicolon"},
    {VK_OEM_2, "Slash"},
    {VK_OEM_5, "Backslash"},
    {VK_OEM_4, "LBracket"},
    {VK_OEM_6, "RBracket"},
    {VK_OEM_7, "Quote"},
    {VK_ESCAPE, "Escape"},
    {VK_RETURN, "Return"},
    {VK_SPACE, "Space"},
    {VK_TAB, "Tab"},
    {VK_BACK, "Backspace"},
    {VK_DELETE, "Delete"},
    {VK_INSERT, "Insert"},
    {VK_HOME, "Home"},
    {VK_END, "End"},
    {VK_PRIOR, "PageUp"},
    {VK_NEXT, "PageDown"},
    {VK_LEFT, "Left"},
    {VK_RIGHT, "Right"},
    {VK_UP, "Up"},
    {VK_DOWN, "Down"},
    {VK_SHIFT, "Shift"},
    {VK_CONTROL, "Control"},
    {VK_MENU, "Alt"},
    {VK_ADD, "NumpadPlus"},
    {VK_SUBTRACT, "NumpadMinus"},
    {VK_MULTIPLY, "NumpadStar"},
    {VK_DIVIDE, "NumpadSlash"},
    {VK_SNAPSHOT, "PrintScreen"},
    {VK_PAUSE, "Pause"},
    {VK_CAPITAL, "CapsLock"},
    {VK_NUMLOCK, "NumLock"},
    {VK_SCROLL, "ScrollLock"},
};

// Empty when the key has no name the game's parser knows (Win keys, media keys...).
std::string KeyNameFromVk(UINT vk) {
  if ((vk >= 'A' && vk <= 'Z') || (vk >= '0' && vk <= '9')) {
    return std::string(1, static_cast<char>(vk));
  }
  if (vk >= VK_F1 && vk <= VK_F24) {
    return "F" + std::to_string(vk - VK_F1 + 1);
  }
  if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9) {
    return "Numpad" + std::to_string(vk - VK_NUMPAD0);
  }
  // Left/right variants never arrive in WM_KEYDOWN, but be safe.
  if (vk == VK_LSHIFT || vk == VK_RSHIFT)
    return "Shift";
  if (vk == VK_LCONTROL || vk == VK_RCONTROL)
    return "Control";
  if (vk == VK_LMENU || vk == VK_RMENU)
    return "Alt";
  for (const auto& k : kKeyNames) {
    if (k.vk == vk)
      return k.name;
  }
  return {};
}

struct KeyboardSettings {
  UINT dpi = 96;
  HINSTANCE hinst = nullptr;
  HFONT font = nullptr, font_bold = nullptr, font_small = nullptr;
  HWND canvas = nullptr;
  bool mnk = true;
  std::string binds[kBindCount];
  HBITMAP pad_normal = nullptr, pad_grey = nullptr;
  int pad_w = 0, pad_h = 0;
  RECT box_rects[kBindCount] = {}, label_rects[kBindCount] = {}, pad_rect = {};
  POINT anchors[aCount] = {};
  int bracket_x_left = 0, bracket_x_right = 0, status_y = 0;
  int capture = -1;
  std::wstring status;
};
int Scale(int px, UINT dpi) {
  return MulDiv(px, static_cast<int>(dpi), 96);
}
std::wstring Widen(const std::string& s);
bool LoadPadPixels(HINSTANCE hinst, int w, int h, std::vector<uint8_t>& out) {
  const void* data = kControllerPng;
  const DWORD size = sizeof(kControllerPng);
  (void)hinst;

  bool ok = false;
  IWICImagingFactory* factory = nullptr;
  IWICStream* stream = nullptr;
  IWICBitmapDecoder* decoder = nullptr;
  IWICBitmapFrameDecode* frame = nullptr;
  IWICBitmapScaler* scaler = nullptr;
  IWICFormatConverter* conv = nullptr;

  do {
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&factory)))) {
      break;
    }
    if (FAILED(factory->CreateStream(&stream)))
      break;
    if (FAILED(
            stream->InitializeFromMemory(reinterpret_cast<BYTE*>(const_cast<void*>(data)), size))) {
      break;
    }
    // PNG container; the decoder is picked from the stream signature (the PNG
    // codec, CLSID_WICPngDecoder, is what this resolves to for this asset).
    if (FAILED(factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnDemand,
                                                &decoder))) {
      break;
    }
    if (FAILED(decoder->GetFrame(0, &frame)))
      break;
    if (FAILED(factory->CreateBitmapScaler(&scaler)))
      break;
    if (FAILED(scaler->Initialize(frame, static_cast<UINT>(w), static_cast<UINT>(h),
                                  WICBitmapInterpolationModeFant))) {
      break;
    }
    if (FAILED(factory->CreateFormatConverter(&conv)))
      break;
    if (FAILED(conv->Initialize(scaler, GUID_WICPixelFormat32bppPBGRA, WICBitmapDitherTypeNone,
                                nullptr, 0.0, WICBitmapPaletteTypeCustom))) {
      break;
    }
    const UINT stride = static_cast<UINT>(w) * 4;
    out.assign(static_cast<size_t>(stride) * static_cast<size_t>(h), 0);
    if (FAILED(conv->CopyPixels(nullptr, stride, static_cast<UINT>(out.size()), out.data()))) {
      out.clear();
      break;
    }
    ok = true;
  } while (false);

  if (conv)
    conv->Release();
  if (scaler)
    scaler->Release();
  if (frame)
    frame->Release();
  if (decoder)
    decoder->Release();
  if (stream)
    stream->Release();
  if (factory)
    factory->Release();
  return ok;
}

HBITMAP MakeDib(int w, int h, const uint8_t* bgra) {
  BITMAPINFO bi = {};
  bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  bi.bmiHeader.biWidth = w;
  bi.bmiHeader.biHeight = -h;  // top-down
  bi.bmiHeader.biPlanes = 1;
  bi.bmiHeader.biBitCount = 32;
  bi.bmiHeader.biCompression = BI_RGB;
  void* bits = nullptr;
  HBITMAP bmp = CreateDIBSection(nullptr, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
  if (bmp && bits) {
    std::memcpy(bits, bgra, static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
  }
  return bmp;
}

// Greyed copy for the disabled state: desaturate + lighten (v = lum * 0.35 +
// 235 * 0.65) and halve the alpha, like the approved mockup.
void GreyPixels(std::vector<uint8_t>& px) {
  for (size_t i = 0; i + 3 < px.size(); i += 4) {
    const unsigned a = px[i + 3];
    if (a == 0) {
      px[i] = px[i + 1] = px[i + 2] = 0;
      continue;
    }
    // Un-premultiply, then luminance.
    const unsigned b = px[i] * 255 / a;
    const unsigned g = px[i + 1] * 255 / a;
    const unsigned r = px[i + 2] * 255 / a;
    const unsigned lum = (r * 30 + g * 59 + b * 11) / 100;
    const unsigned v = (lum * 35 + 235 * 65) / 100;
    const unsigned a2 = a / 2;
    const uint8_t c = static_cast<uint8_t>(v * a2 / 255);  // re-premultiply
    px[i] = px[i + 1] = px[i + 2] = c;
    px[i + 3] = static_cast<uint8_t>(a2);
  }
}

void LoadPadBitmaps(KeyboardSettings* st, int w, int h) {
  std::vector<uint8_t> px;
  if (!LoadPadPixels(st->hinst, w, h, px))
    return;
  st->pad_normal = MakeDib(w, h, px.data());
  GreyPixels(px);
  st->pad_grey = MakeDib(w, h, px.data());
  st->pad_w = w;
  st->pad_h = h;
}

// ---- canvas (Controls tab drawing) -------------------------------------------------

const wchar_t* const kCanvasClass = L"NarutoSetupCanvas";

// Computes the row / box / pointer geometry for the canvas client size.
void LayoutCanvas(KeyboardSettings* st, int W, int H) {
  const UINT dpi = st->dpi;
  auto S = [dpi](int v) { return Scale(v, dpi); };

  const int top = S(6);
  const int rowh = S(32);
  const int labw = S(108);
  const int keyw = S(72);
  const int hh = S(22);
  const int margin = S(12);

  for (int i = 0; i < kBindCount; ++i) {
    const bool left = i < kLeftCount;
    const int row = left ? i : i - kLeftCount;
    const int y = top + row * rowh;
    if (left) {
      st->label_rects[i] = {margin, y, margin + labw, y + hh};
      st->box_rects[i] = {margin + labw, y, margin + labw + keyw, y + hh};
    } else {
      const int bx = W - margin - labw - keyw;
      st->box_rects[i] = {bx, y, bx + keyw, y + hh};
      st->label_rects[i] = {bx + keyw + S(8), y, W - margin, y + hh};
    }
  }
  st->bracket_x_left = margin + labw + keyw + S(6);
  st->bracket_x_right = W - margin - labw - keyw - S(6);

  // Photo: centred between the columns, a little below the first rows.
  const int pw = S(340);
  const int ph = pw * kPadSrcH / kPadSrcW;
  const int px = (W - pw) / 2;
  const int py = top + S(60);
  st->pad_rect = {px, py, px + pw, py + ph};
  for (int a = 0; a < aCount; ++a) {
    st->anchors[a].x = px + kAnchorPreview[a].x * pw / kAnchorPreviewW;
    st->anchors[a].y = py + kAnchorPreview[a].y * pw / kAnchorPreviewW;
  }

  st->status_y = top + kLeftCount * rowh + S(6);
  (void)H;

  if (!st->pad_normal || st->pad_w != pw) {
    if (st->pad_normal)
      DeleteObject(st->pad_normal);
    if (st->pad_grey)
      DeleteObject(st->pad_grey);
    st->pad_normal = st->pad_grey = nullptr;
    LoadPadBitmaps(st, pw, ph);
  }
}

int HitTestBox(const KeyboardSettings* st, POINT p) {
  for (int i = 0; i < kBindCount; ++i) {
    if (PtInRect(&st->box_rects[i], p))
      return i;
  }
  return -1;
}

void CancelCapture(KeyboardSettings* st) {
  if (st->capture >= 0) {
    st->capture = -1;
    if (st->canvas)
      InvalidateRect(st->canvas, nullptr, FALSE);
  }
}

void PaintCanvas(KeyboardSettings* st, HDC hdc, const RECT& rc) {
  const UINT dpi = st->dpi;
  auto S = [dpi](int v) { return Scale(v, dpi); };
  const int W = rc.right - rc.left;
  const int H = rc.bottom - rc.top;
  const bool on = st->mnk;

  // Double buffer.
  HDC mem = CreateCompatibleDC(hdc);
  HBITMAP bmp = CreateCompatibleBitmap(hdc, W, H);
  HGDIOBJ old_bmp = SelectObject(mem, bmp);

  RECT full = {0, 0, W, H};
  FillRect(mem, &full, static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH)));
  SetBkMode(mem, TRANSPARENT);

  const COLORREF txt = on ? RGB(0, 0, 0) : RGB(150, 150, 150);
  const COLORREF line = on ? RGB(120, 120, 120) : RGB(190, 190, 190);
  const COLORREF box_edge = on ? RGB(70, 70, 70) : RGB(200, 200, 200);
  const COLORREF box_fill = on ? RGB(255, 255, 255) : RGB(238, 238, 238);
  const COLORREF cap_edge = RGB(0, 120, 215);
  const COLORREF cap_fill = RGB(204, 228, 255);

  // Photo.
  HBITMAP pad = on ? st->pad_normal : st->pad_grey;
  if (pad) {
    HDC pdc = CreateCompatibleDC(mem);
    HGDIOBJ old = SelectObject(pdc, pad);
    BLENDFUNCTION bf = {AC_SRC_OVER, 0, 255, AC_SRC_ALPHA};
    AlphaBlend(mem, st->pad_rect.left, st->pad_rect.top, st->pad_w, st->pad_h, pdc, 0, 0, st->pad_w,
               st->pad_h, bf);
    SelectObject(pdc, old);
    DeleteDC(pdc);
  }

  // Pointer lines: one per physical control (bracket over grouped rows, then a
  // line to the anchor, dot at the anchor).
  HPEN pen = CreatePen(PS_SOLID, 1, line);
  HBRUSH dot = CreateSolidBrush(line);
  HGDIOBJ old_pen = SelectObject(mem, pen);
  HGDIOBJ old_brush = SelectObject(mem, dot);
  for (int col = 0; col < 2; ++col) {
    const int first = col == 0 ? 0 : kLeftCount;
    const int last = col == 0 ? kLeftCount : kBindCount;
    const int bx = col == 0 ? st->bracket_x_left : st->bracket_x_right;
    int i = first;
    while (i < last) {
      int j = i;
      while (j + 1 < last && kBinds[j + 1].anchor == kBinds[i].anchor)
        ++j;
      const int y0 = (st->box_rects[i].top + st->box_rects[i].bottom) / 2;
      const int y1 = (st->box_rects[j].top + st->box_rects[j].bottom) / 2;
      if (j > i) {
        MoveToEx(mem, bx, y0, nullptr);
        LineTo(mem, bx, y1);
      }
      const POINT a = st->anchors[kBinds[i].anchor];
      MoveToEx(mem, bx, (y0 + y1) / 2, nullptr);
      LineTo(mem, a.x, a.y);
      const int r = S(3);
      Ellipse(mem, a.x - r, a.y - r, a.x + r + 1, a.y + r + 1);
      i = j + 1;
    }
  }
  SelectObject(mem, old_brush);
  SelectObject(mem, old_pen);
  DeleteObject(dot);
  DeleteObject(pen);

  // Labels and key boxes.
  HPEN box_pen = CreatePen(PS_SOLID, 1, box_edge);
  HBRUSH box_brush = CreateSolidBrush(box_fill);
  HPEN cap_pen = CreatePen(PS_SOLID, 1, cap_edge);
  HBRUSH cap_brush = CreateSolidBrush(cap_fill);
  for (int i = 0; i < kBindCount; ++i) {
    SelectObject(mem, st->font);
    SetTextColor(mem, txt);
    RECT lr = st->label_rects[i];
    DrawTextW(mem, kBinds[i].label, -1, &lr, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);

    const bool capturing = st->capture == i;
    HGDIOBJ op = SelectObject(mem, capturing ? cap_pen : box_pen);
    HGDIOBJ ob = SelectObject(mem, capturing ? cap_brush : box_brush);
    const RECT& b = st->box_rects[i];
    RoundRect(mem, b.left, b.top, b.right, b.bottom, S(6), S(6));
    SelectObject(mem, ob);
    SelectObject(mem, op);

    std::wstring key = capturing ? L"..." : Widen(st->binds[i]);
    if (key.empty() && !capturing)
      key = L"-";
    RECT tr = b;
    SelectObject(mem, st->font_bold);
    SIZE sz = {};
    GetTextExtentPoint32W(mem, key.c_str(), static_cast<int>(key.size()), &sz);
    if (sz.cx > (b.right - b.left) - S(6)) {
      SelectObject(mem, st->font_small);
    }
    SetTextColor(mem, capturing ? RGB(0, 60, 120) : txt);
    DrawTextW(mem, key.c_str(), -1, &tr, DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
  }
  DeleteObject(cap_brush);
  DeleteObject(cap_pen);
  DeleteObject(box_brush);
  DeleteObject(box_pen);

  // Status line.
  if (!st->status.empty()) {
    SelectObject(mem, st->font_small);
    SetTextColor(mem, RGB(90, 90, 90));
    RECT sr = {S(12), st->status_y, W - S(12), st->status_y + S(18)};
    DrawTextW(mem, st->status.c_str(), -1, &sr, DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX);
  }

  BitBlt(hdc, 0, 0, W, H, mem, 0, 0, SRCCOPY);
  SelectObject(mem, old_bmp);
  DeleteObject(bmp);
  DeleteDC(mem);
}

// Assigns `name` to the captured bind, clearing any other bind that used it.
void AssignCapturedKey(KeyboardSettings* st, const std::string& name) {
  const int i = st->capture;
  if (i < 0)
    return;
  st->status.clear();
  for (int j = 0; j < kBindCount; ++j) {
    if (j != i && st->binds[j] == name) {
      st->binds[j].clear();
      st->status =
          Widen(name) + L" was also used by " + kBinds[j].label + L"; that one is now unassigned.";
    }
  }
  st->binds[i] = name;
  st->capture = -1;
  InvalidateRect(st->canvas, nullptr, FALSE);
}

LRESULT CALLBACK CanvasProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
  if (msg == WM_CREATE) {
    auto* cs = reinterpret_cast<CREATESTRUCTW*>(lparam);
    SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
    return 0;
  }
  auto* st = reinterpret_cast<KeyboardSettings*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
  if (!st)
    return DefWindowProcW(hwnd, msg, wparam, lparam);

  switch (msg) {
    case WM_SIZE:
      LayoutCanvas(st, LOWORD(lparam), HIWORD(lparam));
      InvalidateRect(hwnd, nullptr, FALSE);
      return 0;
    case WM_ERASEBKGND:
      return 1;
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC hdc = BeginPaint(hwnd, &ps);
      RECT rc;
      GetClientRect(hwnd, &rc);
      PaintCanvas(st, hdc, rc);
      EndPaint(hwnd, &ps);
      return 0;
    }
    case WM_SETCURSOR: {
      if (st->mnk && LOWORD(lparam) == HTCLIENT) {
        POINT p;
        GetCursorPos(&p);
        ScreenToClient(hwnd, &p);
        if (HitTestBox(st, p) >= 0) {
          SetCursor(LoadCursor(nullptr, IDC_HAND));
          return TRUE;
        }
      }
      break;
    }
    case WM_LBUTTONDOWN: {
      if (!st->mnk)
        return 0;  // everything below the checkbox is inert when off
      POINT p = {static_cast<short>(LOWORD(lparam)), static_cast<short>(HIWORD(lparam))};
      const int hit = HitTestBox(st, p);
      st->status.clear();
      st->capture = hit;
      if (hit >= 0)
        SetFocus(hwnd);
      InvalidateRect(hwnd, nullptr, FALSE);
      return 0;
    }
    case WM_GETDLGCODE:
      // While capturing, take every key (Tab / Return / Esc / arrows included)
      // so IsDialogMessage does not turn them into navigation or Play / Quit.
      if (st->capture >= 0)
        return DLGC_WANTALLKEYS | DLGC_WANTARROWS | DLGC_WANTCHARS;
      break;
    case WM_KEYDOWN:
    case WM_SYSKEYDOWN: {
      if (st->capture < 0)
        break;
      if (lparam & (1 << 30))
        return 0;  // key repeat
      const UINT vk = static_cast<UINT>(wparam);
      if (vk == VK_ESCAPE) {
        st->status.clear();
        CancelCapture(st);
        return 0;
      }
      const std::string name = KeyNameFromVk(vk);
      if (name.empty()) {
        st->status = L"That key cannot be used. Press another key, or Esc to cancel.";
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
      }
      AssignCapturedKey(st, name);
      return 0;
    }
    case WM_SYSCHAR:
    case WM_CHAR:
    case WM_KEYUP:
    case WM_SYSKEYUP:
      if (st->capture >= 0)
        return 0;
      break;
    case WM_KILLFOCUS:
      CancelCapture(st);
      break;
  }
  return DefWindowProcW(hwnd, msg, wparam, lparam);
}

// ---- tab pages ----------------------------------------------------------------------------

const wchar_t* const kPageClass = L"NarutoSetupPage";

// Plain white page under each tab; command / notify / static-colour messages go
// to the main window so all handling lives in one place.
LRESULT CALLBACK PageProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
  switch (msg) {
    case WM_ERASEBKGND: {
      RECT rc;
      GetClientRect(hwnd, &rc);
      FillRect(reinterpret_cast<HDC>(wparam), &rc,
               static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH)));
      return 1;
    }
    case WM_COMMAND:
    case WM_NOTIFY:
    case WM_CTLCOLORSTATIC:
      return SendMessageW(GetParent(hwnd), msg, wparam, lparam);
  }
  return DefWindowProcW(hwnd, msg, wparam, lparam);
}
