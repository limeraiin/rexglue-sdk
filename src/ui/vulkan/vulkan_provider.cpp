/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2025 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/immediate_drawer.h>
#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/provider.h>

REXCVAR_DEFINE_BOOL(vulkan_validation_enabled, false, "UI/Vulkan",
                    "Enable Vulkan validation layers")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(vulkan_device, -1, "UI/Vulkan", "Vulkan device index (-1 for auto selection)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_BOOL(vulkan_prefer_geometry_shader, true, "UI/Vulkan",
                    "Prefer physical devices supporting geometryShader when auto-selecting")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(
    vulkan_prefer_fragment_stores_and_atomics, true, "UI/Vulkan",
    "Prefer physical devices supporting fragmentStoresAndAtomics when auto-selecting")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(vulkan_prefer_vertex_pipeline_stores_and_atomics, true, "UI/Vulkan",
                    "Prefer physical devices supporting vertexPipelineStoresAndAtomics when "
                    "auto-selecting")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(vulkan_prefer_fill_mode_non_solid, true, "UI/Vulkan",
                    "Prefer physical devices supporting fillModeNonSolid when auto-selecting")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace rex {
namespace ui {
namespace vulkan {

std::unique_ptr<VulkanProvider> VulkanProvider::Create(const bool with_gpu_emulation,
                                                       const bool with_presentation) {
  std::unique_ptr<VulkanProvider> provider(new VulkanProvider());

  provider->vulkan_instance_ =
      VulkanInstance::Create(with_presentation, REXCVAR_GET(vulkan_validation_enabled));
  if (!provider->vulkan_instance_) {
    return nullptr;
  }

  std::vector<VkPhysicalDevice> physical_devices;
  provider->vulkan_instance_->EnumeratePhysicalDevices(physical_devices);

  if (physical_devices.empty()) {
    REXLOG_WARN("No Vulkan physical devices available");
    return nullptr;
  }

  const VulkanInstance::Functions& ifn = provider->vulkan_instance_->functions();

  REXLOG_WARN(
      "Available Vulkan physical devices (use the 'vulkan_device' "
      "configuration variable to force a specific device):");
  for (size_t physical_device_index = 0; physical_device_index < physical_devices.size();
       ++physical_device_index) {
    VkPhysicalDeviceProperties physical_device_properties;
    ifn.vkGetPhysicalDeviceProperties(physical_devices[physical_device_index],
                                      &physical_device_properties);
    REXLOG_WARN("* {}: {}", physical_device_index, physical_device_properties.deviceName);
  }

  if (REXCVAR_GET(vulkan_device) >= 0 &&
      uint32_t(REXCVAR_GET(vulkan_device)) < physical_devices.size()) {
    provider->vulkan_device_ = VulkanDevice::CreateIfSupported(
        provider->vulkan_instance_.get(), physical_devices[REXCVAR_GET(vulkan_device)],
        with_gpu_emulation, with_presentation);
  }

  if (!provider->vulkan_device_) {
    std::vector<VkPhysicalDevice> physical_devices_ordered = physical_devices;
    bool prefer_geometry_shader = REXCVAR_GET(vulkan_prefer_geometry_shader);
    bool prefer_fragment_stores = REXCVAR_GET(vulkan_prefer_fragment_stores_and_atomics);
    bool prefer_vertex_stores = REXCVAR_GET(vulkan_prefer_vertex_pipeline_stores_and_atomics);
    bool prefer_fill_mode_non_solid = REXCVAR_GET(vulkan_prefer_fill_mode_non_solid);
    if (with_gpu_emulation && physical_devices.size() > 1) {
      struct PhysicalDeviceScore {
        VkPhysicalDevice physical_device;
        uint32_t score;
      };
      std::vector<PhysicalDeviceScore> scored_devices;
      scored_devices.reserve(physical_devices.size());
      for (const VkPhysicalDevice physical_device : physical_devices) {
        VkPhysicalDeviceFeatures supported_features = {};
        ifn.vkGetPhysicalDeviceFeatures(physical_device, &supported_features);
        // Device class dominates every feature bit. On a hybrid laptop the iGPU
        // supports all four preferred features too, so a feature-only score ties
        // and the stable sort hands back physical device 0 - the iGPU. That cost
        // 8.4x on the D3D12 side before EnumAdapterByGpuPreference fixed it there.
        VkPhysicalDeviceProperties physical_device_properties;
        ifn.vkGetPhysicalDeviceProperties(physical_device, &physical_device_properties);
        uint32_t device_class_rank = 0;
        switch (physical_device_properties.deviceType) {
          case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            device_class_rank = 4;
            break;
          case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            device_class_rank = 2;
            break;
          case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            device_class_rank = 1;
            break;
          default:
            device_class_rank = 0;
            break;
        }
        uint32_t score = device_class_rank * 16;
        if (prefer_geometry_shader && supported_features.geometryShader) {
          ++score;
        }
        if (prefer_fragment_stores && supported_features.fragmentStoresAndAtomics) {
          ++score;
        }
        if (prefer_vertex_stores && supported_features.vertexPipelineStoresAndAtomics) {
          ++score;
        }
        if (prefer_fill_mode_non_solid && supported_features.fillModeNonSolid) {
          ++score;
        }
        scored_devices.push_back({physical_device, score});
      }

      std::stable_sort(scored_devices.begin(), scored_devices.end(),
                       [](const PhysicalDeviceScore& a, const PhysicalDeviceScore& b) {
                         return a.score > b.score;
                       });

      if (!scored_devices.empty() && scored_devices.front().score != scored_devices.back().score) {
        physical_devices_ordered.clear();
        physical_devices_ordered.reserve(scored_devices.size());
        for (const PhysicalDeviceScore& scored_device : scored_devices) {
          physical_devices_ordered.push_back(scored_device.physical_device);
        }
      }
    }

    for (const VkPhysicalDevice physical_device : physical_devices_ordered) {
      provider->vulkan_device_ = VulkanDevice::CreateIfSupported(
          provider->vulkan_instance_.get(), physical_device, with_gpu_emulation, with_presentation);
      if (provider->vulkan_device_) {
        break;
      }
    }

    if (!provider->vulkan_device_) {
      REXLOG_WARN(
          "Couldn't choose a compatible Vulkan physical device or initialize a "
          "Vulkan logical device");
      return nullptr;
    }
  }

  if (with_presentation) {
    provider->ui_samplers_ = UISamplers::Create(provider->vulkan_device_.get());
    if (!provider->ui_samplers_) {
      return nullptr;
    }
  }

  return provider;
}

std::unique_ptr<Presenter> VulkanProvider::CreatePresenter(
    Presenter::HostGpuLossCallback host_gpu_loss_callback) {
  return VulkanPresenter::Create(host_gpu_loss_callback, vulkan_device(), ui_samplers());
}

std::unique_ptr<ImmediateDrawer> VulkanProvider::CreateImmediateDrawer() {
  return VulkanImmediateDrawer::Create(vulkan_device(), ui_samplers());
}

}  // namespace vulkan
}  // namespace ui
}  // namespace rex
