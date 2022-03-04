// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include "paddle/phi/core/custom_kernel.h"

namespace phi {

void RegisterCustomKernels(const CustomKernelMap& custom_kernel_map) {
  auto& kernel_info_map = custom_kernel_map.GetMap();
  VLOG(3) << "Size of custom_kernel_map: " << kernel_info_map.size();

  auto& kernels = KernelFactory::Instance().kernels();
  for (auto& pair : kernel_info_map) {
    PADDLE_ENFORCE_NE(
        kernels.find(pair.first),
        kernels.end(),
        phi::errors::InvalidArgument(
            "The kernel %s is not ready for custom kernel registering.",
            pair.first));

    for (auto& info_pair : pair.second) {
      PADDLE_ENFORCE_EQ(
          kernels[pair.first].find(info_pair.first),
          kernels[pair.first].end(),
          phi::errors::InvalidArgument(
              "The operator <%s>'s kernel: %s has been already existed "
              "in Paddle, please contribute PR if it is necessary "
              "to optimize the kernel code. Custom kernel does NOT support "
              "to replace existing kernel in Paddle.",
              pair.first,
              info_pair.first));

      kernels[pair.first][info_pair.first] = info_pair.second;

      VLOG(3) << "Successed in registering operator <" << pair.first
              << ">'s kernel: " << info_pair.first
              << " to Paddle. It will be used like native ones.";
    }
  }
}

void LoadCustomKernelLib(const std::string& dso_lib_path, void* dso_handle) {
#ifdef _LINUX
  typedef phi::CustomKernelMap& get_custom_kernel_map_t();
  auto* func = reinterpret_cast<get_custom_kernel_map_t*>(
      dlsym(dso_handle, "PD_GetCustomKernelMap"));

  if (func == nullptr) {
    LOG(WARNING) << "Skipped lib [" << dso_lib_path << "]: fail to find "
                 << "PD_GetCustomKernelMap symbol in this lib.";
    return;
  }
  auto& custom_kernel_map = func();
  phi::RegisterCustomKernels(custom_kernel_map);
  LOG(INFO) << "Successed in loading custom kernels in lib: " << dso_lib_path;
#else
  VLOG(3) << "Unsupported: Custom kernel is only implemented on Linux.";
#endif
  return;
}
}  // namespace phi

#ifdef __cplusplus
extern "C" {
#endif

// C-API to get global CustomKernelMap.
phi::CustomKernelMap& PD_GetCustomKernelMap() {
  return phi::CustomKernelMap::Instance();
}

#ifdef __cplusplus
}  // end extern "C"
#endif
