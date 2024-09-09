#pragma once
#include <algorithm>
#include <cstdio>

#include "../eigen/eigen_libs0.hpp"

namespace FS_libs_version {
class version_t {
public:
  int Major_Version;
  int Minor_Version;
  int Patch_Level;
  char date[32];
  char vcode[32];
};
constexpr version_t FS_Version = {1, 1, 0, "Mar 31, 2019", "FS proto"};

inline void FS_get_version(int &version, char date[32] = nullptr,
                           char vcode[32] = nullptr) {
  version = FS_Version.Major_Version * 100 + FS_Version.Minor_Version * 10 +
            FS_Version.Patch_Level;
  if (date != nullptr) {
    std::snprintf(date, 32, "%s\n", FS_Version.date);
  }
  if (vcode != nullptr) {
    std::snprintf(vcode, 32, "%s\n", FS_Version.vcode);
  }
}

inline void FS_show_version() {
  const auto id = eigen_libs0::eigen_get_id().id;
  const auto i = std::min(26, FS_Version.Patch_Level);
  const auto patchlevel = " abcdefghijklmnopqrstuvwxyz*"[i + 1];

  char version[256];
  std::snprintf(version, 256, "%d.%d%c", FS_Version.Major_Version,
                FS_Version.Minor_Version, patchlevel);

  if (id == 1) {
    std::printf("## FS version (%s) / (%s) / (%s)\n", version, FS_Version.date,
                FS_Version.vcode);
  }
}
} // namespace FS_libs_version