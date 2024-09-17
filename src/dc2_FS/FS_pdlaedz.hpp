#pragma once

#include <algorithm>

#include "../cblas_lapacke_wrapper.hpp"
#include "FS_const.hpp"
#include "FS_dividing.hpp"
#include "FS_prof.hpp"

#if defined(_DEBUGLOG)
#include "../FS_libs/FS_libs.hpp"
#include <cstdio>
#endif

namespace {
template <class Integer, class Float>
void FS_pdlaedz(const Integer n, const Integer n1, const Float q[],
                const Integer ldq, const bt_node<Integer, Float> &subtree,
                Float z[], FS_prof &prof) {
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlaedz start." << std::endl;
  }
#endif
#if TIMER_PRINT
  prof.start(40);
#endif

  // プロセス情報
  const auto grid_info = subtree.FS_grid_info();

  const auto nb = subtree.FS_get_NB();

  std::fill_n(z, n, FS_const::ZERO<Float>);

  // Z1, Z2を含むプロセス行を取得
  const auto iz1_info = subtree.FS_info_G1L('R', n1 - 1);
  const auto &iz1 = iz1_info.l_index;
  const auto &iz1row = iz1_info.rocsrc;
  // Form z1 which consist of the last row of Q1
  if (iz1row == grid_info.myrow) {
    // z1を含むプロセス例
#pragma omp parallel for
    for (Integer j = 0; j < n1; j += nb) {
      const auto jz1 = subtree.FS_info_G1L('C', j);
      if (jz1.rocsrc == grid_info.mycol) {
        const auto nb1 = std::min(n1, j + nb) - j;
        const auto q_index = iz1 + jz1.l_index * ldq;
        lapacke::copy<Float>(nb1, &q[q_index], ldq, &z[j], 1);
      }
    }
  }

  const auto iz2_info = subtree.FS_info_G1L('R', n1);
  const auto &iz2 = iz2_info.l_index;
  const auto &iz2row = iz2_info.rocsrc;
  // Form z2 which consist of the first row of Q2
  if (iz2row == grid_info.myrow) {
    // z2を含むプロセス例
#pragma omp parallel for
    for (Integer j = n1; j < n; j += nb) {
      const auto jz2_info = subtree.FS_info_G1L('C', j);
      const auto &jz2 = jz2_info.l_index;
      const auto &jz2col = jz2_info.rocsrc;
      if (jz2col == grid_info.mycol) {
        const auto nb1 = std::min(n, j + nb) - j;
        const auto q_index = iz2 + jz2 * ldq;
        lapacke::copy<Float>(nb1, &q[q_index], ldq, &z[j], 1);
      }
    }
  }

#if TIMER_PRINT
  prof.end(40);
#endif
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlaedz end." << std::endl;
  }
#endif
}
} // namespace
