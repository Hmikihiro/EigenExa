#pragma once
/**
 * @file   FS_merge_d.hpp
 * @brief  subroutine FS_merge_d
 */
#include <algorithm>

#include "../cblas_lapacke_wrapper.hpp"
#include "FS_const.hpp"
#include "FS_dividing.hpp"

namespace {
namespace dc2_FS {
/**
 * subroutine FS_merge_d
 *
 * @brief  @n
 * Purpose @n
 * ======= @n
 * gather D of sub-matrix.
 *
 *  Arguments
 *  =========
 *
 * @param[in]     N        (global input) INTEGER @n
 *                         The order of the tridiagonal matrix T.  N = 0.
 *
 * @param[in]     D        (local input) DOUBLE PRECISION array, dimension(N)
 *@n  The diagonal elements of the tridiagonal matrix.
 *
 * @param[in]     SUBTREE  (input) type(bt_node) @n
 *                         sub-tree information of merge block.
 *
 * @param[out]    DOUT     (local output) DOUBLE PRECISION array, dimension (N)
 *                         generated D before MPI_ALLREDUCE.
 *
 */
template <class Integer, class Float>
void FS_merge_d(const Integer n, const Float d[],
                const bt_node<Integer, Float> &subtree, Float d_out[]) {
  // reduce用バッファのゼロクリア
  std::fill_n(d_out, n, FS_const::ZERO<Float>);

  // プロセス情報取得
  const auto grid_info = subtree.FS_grid_info();
  const auto NB = subtree.FS_get_NB();

  if (subtree.direction_horizontal_ == 1) {
    // 横分割のとき

    // 先頭列を含むプロセス列
    const auto col = subtree.FS_info_G1L('C', 0).rocsrc;
    if (col == grid_info.mycol) {
      // iを含むプロセス行
      for (Integer i = 0; i < n; i += NB) {
        const auto row = subtree.FS_info_G1L('R', i).rocsrc;
        if (row == grid_info.myrow) {
          const auto NB1 = std::min(n, i + NB) - i;
          lapacke::copy(NB1, &d[i], 1, &d_out[i], 1);
        }
      }
    }
  } else {
    // 横分割のとき

    // 先頭行を含むプロセス行
    const auto row = subtree.FS_info_G1L('R', 0).rocsrc;

    // dをコピー
    if (row == grid_info.myrow) {
      for (Integer j = 0; j < n; j += NB) {
        // jを含むプロセス列
        const auto col = subtree.FS_info_G1L('C', j).rocsrc;
        if (col == grid_info.mycol) {
          const auto NB1 = std::min(n, j + NB) - j;
          lapacke::copy(NB1, &d[j], 1, &d_out[j], 1);
        }
      }
    }
  }
}
} // namespace dc2_FS
} // namespace
