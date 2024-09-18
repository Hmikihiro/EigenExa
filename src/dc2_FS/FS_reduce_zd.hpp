#pragma once
/**
 * @file FS_reduce_zd.hpp
 * @brief FS_reduce_zd
 */

#include <iostream>

#include "../FS_libs/FS_libs.hpp"
#include "../MPI_Allreduce_group.hpp"
#include "../MPI_Datatype_wrapper.hpp"
#include "FS_dividing.hpp"
#include "FS_prof.hpp"

namespace {
namespace dc2_FS {
using FS_libs::FS_COMM_WORLD;

/**
 * subroutine FS_REDUCE_ZD
 *
 * @brief  @n
 * Purpose @n
 * ======= @n
 * MPI_ALLREDUCE Z and D
 *
 * @param[in]     N        (global input) INTEGER @n
 *                         The order of the tridiagonal matrix T.  N >= 0.
 *
 * @param[in]     SUBTREE  (input) type(bt_node) @n
 *                         sub-tree information of merge block.
 *
 * @param[in]     WORK     (input) DOUBLE PRECISION array, dimension (N,2)         @n
 *                         WORK(:,1) is the updating vector before MPI_ALLREDUCE.  @n
 *                         WORK(:,2) is the generated D before MPI_ALLREDUCE.
 *
 * @param[out]    Z        (local output) DOUBLE PRECISION array, dimension (N)                   @n
 *                         The updating vector (the last row of the first sub-eigenvector  @n
 *                         matrix and the first row of the second sub-eigenvector matrix).
 *
 * @param[out]    D        (local output) DOUBLE PRECISION array, dimension (N)
 *                         generated D.
 *
 * @param[out]    prof     (global output) type(FS_prof) @n
 *                         profiling information of each subroutines.
 */
template <class Integer, class Real>
void FS_reduce_zd(const Integer n, const bt_node<Integer, Real> &subtree,
                  Real work[], Real z[], Real d[], FS_prof &prof) {
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_reduce_zd start." << std::endl;
  }
#endif
#if TIMER_PRINT
  prof.start(45);
#endif
#if TIMER_PRINT
#ifdef PROF_DETAIL
  prof.start(46);
  MPI_Barrier(subtree.MERGE_COMM);
  prof.end(46);
  prof.start(47);
#endif
#endif

  // reduce
  MPI_Group_Allreduce(work, z, n * 2, MPI_Datatype_wrapper::MPI_TYPE<Real>,
                      MPI_SUM, FS_COMM_WORLD, subtree.MERGE_GROUP_);

#pragma omp parallel for
  for (Integer j = 0; j < n; j++) {
    d[j] = z[j + n];
  }

#if TIMER_PRINT
#ifdef PROF_DETAIL
  prof.end(47);
#endif
#endif

#if TIMER_PRINT
  prof.end(45);
#endif

#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_reduce_zd end." << std::endl;
  }
#endif
}

} // namespace dc2_FS
} // namespace
