#pragma once
/**
 * @file FS_EDC.hpp
 * @brief FS_EDC
 */

#include <cmath>
#include <iostream>

#include "../cblas_lapacke_wrapper.hpp"
#include "../eigen_libs0.hpp"
#include "FS_const.hpp"
#include "FS_pdlaed0.hpp"
#include "FS_prof.hpp"

namespace {
namespace dc2_FS {
using FS_const::ONE;
using FS_const::ZERO;
/**
 * @brief @n
 *   Purpose @n
 *   ======= @n
 *   FS_EDC computes all eigenvalues and eigenvectors of a             @n
 *   symmetric tridiagonal matrix in parallel, using the divide and    @n
 *   conquer algorithm.
 *
 *  @return              INFO  @n
 *                       = 0: successful exit   @n
 *                       !=0: error exit
 *
 *
 *
 * @param[in]     n      (global input) int @n
 *                       The order of the tridiagonal matrix T.  N >= 0.
 *
 * @param[in,out] D      (global input/output) float or double array, dimension
 * (N) @n On entry, the diagonal elements of the tridiagonal matrix.  @n On
 * exit, if INFO = 0, the eigenvalues in descending order.
 *
 * @param[in,out] E      (global input/output) float or double array,
 * dimension(N-1) @n On entry, the subdiagonal elements of the tridiagonal
 * matrix. @n On exit, E has been destroyed.
 *
 * @param[in,out] Q      (local output) float or double array, local dimension
 * (LDQ, *)   @n Q contains the orthonormal eigenvectors of the symmetric
 * tridiagonal matrix.   @n On output, Q is distributed across the P processes
 * in non block cyclic format. @n
 *
 * @param[in]     ldq    (local input) int @n
 *                       leading dimension of array Q.
 *
 * @param         work   (local workspace/output) float or double array,
 * dimension (LWORK)
 *
 * @param[in]     lwork  (local input/output) int, the dimension of the array
 * WORK. @n LWORK = 1 + 6*N + 3*NP*NQ + NQ*NQ @n LWORK can be obtained from
 * subroutine FS_WorkSize.
 *
 * @param         iwork  (local workspace/output) int array, dimension (LIWORK)
 *
 * @param[in]     liwork (input) int                   @n
 *                       The dimension of the array IWORK. @n
 *                       LIWORK = 1 + 8*N + 8*NPCOL        @n
 *                       LIWORK can be obtained from subroutine FS_WorkSize.
 *
 * @param[out]    prof   (global output) type(FS_prof) @n
 *                       profiling information of each subroutines.
 *
 * @note This routine is modified from ScaLAPACK PDSTEDC.f
 */

template <class Integer, class Real>
Integer FS_EDC(const Integer n, Real *d, Real *e, Real *q, const Integer ldq,
               Real *work, const Integer lwork, Integer *iwork,
               const Integer liwork, FS_prof *prof) {

  FS_prof prof_tmp = {};

  Integer info = 0;
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_EDC start" << std::endl;
  }
#endif

#if TIMER_PRINT
  if (prof != nullptr) {
    prof_tmp = *prof;
  } else {
    prof_tmp.init();
  }
  prof_tmp.start(10);
#endif

  const auto eigen_procs = eigen_libs0_wrapper::eigen_get_procs();
  const auto x_nnod = eigen_procs.x_procs;
  const auto y_nnod = eigen_procs.y_procs;

  const auto eigen_id = eigen_libs0_wrapper::eigen_get_id();
  const auto x_inod = eigen_id.x_id;
  const auto y_inod = eigen_id.y_id;

  if (n == 0) {
    // 何もしない
  } else if (n == 1) {
    if (x_inod == 1 && y_inod == 1) {
      q[0] = ONE<Real>;
    }
  } else if (x_nnod * y_nnod == 1) {
    // If P=NPROW*NPCOL=1, solve the problem with DSTEDC.
#if TIMER_PRINT
    prof_tmp.start(11);
#endif
    info = lapacke::stedc<Real>('I', n, d, e, q, ldq, work, lwork,
                                 reinterpret_cast<eigen_mathlib_int *>(iwork),
                                 liwork);

#if TIMER_PRINT
    prof_tmp.end(11);
#endif
  } else {
    // Scale matrix to allowable range, if necessary.
    auto orgnrm = lapacke::lanst<Real>('M', n, d, e);
    if (std::isnan(orgnrm)) {
      orgnrm = ZERO<Real>;
    }
    if (orgnrm != ZERO<Real>) {
      info = lapacke::lascl<Real>('G', 0, 0, orgnrm, ONE<Real>, n, 1, d, n);
      if (n - 1 >= 1) {
        info = lapacke::lascl<Real>('G', 0, 0, orgnrm, ONE<Real>, n - 1, 1, e,
                                     n - 1);
      }
    }
    info = FS_pdlaed0<Integer, Real>(n, d, e, q, ldq, work, lwork, iwork,
                                      liwork, prof_tmp);
    // Scale back.
    if (info == 0 && orgnrm != ZERO<Real>) {
      info = lapacke::lascl<Real>('G', 0, 0, ONE<Real>, orgnrm, n, 1, d, n);
    }
  }

#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_EDC end. INFO=" << info << std::endl;
  }
#endif

#if TIMER_PRINT
  prof_tmp.end(10);
  if (prof != nullptr) {
    *prof = prof_tmp;
  } else {
    prof_tmp.finalize();
  }
#endif
  return info;
}
} // namespace dc2_FS
} // namespace
