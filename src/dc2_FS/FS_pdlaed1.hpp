#pragma once
/**
 * @file FS_pdlaed1.hpp
 * @brief FS_pdlaed1
 */
#include <mpi.h>

#include <algorithm>
#include <iostream>

#include "FS_dividing.hpp"
#include "FS_merge_d.hpp"
#include "FS_pdlaed2.hpp"
#include "FS_pdlaed3.hpp"
#include "FS_pdlaedz.hpp"
#include "FS_prof.hpp"
#include "FS_reduce_zd.hpp"

namespace {
namespace dc2_FS {
/**
 * @brief @n
 *  Purpose @n
 *  ======= @n
 *  FS_pdlaed1 computes the updated eigensystem of a diagonal @n
 *  matrix after modification by a rank-one symmetric matrix, @n
 *  in parallel.
 *
 * @param n
 * @param n1
 * @param d
 * @param q
 * @param ldq
 * @param subtree
 * @param rho
 * @param work
 * @param iwork
 * @param prof
 * @return info Integer @n
 *              = 0: successuful exit @n
 *              !=0: error exit
 */
template <class Integer, class Real>
Integer FS_pdlaed1(const Integer n, const Integer n1, Real d[], Real q[],
                   const Integer ldq, const bt_node<Integer, Real> &subtree,
                   const Real rho, Real work[], Integer iwork[],
                   FS_prof &prof) {
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlaed1 start." << std::endl;
  }
#endif
#if TIMER_PRINT
  prof.start(30);
#endif
  const auto info = [&]() mutable -> Integer {
    if (n == 0) {
      // Quick return if possible
      return 0;
    }
    // The following values are  integer pointers which indicate
    // the portion of the workspace used by a particular array
    // in FS_PDLAED2 and FS_PDLAED3.
    const auto grid_info = subtree.FS_grid_info();

    const auto nblk = subtree.FS_get_NBLK();
    const auto nb = subtree.FS_get_NB();
    const auto np = (nblk / grid_info.nprow) * nb;
    const auto nq = (nblk / grid_info.npcol) * nb;
    const auto ldq2 = std::max(np, (Integer)1);
    const auto ldu = nq;
    const auto lsendq2 = ldq2 * nq;
    //
    const Integer iz = 0;
    const auto z = &work[iz];
    const auto idlamda = iz + n;
    const auto dlamda = &work[idlamda];
    const auto iw = idlamda + n;
    const auto w = &work[iw];
    const auto ipq2 = iw + n;
    const auto q2 = &work[ipq2];
    const auto ipu = ipq2 + ldq2 * nq;
    const auto u = &work[ipu];
    const auto ibuf = ipu + ldu * nq;
    const auto buf = &work[ibuf];
    const auto isendq2 = ibuf + 4 * n;
    const auto sendq2 = &work[isendq2];
    const auto irecvq2 = isendq2 + lsendq2;
    const auto recvq2 = &work[irecvq2];
    //
    const Integer ictot = 0;
    const auto ipsm = ictot + grid_info.npcol * 4;
    const auto indx = ipsm + grid_info.npcol * 4;
    const auto indxc = indx + n;
    const auto indxp = indxc + n;
    const auto indcol = indxp + n;
    const auto coltyp = indcol + n;
    const auto indrow = coltyp + n;
    const auto indxr = indrow + n;
    //
    // for FS_merge_d, FS_pdlaedz, FS_reduce_zd
    const auto izwork = iz + n * 2;
    const auto idwork = izwork + n;
//
// merge d
//
#if TIMER_PRINT
    prof.start(31);
#endif
    FS_merge_d<Integer, Real>(n, d, subtree, &work[idwork]);
#if TIMER_PRINT
    prof.end(31);
#endif
    //
    // Form the z-vector which consists of the last row of Q_1 and the
    // first row of Q_2.
    //
    FS_pdlaedz<Integer, Real>(n, n1, q, ldq, subtree, &work[izwork], prof);
    //
    // MPI_allreduce d and z
    //
    FS_reduce_zd<Integer, Real>(n, subtree, &work[izwork], &work[iz], d, prof);
    //
    // Deflate eigenvalues.
    //
    const auto result = FS_pdlaed2<Integer, Real>(
        n, n1, d, q, ldq, subtree, rho, z, w, dlamda, ldq2, q2, &iwork[indx],
        &iwork[ictot], buf, &iwork[coltyp], &iwork[indcol], &iwork[indxc],
        &iwork[indxp], &iwork[ipsm], prof);

    //
    // Solve Secular Equation.
    //
    const Integer lctot = subtree.y_nnod_;
    Integer info = 0;
    if (result.k != 0) {
      info = FS_pdlaed3<Integer, Real>(
          result.k, n, n1, d, result.rho, dlamda, w, ldq, q, subtree, ldq2, q2,
          ldu, u, &iwork[indx], lctot, &iwork[ictot], sendq2, recvq2, z, buf,
          &iwork[indrow], &iwork[indcol], &iwork[indxc], &iwork[indxr], prof);
    }
    return info;
  }();

#if TIMER_PRINT
  prof.end(30);
#endif
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlaed1 end. INFO=" << info << std::endl;
  }
#endif
  return info;
}
} // namespace dc2_FS
} // namespace
