#pragma once
/**
 * @file FS_pdlaed2.hpp
 * @brief FS_pdlaed2
 */
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#include "../FS_libs/FS_libs.hpp"
#include "../MPI_Datatype_wrapper.hpp"
#include "../cblas_lapacke_wrapper.hpp"
#include "FS_const.hpp"
#include "FS_dividing.hpp"
#include "FS_prof.hpp"

namespace {
namespace dc2_FS {
template <class Integer, class Float>
Integer get_NPA(const Integer n, const Integer nb,
                const bt_node<Integer, Float> &subtree, const Integer myrow) {
  Integer npa = 0;
#pragma omp parallel for reduction(+ : npa)
  for (Integer i = 0; i < n; i += nb) {
    const auto row = subtree.FS_info_G1L('R', i).rocsrc;
    if (row == myrow) {
      for (Integer j = 0; j < nb; j++) {
        if (i + j < n) {
          npa += 1;
        }
      }
    }
  }
  return npa;
}

template <class Integer>
void init_ctot(const Integer n, const Integer coltyp[], const Integer indcol[],
               const Integer lctot, Integer ctot[], const Integer npcol) {
#pragma omp parallel for
  for (Integer j = 0; j < 4; j++) {
    std::fill_n(&ctot[j * lctot], npcol, 0);
  }

  for (Integer j = 0; j < n; j++) {
    auto ct = coltyp[j];
    auto col = indcol[j];
    ctot[ct * lctot + col] += 1;
  }
}

/**
 * \brief PSM(*) = Position in SubMatrix (of types 0 through 3)
 */
template <class Integer>
void set_psm(const Integer lctot, Integer psm[], const Integer ctot[],
             const Integer npcol) {
#pragma omp parallel for
  for (Integer col = 0; col < npcol; col++) {
    psm[0 * lctot + col] = 1;
    psm[1 * lctot + col] = 1 + ctot[0 * lctot + col];
    psm[2 * lctot + col] = psm[1 * lctot + col] + ctot[1 * lctot + col];
    psm[3 * lctot + col] = psm[2 * lctot + col] + ctot[2 * lctot + col];
  }
}

template <class Integer>
void set_ptt(const Integer lctot, Integer ptt[4], const Integer ctot[],
             const Integer npcol) {
  std::fill_n(ptt, 4, 0);
  ptt[0] = 1;
#pragma omp parallel for
  for (Integer i = 1; i < 4; i++) {
    Integer ct = 0;
    for (Integer j = 0; j < npcol; j++) {
      ct += ctot[(i - 1) * lctot + j];
    }
    ptt[i] = ct;
  }

  for (Integer i = 1; i < 4; i++) {
    ptt[i] += ptt[i - 1];
  }
}

template <class Integer, class Float>
void set_indxp(const Integer n, Integer &k2, const Integer nj, const Integer pj,
               Float d[], const Float c, const Float s, Integer indxp[]) {
  const auto c_2 = static_cast<Float>(pow(c, 2));
  const auto s_2 = static_cast<Float>(pow(s, 2));
  const auto t = d[pj] * c_2 + d[nj] * s_2;
  d[nj] = d[pj] * s_2 + d[nj] * c_2;
  d[pj] = t;

  Integer i;
  for (i = 0; k2 + i < n; i++) {
    if (d[pj] < d[indxp[k2 + i]]) {
      indxp[k2 + i - 1] = indxp[k2 + i];
      indxp[k2 + i] = pj;
    } else {
      break;
    }
  }
  k2 -= 1;
  indxp[k2 + i] = pj;
}

template <class Integer, class Float>
void pdlaed2_comm(const Integer mycol, const Integer ldq, Float q[],
                  const Integer npa, const bt_node<Integer, Float> &subtree,
                  Float qbuf[], const Integer nj, const Integer pj,
                  const Float c, const Float s, Integer indcol[]) {
  const auto njj_info = subtree.FS_info_G1L('C', nj);
  const auto &njj = njj_info.l_index;
  const auto &njcol = njj_info.rocsrc;
  const auto pjj_info = subtree.FS_info_G1L('C', pj);
  const auto &pjj = pjj_info.l_index;
  const auto &pjcol = pjj_info.rocsrc;

  if (indcol[pj] == indcol[nj] && mycol == njcol) {
    lapacke::rot<Float>(npa, &q[pjj * ldq + 0], 1, &q[njj * ldq + 0], 1, c, s);
  } else if (mycol == pjcol) {
    MPI_Request req[2];
    MPI_Irecv(qbuf, npa, MPI_Datatype_wrapper::MPI_TYPE<Float>,
              subtree.group_Y_processranklist_[njcol], 1,
              FS_libs::FS_COMM_WORLD, &req[1]);

    MPI_Isend(&q[pjj * ldq + 0], npa, MPI_Datatype_wrapper::MPI_TYPE<Float>,
              subtree.group_Y_processranklist_[njcol], 1,
              FS_libs::FS_COMM_WORLD, &req[0]);
    MPI_Waitall(2, req, MPI_STATUS_IGNORE);
    lapacke::rot<Float>(npa, &q[pjj * ldq + 0], 1, qbuf, 1, c, s);
  } else if (mycol == njcol) {
    MPI_Request req[2];
    MPI_Irecv(qbuf, npa, MPI_Datatype_wrapper::MPI_TYPE<Float>,
              subtree.group_Y_processranklist_[pjcol], 1,
              FS_libs::FS_COMM_WORLD, &req[1]);

    MPI_Isend(&q[njj * ldq + 0], npa, MPI_Datatype_wrapper::MPI_TYPE<Float>,
              subtree.group_Y_processranklist_[pjcol], 1,
              FS_libs::FS_COMM_WORLD, &req[0]);
    MPI_Waitall(2, req, MPI_STATUS_IGNORE);
    lapacke::rot<Float>(npa, qbuf, 1, &q[njj * ldq + 0], 1, c, s);
  }
}


/**
 * @brief return of FS_pdlaed2
 */
template <class Integer, class Float> struct FS_pdlead2_result {
  /**
   * @brief  On exit, RHO has been modified to the value
   *         \n required by FS_PDLAED3.
   */
  Float rho;
  /**
   * @brief The number of non-deflated eigenvalues, and the order of the @n
   *        related secular equation. 0 <= K <=N.
   */
  Integer k;
};

/**
 * subroutine FS_PDLAED2
 *
 * @brief @n
 *  Purpose @n
 *  ======= @n
 *  FS_PDLAED2 sorts the two sets of eigenvalues together into a single  @n
 *  sorted set.  Then it tries to deflate the size of the problem.       @n
 *  There are two ways in which deflation can occur:  when two or more   @n
 *  eigenvalues are close together or if there is a tiny entry in the    @n
 *  Z vector.  For each such occurrence the order of the related secular
 *  equation problem is reduced by one.
 *
 * @param[in]     N        (input) INTEGER @n
 *                         The dimension of the symmetric tridiagonal matrix.  N >= 0.
 *
 * @param[in]     N1       (input) INTEGER @n
 *                         The location of the last eigenvalue in the leading sub-matrix. @n
 *                         min(1,N) <= N1 <= N.
 *
 * @param[in,out] D        (input/output) DOUBLE PRECISION array, dimension (N)           @n
 *                         On entry, D contains the eigenvalues of the two submatrices to @n
 *                         be combined.                                                   @n
 *                         On exit, D contains the trailing (N-K) updated eigenvalues     @n
 *                         (those which were deflated) sorted into increasing order.
 *
 * @param[in,out] Q        (input/output) DOUBLE PRECISION array, dimension (LDQ, NQ)  @n
 *                         On entry, Q contains the eigenvectors of two submatrices in @n
 *                         the two square blocks with corners at (1,1), (N1,N1)        @n
 *                         and (N1+1, N1+1), (N,N).                                    @n
 *                         On exit, Q contains the trailing (N-K) updated eigenvectors @n
 *                         (those which were deflated) in its last N-K columns.
 *
 * @param[in]     LDQ      (local input) INTEGER @n
 *                         The leading dimension of the array Q.  LDQ >= max(1,NP).
 *
 * @param[in]     SUBTREE  (input) type(bt_node) @n
 *                         sub-tree information of merge block.
 *
 * @param[in] RHO          (global input/output) DOUBLE PRECISION                        @n
 *                         On entry, the off-diagonal element associated with the rank-1 @n
 *                         cut which originally split the two submatrices which are now  @n
 *                         being recombined.                                             @n
 *
 *
 * @param[in,out] Z        (global input) DOUBLE PRECISION array, dimension (N)           @n
 *                         On entry, Z contains the updating vector (the last             @n
 *                         row of the first sub-eigenvector matrix and the first row of   @n
 *                         the second sub-eigenvector matrix).                            @n
 *                         On exit, the contents of Z have been destroyed by the updating @n
 *                         process.
 *
 * @param[out]    W        (global output) DOUBLE PRECISION array, dimension (N)      @n
 *                         The first k values of the final deflation-altered z-vector @n
 *                         which will be passed to FS_PDLAED3.
 *
 * @param[out]    DLAMDA   (global output) DOUBLE PRECISION array, dimension (N)   @n
 *                         A copy of the first K eigenvalues which will be used by @n
 *                         FS_PDLAED3 to form the secular equation.
 *
 * @param[out]    Q2       (output) DOUBLE PRECISION array, dimension (LDQ2, NQ) @n
 *                         The eigen vectors which sorted by COLTYP
 *
 * @param[in]     LDQ2     (input) INTEGER @n
 *                         The leading dimension of the array Q2.
 *
 * @param[out]    INDX     (output) INTEGER array, dimension (N)                    @n
 *                         The permutation used to sort the contents of DLAMDA into @n
 *                         ascending order which will be passed to FS_PDLAED3.
 *
 * @param[out]    CTOT     (output) INTEGER array, dimension (NPCOL, 4) @n
 *                         The number of COLTYP of each process column  @n
 *                         which will be passed to FS_PDLAED3.
 *
 * @param         QBUF     (workspace) DOUBLE PRECISION array, dimension (N)
 *
 * @param         COLTYP   (workspace) INTEGER array, dimension (N)                   @n
 *                         During execution, a label which will indicate which of the @n
 *                         following types a column in the Q2 matrix is:              @n
 *                         1 : non-zero in the upper half only;                       @n
 *                         2 : dense;                                                 @n
 *                         3 : non-zero in the lower half only;                       @n
 *                         4 : deflated.
 *
 * @param         INDCOL   (workspace) INTEGER array, dimension (N)
 *
 * @param         INDXC    (workspace) INTEGER array, dimension (N)                       @n
 *                         The permutation used to arrange the columns of the deflated    @n
 *                         Q matrix into three groups:  the first group contains non-zero @n
 *                         elements only at and above N1, the second contains             @n
 *                         non-zero elements only below N1, and the third is dense.
 *
 * @param         INDXP    (workspace) INTEGER array, dimension (N)                      @n
 *                         The permutation used to place deflated values of D at the end @n
 *                         of the array.  INDXP(1:K) points to the nondeflated D-values  @n
 *                         and INDXP(K+1:N) points to the deflated eigenvalues.
 *
 * @param         PSM      (workspace) INTEGER array, dimension (NPCOL, 4)
 *
 *
 * @param[out]    prof     (global output) type(FS_prof) @n
 *                         profiling information of each subroutines.
 *
  * @return    FS_pdlead2_result @n
 * @note This routine is modified from ScaLAPACK PDLAED2.f
 */
template <class Integer, class Float>
FS_pdlead2_result<Integer, Float>
FS_pdlaed2(const Integer n, const Integer n1, Float d[], Float q[],
           const Integer ldq, const bt_node<Integer, Float> &subtree,
           const Float rho, Float z[], Float w[], Float dlamda[],
           const Integer ldq2, Float q2[], Integer indx[], Integer ctot[],
           Float qbuf[], Integer coltyp[], Integer indcol[], Integer indxc[],
           Integer indxp[], Integer psm[], FS_prof &prof) {
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_PDLAED2 start." << std::endl;
  }
#endif
#if TIMER_PRINT
  prof.start(50);
#endif

  const auto result = [&]() mutable -> FS_pdlead2_result<Integer, Float> {
    // Quick return if possible
    if (n == 0) {
      return {rho, 0};
    }

    const auto grid_info = subtree.FS_grid_info();
    const auto npcol = grid_info.npcol;
    const auto myrow = grid_info.myrow;
    const auto mycol = grid_info.mycol;

    const auto nb = subtree.FS_get_NB();

    const auto n2 = n - n1;
    std::fill_n(dlamda, n, 0);

    if (rho < FS_const::ZERO<Float>) {
      lapacke::scal<Float>(n2, FS_const::MONE<Float>, &z[n1], 1);
    }
    // Normalize z so that norm(z) = 1.  Since z is the concatenation of
    // two normalized vectors, norm2(z) = sqrt(2).

    const auto T = FS_const::ONE<Float> / std::sqrt(FS_const::TWO<Float>);
    lapacke::scal<Float>(n, T, z, 1);

    const auto result_rho = std::abs(FS_const::TWO<Float> * rho);

    // Calculate the allowable deflation tolerance
    const auto imax = lapacke::iamax<Float>(n, z, 1);
    const auto jmax = lapacke::iamax<Float>(n, d, 1);
    const auto abs_zmax = std::abs(z[imax]);
    const auto abs_dmax = std::abs(d[jmax]);
    const auto eps = std::numeric_limits<Float>::epsilon() / 2;
    const auto tol =
        FS_const::EIGHT<Float> * eps * std::max(abs_dmax, abs_zmax);

    // If the rank-1 modifier is small enough, no more needs to be done
    // except to reorganize Q so that its columns correspond with the
    // elements in D.
    if (result_rho * abs_zmax <= tol) {
      return {result_rho, 0};
    }

    // If there are multiple eigenvalues then the problem deflates.  Here
    // the number of equal eigenvalues are found.  As each equal
    // eigenvalue is found, an elementary reflector is computed to rotate
    // the corresponding eigensubspace so that the corresponding
    // components of Z are zero in this new basis.
    std::iota(indx, indx + n, 0);
    std::sort(indx, indx + n,
              [&d](Integer i1, Integer i2) { return d[i1] < d[i2]; });
#pragma omp parallel
    {
#pragma omp for
      for (Integer i = 0; i < n1; i++) {
        coltyp[i] = 0;
      }
#pragma omp for
      for (Integer i = n1; i < n; i++) {
        coltyp[i] = 2;
      }
#pragma omp for
      for (Integer i = 0; i < n; i += nb) {
        const auto col = subtree.FS_info_G1L('C', i).rocsrc;
        for (Integer j = 0; j < nb; j++) {
          if (i + j < n) {
            indcol[i + j] = col;
          }
        }
      }
    }

    const auto npa = get_NPA<Integer>(n, nb, subtree, myrow);
    Integer k = 0;
    {
      Integer k2 = n;
      Integer pj = 0;
      bool set_pj = true;
      for (Integer j = 0; j < n; j++) {
        const auto nj = indx[j];
        if (result_rho * std::abs(z[nj]) <= tol) {
          // Deflate due to small z component.
          k2 -= 1;
          coltyp[nj] = 3;
          indxp[k2] = nj;
        } else if (set_pj) {
          set_pj = false;
          pj = nj;
        } else {
          // Check if eigenvalues are close enough to allow deflation.
          const auto s_pj = z[pj];
          const auto c_nj = z[nj];
          // Find sqrt(a**2+b**2) without overflow or
          // destructive underflow.
          const auto tau = lapacke::lapy2<Float>(c_nj, s_pj);

          const auto t = d[nj] - d[pj];
          const auto c = c_nj / tau;
          const auto s = -s_pj / tau;
          if (std::abs(t * c * s) <= tol) {
            // Deflation is possible.
            z[nj] = tau;
            z[pj] = FS_const::ZERO<Float>;
            if (coltyp[nj] != coltyp[pj]) {
              coltyp[nj] = 1;
            }
            coltyp[pj] = 3;
#pragma omp parallel
            {
#pragma omp master
              {
                pdlaed2_comm<Integer>(mycol, ldq, q, npa, subtree, qbuf, nj, pj,
                                      c, s, indcol);
              }
#pragma omp single nowait
              { set_indxp<Integer>(n, k2, nj, pj, d, c, s, indxp); }
            }
            pj = nj;
          } else {
            dlamda[k] = d[pj];
            w[k] = z[pj];
            indxp[k] = pj;
            k += 1;
            pj = nj;
          }
        }
      }

      // Record the last eigenvalue.
      dlamda[k] = d[pj];
      w[k] = z[pj];
      indxp[k] = pj;
      k += 1;
    }
    // Count up the total number of the various types of columns, then
    // form a permutation which positions the four column types into
    // four uniform groups (although one or more of these groups may be
    // empty).

    Integer ptt[4];
    const auto lctot = subtree.y_nnod_;
    init_ctot<Integer>(n, coltyp, indcol, lctot, ctot, npcol);
    // PSM[*] =  Position in SubMatrix (of types 0 through 3)
    set_psm(lctot, psm, ctot, npcol);
    set_ptt(lctot, ptt, ctot, npcol);
    // Fill out the INDXC array so that the permutation which it induces
    // will place all type-1 columns first, all type-2 columns next,
    // then all type-3's, and finally all type-4's.
    for (Integer j = 0; j < n; j++) {
      auto js = indxp[j];
      auto col = indcol[js];
      auto ct = coltyp[js];
      auto i = subtree.FS_index_L2G('C', psm[col + ct * lctot] - 1, col);
      indx[j] = i;
      indxc[ptt[ct] - 1] = i + 1;
      psm[col + ct * lctot] += 1;
      ptt[ct] += 1;
    }
#pragma omp parallel for
    for (Integer j = 0; j < n; j++) {
      const auto js = indxp[j];
      const auto col = indcol[js];
      if (col == mycol) {
        const auto jjs = subtree.FS_index_G2L('C', js);
        const auto i = indx[j];
        const auto jjq2 = subtree.FS_index_G2L('C', i);
        lapacke::copy(npa, &q[jjs * ldq], 1, &q2[jjq2 * ldq2], 1);
      }
    }

    lapacke::copy(n, d, 1, z, 1);
#pragma omp parallel for
    for (Integer j = k; j < n; j++) {
      const auto js = indxp[j];
      const auto i = indx[j];
      d[i] = z[js];
    }
    return {result_rho, k};
  }();

#if TIMER_PRINT
  prof.end(50);
#endif

#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_PDLAED2 end." << std::endl;
  }
#endif

  return result;
}
} // namespace dc2_FS
} // namespace
