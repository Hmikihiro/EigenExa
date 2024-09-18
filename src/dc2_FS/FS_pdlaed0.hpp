#pragma once
/**
 * @file FS_pdlaed0.hpp
 * @brief FS_pdlaed0
 */
#include <mpi.h>

#include "../FS_libs/FS_libs.hpp"
#include "../cblas_lapacke_wrapper.hpp"
#include "../eigen_libs0.hpp"
#include "FS2eigen_pdlasrt.hpp"
#include "FS_const.hpp"
#include "FS_dividing.hpp"
#include "FS_pdlaed1.hpp"
#include "FS_pdlasrt.hpp"
#include "FS_prof.hpp"

namespace {
namespace dc2_FS {
/**
 * subroutine FS_PDLAED0
 * @brief  @n
 * Purpose @n
 * ======= @n
 * FS_PDLAED0 computes all eigenvalues and corresponding eigenvectors of a @n
 * symmetric tridiagonal matrix using the divide and conquer method.
 *
 *  Arguments
 *  =========
 *
 * @param[in]     N      (global input) INTEGER @n
 *                       The order of the tridiagonal matrix T.  N >= 0.
 *
 * @param[in,out] D      (global input/output) DOUBLE PRECISION array,
 dimension (N) @n
 *                       On entry, the diagonal elements of the tridiagonal
 matrix.  @n
 *                       On exit, if INFO = 0, the eigenvalues in descending
 order.
 *
 * @param[in,out] E      (global input/output) DOUBLE PRECISION array,
 dimension (N-1) @n
 *                       On entry, the subdiagonal elements of the tridiagonal
 matrix. @n
 *                       On exit, E has been destroyed.
 *
 * @param[in,out] Q      (local output) DOUBLE PRECISION array, @n
 *                       global dimension (N, N), @n
 *                       local dimension (LDQ, NQ) @n
 *                       Q contains the orthonormal eigenvectors of the
 symmetric @n
 *                       tridiagonal matrix. @n
 *                       On output, Q is distributed across the P processes in
 non @n
 *                       block cyclic format.
 *
 * @param[in]     LDQ    (local input) INTEGER @n
 *                       The leading dimension of the array Q.  LDQ >=
 max(1,NP).
 *
 * @param         WORK   (local workspace/output) DOUBLE PRECISION array,
 dimension (LWORK)
 *
 * @param[in]     LWORK  (local input/output) INTEGER, the dimension of the
 array WORK. @n
 *                       LWORK = 1 + 6*N + 3*NP*NQ + NQ*NQ @n
 *                       LWORK can be obtained from subroutine FS_WorkSize.
 *
 * @param         IWORK  (local workspace/output) INTEGER array, dimension
 (LIWORK)
 *
 * @param[in]     LIWORK (input) INTEGER                   @n
 *                       The dimension of the array IWORK. @n
 *                       LIWORK = 1 + 8*N + 8*NPCOL        @n
 *                       LIWORK can be obtained from subroutine FS_WorkSize.
 *
 * @param[out]    prof   (global output) type(FS_prof) @n
 *                       profiling information of each subroutines.
 *
 * @return    INFO   (global output) INTEGER @n
 *                       = 0: successful exit   @n
 *                       /=0: error exit
 *
 * @note This routine is modified from ScaLAPACK PDLAED0.f
 *
 */

template <typename Integer, typename Real>
Integer FS_pdlaed0(const Integer n, Real d[], Real e[], Real q[],
                   const Integer ldq, Real work[], long lwork, Integer iwork[],
                   const long liwork, FS_prof &prof) {
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_PDLAED0 start." << std::endl;
  }
#endif
  bt_node<Integer, Real> root_node = {};
  const auto info = [&]() mutable -> Integer {
    auto nnod = (FS_libs::is_FS_comm_member()) ? FS_libs::FS_get_procs()
                                               : FS_libs::Nod{1, 1, 1};

    if (FS_libs::is_FS_comm_member()) {
#if TIMER_PRINT
      prof.start(20);
#endif
      std::unique_ptr<bool[]> hint(new bool[nnod.nod]);
      FS_create_hint(hint.get());
      const auto info_dividing =
          root_node.FS_dividing(n, d, e, std::move(hint), prof);
      if (info_dividing != 0) {
        return info_dividing;
      }
      const auto NBLK = root_node.FS_get_NBLK();
      const auto NB = root_node.FS_get_NB();
      const Integer NP = (NBLK / root_node.x_nnod_) * NB;
      const Integer NQ = (NBLK / root_node.y_nnod_) * NB;

#pragma omp parallel for collapse(2)
      for (Integer j = 0; j < NQ; j++) {
        for (Integer i = 0; i < NP; i++) {
          q[i + j * ldq] = FS_const::ZERO<Real>;
        }
      }

      const auto getleaf = root_node.FS_dividing_getleaf(0);
      const auto leafnode = getleaf.first;
      if (getleaf.second != 0) {
        return getleaf.second;
      }

#if TIMER_PRINT
      prof.start(28);
#endif

      const auto id = leafnode->nstart_;
      const auto mat_size = leafnode->FS_get_N_active();
      if (mat_size > 0) {
        const auto pq = root_node.FS_info_G2L(id, id);
        Integer info = 0;
        if (id < n - 1) {
          // iworkに残された情報はdstedcの外部で使用しないため、キャストして問題ない
          // sizeof(Integer) >= sizeof(eigen_mathlib_int)
          info = lapacke::stedc<Real>(
              'I', mat_size, &d[id], &e[id], &q[pq.row + pq.col * ldq], ldq,
              work, lwork, reinterpret_cast<eigen_mathlib_int *>(iwork),
              liwork);
        } else {
          q[pq.row + pq.col * ldq] = 1.0;
          info = 0;
        }

        if (info != 0) {
          return info;
        }
      }

#if TIMER_PRINT
      prof.end(28);
#endif
      for (auto node = leafnode; node->parent_node_ != nullptr;
           node = node->parent_node_) {
        const auto parent_node = node->parent_node_;
        const auto id = parent_node->nstart_;
        const auto n0 = parent_node->FS_get_N_active();
        const auto n1 = parent_node->sub_bt_node_[0].FS_get_N_active();

        if (n0 == n1) {
          continue;
        }

        const auto rho = e[id + n1 - 1];

        const auto q_top = parent_node->FS_get_QTOP();

#ifdef _DEBUGLOG
        const auto nb = parent_node->FS_get_NB();
        if (FS_libs::FS_get_myrank() == 0) {
          std::cout << "+---------------------" << std::endl;
          std::cout << "FS_PDLAED0 merge loop" << std::endl;
          std::cout << " layer = " << parent_node->layer_ << std::endl;
          std::cout << " N     = " << n0 << std::endl;
          std::cout << " NB    = " << nb << std::endl;
        }
#endif

        FS_prof prof_layer = {};
#if TIMER_PRINT
        prof_layer.init();
#endif
        Integer info_pdlaed1 = FS_pdlaed1<Integer, Real>(
            n0, n1, &d[id], &q[q_top.row + q_top.col * ldq], ldq, *parent_node,
            rho, work, iwork, prof_layer);
#if TIMER_PRINT > 2
        prof_layer.finalize();
#endif
#if TIMER_PRINT
        prof.add(prof_layer);
#endif
        if (info_pdlaed1 != 0) {
          return info_pdlaed1;
        }
        continue;
      }

#ifdef _DEBUGLOG
      if (FS_libs::FS_get_myrank() == 0) {
        std::cout << "+---------------------" << std::endl;
      }
#endif
    }

#ifdef _DEBUGLOG
    if (FS_libs::FS_get_myrank() == 0) {
      std::cout << "START Bcast" << std::endl;
    }
#endif

    const auto eigen_comm = eigen_libs0_wrapper::eigen_get_comm().eigen_comm;

    constexpr auto datatype = MPI_Datatype_wrapper::MPI_TYPE<typeof(nnod.x)>;
    MPI_Bcast(&nnod.x, 1, datatype, 0, eigen_comm);
    MPI_Bcast(&nnod.y, 1, datatype, 0, eigen_comm);

    const auto NBLK = root_node.FS_get_NBLK();
    const auto NB = root_node.FS_get_NB();
    const Integer NP = (NBLK / nnod.x) * NB;
    const Integer NQ = (NBLK / nnod.y) * NB;

    const auto ldq2 = NP;
    const Integer ipq2 = 0;

    const auto i_send_q = ipq2 + ldq2 * NQ;
    const auto i_recv_q = i_send_q + ldq2 * NQ;
    const auto i_buffer = i_recv_q + ldq2 * NQ;

    const Integer index_row = 0;
    const auto index_col = index_row + n;
    const auto index = index_col + n;
    const auto index_recv = index + n;

    const auto eigen_np = eigen_libs0_wrapper::eigen_get_procs().procs;

    if (nnod.nod == eigen_np) {
      FS_pdlasrt<Integer, Real>(
          n, d, q, ldq, root_node, &work[ipq2], ldq2, &work[i_send_q],
          &work[i_recv_q], &work[i_buffer], &iwork[index_row],
          &iwork[index_col], &iwork[index], &iwork[index_recv], prof);
    } else {
      Integer *ibuf = reinterpret_cast<Integer *>(work);
      auto *tbuf = reinterpret_cast<FS2eigen::GpositionValue<Integer, Real> *>(
          &ibuf[std::max((Integer)0, (NP * NQ))]);
      FS2eigen_pdlasrt<Integer, Real>(n, d, ldq, q, root_node, ibuf, work, tbuf,
                                      iwork, prof);
    }
    return 0;
  }();

  if (FS_libs::is_FS_comm_member()) {
    root_node.FS_dividing_free();
  }

#if TIMER_PRINT
  prof.end(20);
#endif
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_PDLAED0 end. info = " << info << std::endl;
  }
#endif
  return info;
}

} // namespace dc2_FS
} // namespace
