#pragma once
#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <memory>

#include "../MPI_Allreduce_group.hpp"
#include "../MPI_Datatype_wrapper.hpp"
#include "../cblas_lapacke_wrapper.hpp"
#include "../eigen_dc_interface.hpp"
#include "FS_const.hpp"
#include "FS_dividing.hpp"
#include "FS_prof.hpp"

namespace {
using FS_libs::FS_COMM_WORLD;

template <class Integer>
Integer get_pdc(Integer lctot, const eigen_mathlib_int ctot[], Integer npcol,
                Integer mycol) {
  Integer pdc = 0;
  for (Integer col = 0; col != mycol; col = (col + 1) % npcol) {
    pdc +=
        ctot[0 * lctot + col] + ctot[1 * lctot + col] + ctot[2 * lctot + col];
  }
  return pdc;
}

template <class Integer>
Integer get_pdr(Integer pdc, Integer klr, Integer mykl, Integer nprow,
                Integer myrow) {
  auto pdr = pdc;
  auto kl = klr + (mykl % nprow);
  for (Integer row = 0; row != myrow; row = (row + 1) % nprow) {
    pdr += kl;
    kl = klr;
  }
  return pdr;
}

/**
 * \brief pjcolのrowインデックスリストを作成
 */
template <class Integer>
Integer get_klr(Integer k, const eigen_mathlib_int indx[],
                const eigen_mathlib_int indcol[], Integer pjcol,
                eigen_mathlib_int indxr[]) {
  Integer klr = 0;
  for (Integer i = 0; i < k; i++) {
    const auto gi = indx[i];
    const auto row = indcol[gi];
    if (row == pjcol) {
      indxr[klr] = i;
      klr += 1;
    }
  }
  return klr;
}

/**
 * \brief 自身のCOLインデクスリストを作成
 */
template <class Integer>
void set_indxc(Integer k, const eigen_mathlib_int indx[],
               const eigen_mathlib_int indcol[], Integer mycol,
               eigen_mathlib_int indxc[]) {
  Integer klc = 0;
  for (Integer i = 0; i < k; i++) {
    const auto gi = indx[i];
    const auto col = indcol[gi];
    if (col == mycol) {
      indxc[klc] = i;
      klc += 1;
    }
  }
}

template <class Integer> class ComputeArea {
public:
  Integer np1; // 行列の上側
  Integer np2; // 行列の下側
};
template <class Integer>
ComputeArea<Integer> get_np12(Integer n, Integer n1, Integer np, Integer myrow,
                              const eigen_mathlib_int indrow[]) {
  Integer minrow = n - 1;
  Integer maxrow = 0;
  Integer npa = 0;
  Integer np1 = 0;
  Integer np2 = 0;
#pragma omp parallel for reduction(min : minrow) reduction(max : maxrow)       \
    reduction(+ : npa)
  for (Integer i = 0; i < n; i++) {
    if (indrow[i] == myrow) {
      minrow = std::min(minrow, (Integer)i);
      maxrow = std::max(maxrow, (Integer)i);
      npa += 1;
    }
  }
  if (minrow >= n1) {
    // 上側を持たない
    np1 = 0;
    np2 = npa;
  } else if (maxrow < n1) {
    // 下側を持たない
    np1 = npa;
    np2 = 0;
  } else {
    // 両側を持つ
    np1 = np / 2;
    np2 = npa - np / 2;
  }
  return ComputeArea<Integer>{np1, np2};
}

template <class Integer, class Float>
Integer FS_pdlaed3(Integer k, Integer n, Integer n1, Float d[], Float rho,
                   Float dlamda[], const Float w[], Integer ldq, Float q[],
                   const bt_node<Integer, Float> &subtree, Integer ldq2,
                   Float q2[], Integer ldu, Float u[], eigen_mathlib_int indx[],
                   Integer lctot, const eigen_mathlib_int ctot[],
                   Float q2buf1[], Float q2buf2[], Float z[], Float buf[],
                   eigen_mathlib_int indrow[], eigen_mathlib_int indcol[],
                   eigen_mathlib_int indxc[], eigen_mathlib_int indxr[],
                   FS_prof &prof) {
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlaed3 start" << std::endl;
  }
#endif
  Integer info = 0;
  { // FS_pdlaed3_end;
#if TIMER_PRINT
    prof.start(60);
#endif
    if (k == 0) {
      info = 0;
      goto FS_pdlead3_end;
    }

    const auto grid_info = subtree.FS_grid_info();
    const Integer nprow = grid_info.nprow;
    const Integer npcol = grid_info.npcol;
    const Integer myrow = grid_info.myrow;
    const Integer mycol = grid_info.mycol;
    const Integer nblk = subtree.FS_get_NBLK();
    const Integer nb = subtree.FS_get_NB();
    const Integer np = (nblk / nprow) * nb;
    const Integer nq = (nblk / npcol) * nb;

#pragma omp parallel for schedule(static, 1)
    for (Integer i = 0; i < n; i += nb) {
      Integer row = subtree.FS_info_G1L('R', i).rocsrc;
      Integer col = subtree.FS_info_G1L('C', i).rocsrc;
      for (Integer j = 0; j < nb; j++) {
        if (i + j < n) {
          indrow[i + j] = row;
          indcol[i + j] = col;
        }
      }
    }

    const auto mykl = ctot[0 * lctot + mycol] + ctot[1 * lctot + mycol] +
                      ctot[2 * lctot + mycol];
    const auto klr = mykl / nprow;
    const auto myklr = (myrow == 0) ? klr + (mykl % nprow) : klr;

    const auto pdc = get_pdc<Integer>(lctot, ctot, npcol, mycol);
    const auto pdr = get_pdr<Integer>(pdc, klr, mykl, nprow, myrow);

#pragma omp parallel for
    for (Integer i = 0; i < k; i++) {
      dlamda[i] = lapacke::lamc3<Float>(dlamda[i], dlamda[i]) - dlamda[i];
    }

#pragma omp parallel for
    for (Integer i = 0; i < 4 * k; i++) {
      buf[i] = FS_const::ZERO<Float>;
    }

#if TIMER_PRINT > 1
    prof.start(61);
#endif
    Integer sinfo = 0;
    if (myklr > 0) {
#pragma omp parallel reduction(max : sinfo)
      {
        std::unique_ptr<Float[]> sz(new Float[k]);
        std::fill_n(sz.get(), k, FS_const::ONE<Float>);
        std::unique_ptr<Float[]> sbuf(new Float[k]);
#pragma omp for schedule(static, 1)
        for (Integer i = 0; i < myklr; i++) {
          const Integer kk = pdr + i;
          const auto buf_index = (pdr - pdc + i);
          Float aux;
          Integer iinfo =
              lapacke::laed4<Float>(k, kk + 1, dlamda, w, sbuf.get(), rho, aux);
          if (k == 1 || k == 2) {
            buf[buf_index] = FS_const::ZERO<Float>;
            buf[mykl + buf_index] = aux;
          } else {
            auto sbufD_min = sbuf[kk];
            auto sbufB_min = dlamda[kk];
            if ((kk + 1) < k) {
              if (std::abs(sbuf[kk + 1]) < std::abs(sbufD_min)) {
                sbufD_min = sbuf[kk + 1];
                sbufB_min = dlamda[kk + 1];
              }
            }

            buf[buf_index] = sbufD_min;
            buf[mykl + buf_index] = sbufB_min;
          }
          if (iinfo != 0) {
            sinfo = kk;
            std::cout << "error" << std::endl;
          }
          // ..Compute part of z
#pragma loop nofp_relaxed nofp_contract noeval
          for (Integer j = 0; j < k; j++) {
            auto temp = dlamda[j] - dlamda[kk];
            if (j == kk) {
              temp = FS_const::ONE<Float>;
            } else {
              temp = temp;
            }
            sbuf[j] = sbuf[j] / temp;
          }
          for (Integer j = 0; j < k; j++) {
            sz[j] *= sbuf[j];
          }
        }
#pragma omp master
        {
          std::copy_n(sz.get(), k, z);
          // count up the flops on the Loewner law's update
          eigen_dc_interface::flops +=
              static_cast<double>(myklr) * static_cast<double>(k * 3);
        }
#pragma omp barrier
        for (Integer i = 1; i < omp_get_num_threads(); i++) {
          if (omp_get_thread_num() == i) {
            for (Integer i = 0; i < k; i++) {
              z[i] *= sz[i];
            }
          }
#pragma omp barrier
        }
      }
      info = sinfo;
    } else {
#pragma omp parallel for
      for (Integer i = 0; i < k; i++) {
        z[i] = FS_const::ONE<Float>;
      }
    }
#if TIMER_PRINT > 1
    prof.end(61);
#endif

#if TIMER_PRINT > 1
    prof.start(62);
#endif

#pragma omp parallel for
    for (Integer i = 0; i < k; i++) {
      buf[2 * k + i] = z[i];
    }

    MPI_Group_Allreduce<Float>(&buf[2 * k], z, k,
                               MPI_Datatype_wrapper::MPI_TYPE<Float>, MPI_PROD,
                               FS_COMM_WORLD, subtree.MERGE_GROUP_);

#pragma omp parallel for
    for (Integer i = 0; i < k; i++) {
      const auto sign = static_cast<Float>((w[i] >= 0) ? 1 : -1);
      z[i] = sign * std::abs(std::sqrt(-z[i]));
    }

    if (mykl > 0) {
      MPI_Group_Allreduce<Float>(buf, &buf[2 * k], 2 * mykl,
                                 MPI_Datatype_wrapper::MPI_TYPE<Float>, MPI_SUM,
                                 FS_COMM_WORLD, subtree.MERGE_GROUP_X_);
    }

#pragma omp parallel
    {
#pragma omp for
      for (Integer i = 0; i < 2 * mykl; i++) {
        buf[i] = FS_const::ZERO<Float>;
      }
#pragma omp for
      for (Integer i = 0; i < mykl; i++) {
        buf[pdc + i] = buf[2 * k + i];
        buf[k + pdc + i] = buf[2 * k + mykl + i];
      }
    }

    MPI_Group_Allreduce<Float>(buf, &buf[2 * k], 2 * k,
                               MPI_Datatype_wrapper::MPI_TYPE<Float>, MPI_SUM,
                               FS_COMM_WORLD, subtree.MERGE_GROUP_Y_);

    // Copy of D at the good place

    Float *sbufd = &buf[2 * k];
    Float *sbufb = &buf[3 * k];
    if (k == 1 || k == 2) {
      for (Integer i = 0; i < k; i++) {
        const auto gi = indx[i];
        d[gi] = sbufb[i];
      }
    } else {
#pragma omp parallel for
      for (Integer i = 0; i < k; i++) {
        const auto gi = indx[i];
        d[gi] = sbufb[i] - sbufd[i];
      }
    }

#if TIMER_PRINT > 1
    prof.end(62);
#endif

    set_indxc(k, indx, indcol, mycol, indxc);

#ifdef _BLOCKING_DGEMM
    // ブロッキングのために列方向昇順の逆引きリストを作成
#pragma omp parallel for
    for (Integer j = 0; j < mykl; j++) {
      const auto kk = indxc[j];
      const auto ju = indx[kk];
      const auto jju = subtree.FS_index_G2L('C', ju);
      indxcb[jju] = j;
    }
#endif

    const auto compute_area = get_np12<Integer>(n, n1, np, myrow, indrow);
    const auto np1 = compute_area.np1;
    const auto np2 = compute_area.np2;

    // Qの初期化
    // デフレーションされた列はQ2からコピー
#if TIMER_PRINT > 1
    prof.start(63);
#endif
#pragma omp parallel
#pragma omp for collapse(2)
    for (Integer j = 0; j < mykl; j++) {
      for (Integer i = 0; i < np; i++) {
        q[j * ldq + i] = FS_const::ZERO<Float>;
      }
    }
#pragma omp for collapse(2)
    for (Integer j = mykl; j < nq; j++) {
      for (Integer i = 0; i < np; i++) {
        q[j * ldq + i] = q2[j * ldq2 + i];
      }
    }
#if TIMER_PRINT > 1
    prof.end(63);
#endif

    // Compute eigenvectors of the modified rank-1 modification.

    MPI_Status stat;
    MPI_Request req[2];
    Integer nrecv = 0;
    Integer nsend = 0;
    for (Integer pj = 0; pj < npcol; pj++) {
      Float *sendq2 = nullptr;
      Float *recvq2 = nullptr;

      // 送受信バッファのポインタ
      if (pj % 2 == 0) {
        sendq2 = q2buf1;
        recvq2 = q2buf2;
      } else {
        sendq2 = q2buf2;
        recvq2 = q2buf1;
      }

      // 演算対象のプロセス列
      const auto pjcol = (mycol + npcol + pj) % npcol;

      // 処理する列数
      const auto mykl = ctot[0 * lctot + mycol] + ctot[1 * lctot + mycol] +
                        ctot[2 * lctot + mycol];
      const auto n12 = ctot[0 * lctot + pjcol] + ctot[1 * lctot + pjcol];
      const auto n23 = ctot[1 * lctot + pjcol] + ctot[2 * lctot + pjcol];

      // pjcolのrowインデックスリストを作成
      const auto klr = get_klr<Integer>(k, indx, indcol, pjcol, indxr);

      // 前ループの送受信のwaitと展開
      // wait -> recvbuf -> q2
      if (pj != 0 && npcol > 1) {
#if TIMER_PRINT > 1
        prof.start(64);
#endif
        // wait for irecv
        if (nrecv > 0) {
          MPI_Wait(&req[1], &stat);
        }

        // copy recvbuf -> q2
        // 上側
        if (np1 > 0) {
#pragma omp parallel for
          for (Integer jq2 = 0; jq2 < n12; jq2++) {
            const auto js = jq2 * np1;
            std::copy_n(&recvq2[js], np1, &q2[jq2 * ldq2 + 0]);
          }
        }
        // 下側
        if (np2 > 0) {
#pragma omp parallel for
          for (Integer j = 0; j < n23; j++) {
            const auto jq2 = j + ctot[0 * lctot + pjcol];
            const auto js = n12 * np1 + j * np2;
            std::copy_n(&recvq2[js], np2, &q2[jq2 * ldq2 + np1]);
          }
        }

        // wait for isend
        if (nsend > 0) {
          MPI_Wait(&req[0], &stat);
        }
#if TIMER_PRINT > 1
        prof.end(64);
#endif
      }

      // 送受信バッファのポインタの入れ替え
      if (pj % 2 == 0) {
        sendq2 = q2buf2;
        recvq2 = q2buf1;
      } else {
        sendq2 = q2buf1;
        recvq2 = q2buf2;
      }

      // 次ループの送受信
      // q2 -> sendbuf -> isend
      if (pj != npcol - 1 && npcol > 1) {
#if TIMER_PRINT > 1
        prof.start(65);
#endif
        // copy Q2 -> sendbuf
        // recvbufとsendbufを切り替えながら処理するためsendbufへの格納は初回のみ
        if (pj == 0) {
          // 上側
          if (np1 > 0) {
#pragma omp parallel for
            for (Integer jq2 = 0; jq2 < n12; jq2++) {
              const auto js = jq2 * np1;
              std::copy_n(&q2[jq2 * ldq2 + 0], np1, &sendq2[js]);
            }
          }
          // 下側
          if (np2 > 0) {
#pragma omp parallel for
            for (Integer j = 0; j < n23; j++) {
              const auto jq2 = j + ctot[0 * lctot + pjcol];
              const auto js = n12 * np1 + j * np2;
              std::copy_n(&q2[jq2 * ldq2 + np1], np2, &sendq2[js]);
            }
          }
        }
        // isend
        nsend = np1 * n12 + np2 * n23;
        if (nsend > 0) {
          const auto dstcol = (mycol + npcol - 1) % npcol; // 左側に送信
          const auto dest = subtree.group_Y_processranklist_[dstcol];
          MPI_Isend(sendq2, nsend, MPI_Datatype_wrapper::MPI_TYPE<Float>, dest,
                    1, FS_libs::FS_COMM_WORLD, &req[0]);
        }

        // irecv
        const auto pjcoln = (mycol + npcol + pj + 1) % npcol;
        nrecv = np1 * (ctot[0 * lctot + pjcoln] + ctot[1 * lctot + pjcoln]) +
                np2 * (ctot[1 * lctot + pjcoln] + ctot[2 * lctot + pjcoln]);
        if (nrecv > 0) {
          const auto srccol = (mycol + npcol + 1) % npcol; // 右側から受信
          const auto source = subtree.group_Y_processranklist_[srccol];
          MPI_Irecv(recvq2, nrecv, MPI_Datatype_wrapper::MPI_TYPE<Float>,
                    source, 1, FS_libs::FS_COMM_WORLD, &req[1]);
        }
#ifdef _MPITEST
        bool flag;
        if (nsend > 0) {
          MPI_Test(&req[0], &flag, &stat);
        }
        if (nrecv > 0) {
          MPI_Test(&req[1], &flag, &stat);
        }
#endif

#if TIMER_PRINT > 1
        prof.end(65);
#endif
      }

      if (mykl != 0) {
#if TIMER_PRINT > 1
        prof.start(66);
#endif
#pragma omp parallel
        {
          std::unique_ptr<Float[]> sbuf(new Float[k]);
#pragma omp for
          for (Integer j = 0; j < mykl; j++) {
            const auto kk = indxc[j];
            const auto ju = indx[kk];
            const auto jju = subtree.FS_index_G2L('C', ju);
            Float aux;
            if (k == 1 || k == 2) {
              lapacke::laed4<Float>(k, kk + 1, dlamda, w, sbuf.get(), rho, aux);
            } else {
              for (Integer i = 0; i < k; i++) {
                sbuf[i] = dlamda[i] - sbufb[kk];
                sbuf[i] += sbufd[kk];
              }
            }

            if (k == 1 || k == 2) {
              for (Integer i = 0; i < klr; i++) {
                const auto kk = indxr[i];
                const auto iu = indx[kk];
                const auto iiu = subtree.FS_index_G2L('C', iu);
                u[jju * ldu + iiu] = sbuf[kk];
              }
              continue;
            }
            for (Integer i = 0; i < k; i++) {
              sbuf[i] = z[i] / sbuf[i];
            }
            const auto temp = lapacke::nrm2<Float>(k, sbuf.get(), 1);

            for (Integer i = 0; i < klr; i++) {
              const auto kk = indxr[i];
              const auto iu = indx[kk];
              const auto iiu = subtree.FS_index_G2L('C', iu);
              u[jju * ldu + iiu] = sbuf[kk] / temp;
            }
          }
        }

        // Compute the updated eigenvectors.

        // 上側のdgemm
        if (np1 != 0 && n12 != 0) {
#if TIMER_PRINT > 1
          prof.start(67);
#endif
          lapacke::gemm<Float>(CblasNoTrans, CblasNoTrans, np1, mykl, n12,
                               FS_const::ONE<Float>, q2, ldq2, u, ldu,
                               FS_const::ONE<Float>, q, ldq);
          eigen_dc_interface::flops += 2 * static_cast<double>(np1) *
                                       static_cast<double>(mykl) *
                                       static_cast<double>(n12);
#if TIMER_PRINT > 1
          prof.end(67);
#endif
        }

        // 下側のdgemm
        if (np2 != 0 && n23 != 0) {
          const auto iq2 = np1;
          const auto jq2 = ctot[0 * lctot + pjcol];
          const auto iiu = ctot[0 * lctot + pjcol];
          const auto jju = 0;
          const auto iq = np1;
          const auto jq = 0;

#if TIMER_PRINT > 1
          prof.start(67);
#endif
          lapacke::gemm<Float>(CblasNoTrans, CblasNoTrans, np2, mykl, n23,
                               FS_const::ONE<Float>, &q2[jq2 * ldq2 + iq2],
                               ldq2, &u[jju * ldu + iiu], ldu,
                               FS_const::ONE<Float>, &q[jq * ldq + iq], ldq);
          eigen_dc_interface::flops += 2 * static_cast<double>(np2) *
                                       static_cast<double>(mykl) *
                                       static_cast<double>(n23);
#if TIMER_PRINT > 1
          prof.end(67);
#endif
        }
#if TIMER_PRINT > 1
        prof.end(66);
#endif
      }
      // ブロッキングなし終了
    }
  }
FS_pdlead3_end:
#if TIMER_PRINT
  prof.end(60);
#endif

#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlaed3 end. info= " << info << std::endl;
  }
#endif
  return info;
}
} // namespace
