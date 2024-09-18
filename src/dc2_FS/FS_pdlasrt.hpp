#pragma once

#include <mpi.h>

#include <algorithm>
#include <numeric>

#include "../FS_libs/FS_libs.hpp"
#include "../MPI_Datatype_wrapper.hpp"
#include "../cblas_lapacke_wrapper.hpp"
#include "FS_dividing.hpp"
#include "FS_prof.hpp"

namespace {
namespace dc2_FS {
template <class Integer, class Float>
void FS_pdlasrt(const Integer n, Float d[], Float q[], const Integer ldq,
                const bt_node<Integer, Float> &subtree, Float q2[],
                const Integer ldq2, Float sendq[], Float recvq[], Float buf[],
                Integer indrow[], Integer indcol[], Integer indx[],
                Integer indrcv[], FS_prof &prof) {
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlasrt start." << std::endl;
  }
#endif
#if TIMER_PRINT
  prof.start(70);
#endif
  //

  const auto grid_info = subtree.FS_grid_info();
  const auto nprow = grid_info.nprow;
  const auto npcol = grid_info.npcol;
  const auto myrow = grid_info.myrow;
  const auto mycol = grid_info.mycol;
  const auto nblk = subtree.FS_get_NBLK();
  const auto nb = subtree.FS_get_NB();
  const auto np = (nblk / nprow) * nb;
  const auto nq = (nblk / npcol) * nb;
//
#pragma omp parallel for schedule(static, 1)
  for (Integer i = 0; i < n; i += nb) {
    const auto row = subtree.FS_info_G1L('R', i).rocsrc;
    const auto col = subtree.FS_info_G1L('C', i).rocsrc;
    for (Integer j = 0; j < nb; j++) {
      if (i + j < n) {
        indrow[i + j] = row;
        indcol[i + j] = col;
      }
    }
  }
  //
  // Sort the eigenvalues in D
  //
  std::iota(indx, indx + n, 0);
  std::sort(indx, indx + n,
            [&d](Integer i1, Integer i2) { return d[i1] < d[i2]; });

#pragma omp parallel
  {
#pragma omp for
    for (Integer i = 0; i < n; i++) {
      buf[i] = d[indx[i]];
    }
#pragma omp for
    for (Integer i = 0; i < n; i++) {
      d[i] = buf[i];
    }
  }

  // 列方向の入れ替え
  // 固有値昇順に合わせたソートと非ブロックサイクリック化
  for (Integer pj = 0; pj < npcol; pj++) {
    // 入れ替え対象のプロセス列
    // デッドロックしないように逆順に回す
    const auto pjcol = (npcol - 1 - mycol - pj + npcol) % npcol;

    //
    Integer nsend = 0;
    Integer nrecv = 0;
    for (Integer j = 0; j < n; j++) {
      // 元のグローバルインデクス
      const auto gi = indx[j];

      // 元のグローバルインデクスを保持するプロセス列
      const auto col = indcol[gi];

      // 入れ替え先インデクスJを保持するプロセス列
      const auto jcol = j % npcol;
      // PJCOLに送信する列
      if (col == mycol && jcol == pjcol) {
        // ローカルインデクス
        const auto jl = subtree.FS_index_G2L('C', gi);

        // 送信バッファに格納

        lapacke::copy(np, &q[jl * ldq], 1, &sendq[nsend * np], 1);
        nsend += 1;
      }

      // PJCOLから受信する列
      if (col == pjcol && jcol == mycol) {
        // 格納先のローカルインデクス
        indrcv[nrecv] = j / npcol;

        // 受信数
        nrecv += 1;
      }
    }

    // irecv
    MPI_Request req;
    if (nrecv > 0) {
      MPI_Irecv(recvq, np * nrecv, MPI_Datatype_wrapper::MPI_TYPE<Float>,
                subtree.group_Y_processranklist_[pjcol], 1,
                FS_libs::FS_COMM_WORLD, &req);
    }

    // send
    if (nsend > 0) {
      MPI_Send(sendq, np * nsend, MPI_Datatype_wrapper::MPI_TYPE<Float>,
               subtree.group_Y_processranklist_[pjcol], 1,
               FS_libs::FS_COMM_WORLD);
    }

    // waitと展開
    MPI_Status stat;
    if (nrecv > 0) {
      MPI_Wait(&req, &stat);
#pragma omp parallel for
      for (Integer j = 0; j < nrecv; j++) {
        const auto jl = indrcv[j];
        lapacke::copy(np, &recvq[j * np], 1, &q2[jl * ldq2], 1);
      }
    }
  }
  //======================================================================
  // 列方向の入れ替え
  // 非ブロックサイクリック化
  for (Integer pj = 0; pj < nprow; pj++) {
    // 入れ替え対象のプロセス行
    // デッドロックしないように逆順に回す
    const auto pjrow = (nprow - 1 - myrow - pj + nprow) % nprow;

    //
    Integer nsend = 0;
    Integer nrecv = 0;
    for (Integer i = 0; i < n; i++) {
      // 元のグローバルインデクスを保持するプロセス行
      const auto row = indrow[i];
      // 入れ替え先インデクスIを保持するプロセス行
      const auto irow = i % nprow;

      // PJROWに送信する行
      if (row == myrow && irow == pjrow) {
        // ローカルインデクス
        const auto il = subtree.FS_index_G2L('R', i);

        // 送信バッファに格納
        lapacke::copy<Float>(nq, &q2[il], ldq2, &sendq[nsend * nq], 1);
        nsend += 1;
      }

      // PJROWから受信する例
      if (row == pjrow && irow == myrow) {
        // 格納先のローカルインデクス
        indrcv[nrecv] = i / nprow;

        // 受信数
        nrecv += 1;
      }
    }
    MPI_Request req;
    // irecv
    if (nrecv > 0) {
      MPI_Irecv(recvq, nrecv * nq, MPI_Datatype_wrapper::MPI_TYPE<Float>,
                subtree.group_X_processranklist_[pjrow], 1,
                FS_libs::FS_COMM_WORLD, &req);
    }

    // send
    if (nsend > 0) {
      MPI_Send(sendq, nsend * nq, MPI_Datatype_wrapper::MPI_TYPE<Float>,
               subtree.group_X_processranklist_[pjrow], 1,
               FS_libs::FS_COMM_WORLD);
    }

    // waitと展開
    if (nrecv > 0) {
      MPI_Status stat;
      MPI_Wait(&req, &stat);

#pragma omp parallel for
      for (Integer i = 0; i < nrecv; i++) {
        const auto il = indrcv[i];
        lapacke::copy<Float>(nq, &recvq[i * nq], 1, &q[il], ldq);
      }
    }
  }

#if TIMER_PRINT
  prof.end(70);
#endif

#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlasrt end." << std::endl;
  }
#endif
}
} // namespace dc2_FS
} // namespace
