#pragma once
/**
 * @file FS2eigen_pdlasrt.hpp
 * @brief FS2eigen_pdlasrt
 */

#include <mpi.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>

#include "../FS_libs/FS_libs.hpp"
#include "../MPI_Datatype_wrapper.hpp"
#include "../cblas_lapacke_wrapper.hpp"
#include "../eigen_libs0.hpp"
#include "FS_dividing.hpp"

namespace {
namespace dc2_FS {
namespace FS2eigen {

/**
 * @brief GRow と
 * GColはそれぞれ行列のindexであり、次数nを超えないから32bit整数を用いる
 *
 */
template <class Real> class GpositionValue {
public:
  int GRow;
  int GCol;
  Real MatrixValue;
};

template <class Integer, class Real> class CommBuf {
public:
  Integer rank;
  Integer Ndata;
  MPI_Request req;
  MPI_Status sta;
  bool flag;
  GpositionValue<Real> *bufp;
};

/**
 * @brief lidの要素はFS_nbrow_max * FS_nbcol_max 以下である
 *
 */
template <class Integer> class RANKLIST {
public:
  Integer index;
  int *lid;
};

template <class Integer> class NrankMaxsize {
public:
  Integer nrank;
  Integer maxsize;
};

template <class Integer>
NrankMaxsize<Integer>
get_nrank_maxsize(const Integer eigen_np,
                  const Integer comm_send_or_recv_info[]) {
  Integer send_or_recv_nrank = 0;
  Integer send_or_recv_maxsize = 1;
#pragma omp parallel for reduction(+ : send_or_recv_nrank)                     \
    reduction(max : send_or_recv_maxsize)
  for (Integer i = 0; i < eigen_np; i++) {
    if (comm_send_or_recv_info[i] != 0) {
      send_or_recv_nrank += 1;
      if (send_or_recv_maxsize < comm_send_or_recv_info[i]) {
        send_or_recv_maxsize = comm_send_or_recv_info[i];
      }
    }
  }
  return NrankMaxsize<Integer>{send_or_recv_nrank, send_or_recv_maxsize};
}

template <class Integer, class Real>
void init_send(const Integer np, const Integer comm_info[],
               CommBuf<Integer, Real> comm_buf[]) {
  Integer nrank = 0;
  for (Integer i = 0; i < np; i++) {
    if (comm_info[i] != 0) {
      comm_buf[nrank].flag = true;
      comm_buf[nrank].rank = static_cast<Integer>(i);
      comm_buf[nrank].Ndata = (comm_info[i]);
      comm_buf[nrank].bufp = nullptr;
      nrank += 1;
    }
  }
}
template <class Integer, class Real>
void init_recv(const Integer np, const Integer comm_info[],
               CommBuf<Integer, Real> comm_buf[]) {
  Integer nrank = 0;
  for (Integer i = 0; i < np; i++) {
    if (comm_info[i] != 0) {
      comm_buf[nrank].flag = true;
      comm_buf[nrank].rank = static_cast<Integer>(i);
      comm_buf[nrank].Ndata = comm_info[i];
      comm_buf[nrank].bufp = nullptr;
      nrank += 1;
    }
  }
}

template <class Integer, class Real>
void send(CommBuf<Integer, Real> &comm_send_data,
          const GpositionValue<Real> sendbuf[], const MPI_Comm comm) {
  auto ncount = sizeof(GpositionValue<Real>) * comm_send_data.Ndata;
  auto srank = comm_send_data.rank;
  comm_send_data.flag = false;
  MPI_Send((void *)sendbuf, ncount, MPI_BYTE, srank, 1, comm);
}
template <class Integer, class Real>
void irecv(CommBuf<Integer, Real> &comm_recv_data,
           GpositionValue<Real> recvbuf[], const MPI_Comm comm) {
  auto ncount = sizeof(GpositionValue<Real>) * comm_recv_data.Ndata;
  auto rrank = comm_recv_data.rank;
  comm_recv_data.flag = false;
  MPI_Irecv((void *)recvbuf, ncount, MPI_BYTE, rrank, 1, comm,
            &comm_recv_data.req);
}
template <class Integer, class Real>
Integer FS_nbroc_max(const char comp, const Integer n,
                     const bt_node<Integer, Real> &subtree,
                     const Integer FS_nbroc, const Integer FS_myroc) {
  for (Integer i = 0; i < FS_nbroc; i++) {
    auto lroc = i;
    auto groc = subtree.FS_index_L2G(comp, lroc, FS_myroc);
    if (n <= groc) {
      return i;
    }
  }
  return FS_nbroc;
}
template <class Integer, class Real>
Integer get_FS_nbrow_max(const Integer n, const bt_node<Integer, Real> &subtree,
                         const Integer FS_nbrow, const Integer FS_myrow) {
  return FS_nbroc_max('R', n, subtree, FS_nbrow, FS_myrow);
}
template <class Integer, class Real>
Integer get_FS_nbcol_max(const Integer n, const bt_node<Integer, Real> &subtree,
                         const Integer FS_nbcol, const Integer FS_mycol) {
  return FS_nbroc_max('C', n, subtree, FS_nbcol, FS_mycol);
}

template <class Integer>
Integer eigen_rank_xy2comm(const char grid_major, const Integer x_inod,
                           const Integer y_inod) {
  const auto procs = eigen_libs0_wrapper::eigen_get_procs();
  const auto x_nnod = procs.x_procs;
  const auto y_nnod = procs.y_procs;

  if (grid_major == 'R') {
    return y_inod + x_inod * y_nnod;
  } else {
    return x_inod + y_inod * x_nnod;
  }
}

/**
 * \brief 送信先が被らないように初回の通信相手を選択する
 */
template <class Integer, class Real>
Integer
select_first_communicater(const Integer send_nrank, const Integer eigen_np,
                          const Integer eigen_myrank,
                          const CommBuf<Integer, Real> comm_send_data[]) {
  Integer i0 = 0;
  if (0 < send_nrank) {
    i0 = -1;

    for (Integer k = 0; k < eigen_np; k++) {
      auto k1 = (k + eigen_myrank) % eigen_np;
      for (Integer i = 0; i < send_nrank; i++) {
        if (k1 == comm_send_data[i].rank) {
          i0 = i;
          break;
        }
      }
      if (i0 != -1) {
        break;
      }
    }
  }
  return i0;
}
} // namespace FS2eigen
} // namespace dc2_FS

namespace dc2_FS {
/**
 * subroutine FS2eigen_PDLASRT
 *
 *  @brief @n
 *  Purpose @n
 *  ======= @n
 *  FS2eigen_PDLASRT Sort the numbers in D in increasing order and the @n
 *  corresponding vectors in Q.
 *
 *  Arguments
 *  =========
 *
 * @param[in]     N        (global input) INTEGER @n
 *                         The number of columns to be operated on i.e the @n
 *                         number of columns of the distributed submatrix @n
 *                         sub( Q ). N >= 0.
 *
 * @param[in,out] D        (global input/output) DOUBLE PRECISION array,
 *                         dimension (N) @n
 *                         On exit, the number in D are sorted in increasing
 order.
 *
 * @param[in,out] Q        (input/output) DOUBLE PRECISION pointer into the
 *                          local memory @n
 *                         to an array of dimension (LDQ, NQ). This array
 *                          contains the   @n
 *                         local pieces of the distributed matrix sub( A ) to be
 *                          copied  @n
 *                         from.
 *
 * @param[in]     LDQ      (local input) INTEGER @n
 *                         The leading dimension of the array Q.  LDQ >=
 *                          max(1,NP).
 *
 * @param[in]     SUBTREE  (input) type(bt_node) @n
 *                         sub-tree information of merge block.
 *
 * @param         IBUF     (workspace) INTEGR array, dimension
 *                          (FS_NBROW*FS_NBCOL)
 *
 * @param         RBUF     (workspace) DOUBLE PRECISION array, dimension (N)
 *
 * @param         TBUF     (workspace) TYPE(GpositionValue) array, dimension
 *                          (FS_NBROW*FS_NBCOL)
 *
 * @param         INDX     (workspace) INTEGER array, dimension (N)
 *
 * @param[out]    prof     (global output) type(FS_prof) @n
 *                         profiling information of each subroutines.
 *
 * @return                 INFO     (global output) INTEGER @n
 *                          = 0: successful exit    @n
 *                         /=0: error exit
 *
 * @note This routine is modified from ScaLAPACK PDLASRT.f
 */
template <class Integer, class Real>
Integer FS2eigen_pdlasrt(const Integer n, Real d[], const Integer ldq, Real q[],
                         const bt_node<Integer, Real> &subtree, int ibuf[],
                         Real rbuf[], FS2eigen::GpositionValue<Real> tbuf[],
                         Integer indx[], FS_prof &prof) {
  double prof_time[40];
  for (Integer i = 0; i < 40; i++) {
    prof_time[i] = 0;
  }
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS2eigen_PDLASRT start." << std::endl;
  }
#endif
#if TIMER_PRINT
  prof.start(70);
#endif

  const auto eigen_procs = eigen_libs0_wrapper::eigen_get_procs();
  const Integer eigen_np = eigen_procs.procs;
  const Integer eigen_nprow = eigen_procs.x_procs;
  const Integer eigen_npcol = eigen_procs.y_procs;
  const Integer eigen_myrank = eigen_libs0_wrapper::eigen_get_id().id;
  const char eigen_grid_major = eigen_libs0_wrapper::eigen_get_grid_major();
  const auto eigen_comm = eigen_libs0_wrapper::eigen_get_comm().eigen_comm;

  Integer *comm_send_info = new Integer[eigen_np];
  std::unique_ptr<Integer[]> comm_recv_info(new Integer[eigen_np]);

#pragma omp parallel for
  for (Integer i = 0; i < eigen_np; i++) {
    comm_send_info[i] = 0;
    comm_recv_info[i] = -1;
  }
  MPI_Bcast(d, n, MPI_Datatype_wrapper::MPI_TYPE<Real>, 0, eigen_comm);

  std::iota(indx, indx + n, 0);
  std::sort(indx, &indx[n],
            [&d](Integer i1, Integer i2) { return d[i1] < d[i2]; });

#pragma omp parallel
  {
#pragma omp for
    for (Integer i = 0; i < n; i++) {
      rbuf[i] = d[indx[i]];
    }
#pragma omp for
    for (Integer i = 0; i < n; i++) {
      d[i] = rbuf[i];
    }
  }

  auto stime = MPI_Wtime();

  const auto FS_comm_member = FS_libs::is_FS_comm_member();

  const auto FS_grid_info =
      FS_comm_member ? subtree.FS_grid_info() : GridInfo<Integer>{0, 0, 0, 0};
  const Integer FS_nblk = FS_comm_member ? subtree.FS_get_NBLK() : 0;
  const Integer FS_nb = FS_comm_member ? subtree.FS_get_NB() : 0;
  const Integer FS_nbrow =
      FS_comm_member ? (FS_nblk / FS_grid_info.nprow) * FS_nb : 0;
  const Integer FS_nbcol =
      FS_comm_member ? (FS_nblk / FS_grid_info.npcol) * FS_nb : 0;
  const Integer FS_nbrow_max =
      FS_comm_member
          ? FS2eigen::get_FS_nbrow_max(n, subtree, FS_nbrow, FS_grid_info.myrow)
          : 0;
  // プロセス数で割り切れない場合に拡張した領域を除いた計算領域の総数
  const Integer FS_nbcol_max =
      FS_comm_member
          ? FS2eigen::get_FS_nbcol_max(n, subtree, FS_nbcol, FS_grid_info.mycol)
          : 0;

  auto etime = MPI_Wtime();
  prof_time[0] = etime - stime;
  stime = etime;

  std::unique_ptr<Integer[]> lcol2gcol_index;
  std::unique_ptr<Integer[]> lrow2grow_index;

  if (FS_comm_member) {
    lcol2gcol_index.reset(new Integer[FS_nbcol]);
#pragma omp parallel for
    for (Integer j = 0; j < FS_nbcol; j++) {
      lcol2gcol_index[j] = -1;
    }

    lrow2grow_index.reset(new Integer[FS_nbrow]);
#pragma omp parallel for
    for (Integer i = 0; i < FS_nbrow; i++) {
      lrow2grow_index[i] = -1;
    }

#pragma omp parallel for
    for (Integer lrow = 0; lrow < FS_nbrow_max; lrow++) {
      const auto grow = subtree.FS_index_L2G('R', lrow, FS_grid_info.myrow);
      lrow2grow_index[lrow] = grow;
    }

    // row <--> x
    // col <--> y
    // grow,
    // gcolはプロセス行で行列字数を割り切れない場合に対応するために拡張した字数の行列番号を返すので，
    // 拡張した範囲を省くために行列字数を超えたらexitする
#pragma omp parallel for reduction(+ : comm_send_info[0 : eigen_np])           \
    schedule(dynamic, 1)
    for (Integer lcol = 0; lcol < FS_nbcol_max; lcol++) {
      // 固有値を並び替える前の列番号を取得
      auto gcol = subtree.FS_index_L2G('C', lcol, FS_grid_info.mycol);

      // 固有値を並び替えた後の列番号に変換
      for (Integer k = 0; k < n; k++) {
        if (indx[k] == gcol) {
          gcol = k;
          lcol2gcol_index[lcol] = k;
          break;
        }
      }

      // グローバル情報から再分散後に担当するノードを求める
      const auto pcol =
          eigen_libs0_wrapper::eigen_owner_node(gcol, eigen_npcol);
      for (Integer lrow = 0; lrow < FS_nbrow_max; lrow++) {
        const auto grow = lrow2grow_index[lrow];
        const auto prow =
            eigen_libs0_wrapper::eigen_owner_node(grow, eigen_nprow);
        const auto pn =
            FS2eigen::eigen_rank_xy2comm(eigen_grid_major, prow, pcol);
        comm_send_info[pn] += 1;
      }
    }
  }

  etime = MPI_Wtime();
  prof_time[1] = etime - stime;
  stime = etime;

  MPI_Alltoall(comm_send_info, 1, MPI_Datatype_wrapper::MPI_TYPE<Integer>,
               comm_recv_info.get(), 1, MPI_Datatype_wrapper::MPI_TYPE<Integer>,
               eigen_comm);

  etime = MPI_Wtime();
  prof_time[2] = etime - stime;
  stime = etime;

  const auto send_nrank_maxsize =
      FS2eigen::get_nrank_maxsize<Integer>((Integer)eigen_np, comm_send_info);
  const auto send_nrank = send_nrank_maxsize.nrank; // 送信相手の総数
  const auto send_maxsize = send_nrank_maxsize.maxsize;

  etime = MPI_Wtime();
  prof_time[3] = etime - stime;
  stime = etime;

  Integer pointer = 0;
  std::unique_ptr<FS2eigen::CommBuf<Integer, Real>[]> comm_send_data;
  std::unique_ptr<FS2eigen::RANKLIST<Integer>[]> sendrank_list;
  FS2eigen::GpositionValue<Real> *sendbuf;

  if (send_nrank != 0) {
    comm_send_data.reset(new FS2eigen::CommBuf<Integer, Real>[send_nrank]);
    FS2eigen::init_send<Integer, Real>(eigen_np, comm_send_info,
                                       comm_send_data.get());

    sendrank_list.reset(new FS2eigen::RANKLIST<Integer>[send_nrank]);

    for (Integer k = 0; k < send_nrank; k++) {
      sendrank_list[k].lid = &ibuf[pointer];
      pointer += comm_send_data[k].Ndata;
      for (Integer j = 0; j < comm_send_data[k].Ndata; j++) {
        sendrank_list[k].lid[j] = -1;
      }
    }
    pointer = (pointer + 4) / 4;

    sendbuf = &tbuf[pointer];

    pointer += send_maxsize;
#pragma omp parallel for
    for (Integer i = 0; i < send_maxsize; i++) {
      sendbuf[i].GRow = -1;
      sendbuf[i].GCol = -1;
      sendbuf[i].MatrixValue = 0;
    }
  }

  etime = MPI_Wtime();
  prof_time[4] = etime - stime;
  stime = etime;

  const auto recv_nrank_maxsize = FS2eigen::get_nrank_maxsize<Integer>(
      (Integer)eigen_np, comm_recv_info.get());
  const auto recv_nrank = recv_nrank_maxsize.nrank; // 受信相手の総数

  etime = MPI_Wtime();
  prof_time[5] = etime - stime;
  stime = etime;

  // 受信バッファの設定
  if (!FS_comm_member) {
    pointer = 0;
  }
  std::unique_ptr<FS2eigen::CommBuf<Integer, Real>[]> comm_recv_data;
  if (recv_nrank != 0) {
    comm_recv_data.reset(new FS2eigen::CommBuf<Integer, Real>[recv_nrank]);
    FS2eigen::init_recv<Integer, Real>(eigen_np, comm_recv_info.get(),
                                       comm_recv_data.get());
    for (Integer k = 0; k < recv_nrank; k++) {
      if (comm_recv_data[k].rank + 1 != eigen_myrank) {
        comm_recv_data[k].bufp = &tbuf[pointer];
        pointer += comm_recv_data[k].Ndata;
        irecv(comm_recv_data[k], comm_recv_data[k].bufp, eigen_comm);
      }
    }
  }

  // 送信データのパック
  if (FS_comm_member) {
    for (Integer k = 0; k < send_nrank; k++) {
      auto pn = comm_send_data[k].rank;
      comm_send_info[pn] = k;
      sendrank_list[k].index = 0;
    }

    for (Integer lcol = 0; lcol < FS_nbcol_max; lcol++) {
      for (Integer lrow = 0; lrow < FS_nbrow_max; lrow++) {
        auto gcol = lcol2gcol_index[lcol];
        auto pcol = eigen_libs0_wrapper::eigen_owner_node(gcol, eigen_npcol);

        auto grow = lrow2grow_index[lrow];
        auto prow = eigen_libs0_wrapper::eigen_owner_node(grow, eigen_nprow);

        auto pn = FS2eigen::eigen_rank_xy2comm(eigen_grid_major, prow, pcol);

        const auto &info = comm_send_info[pn];

        sendrank_list[info].lid[sendrank_list[info].index] =
            lrow + lcol * FS_nbrow_max;
        sendrank_list[info].index += 1;
      }
    }

    etime = MPI_Wtime();
    prof_time[11] = etime - stime;
    stime = etime;

    const auto i0 = FS2eigen::select_first_communicater<Integer>(
        send_nrank, eigen_np, eigen_myrank, comm_send_data.get());
    for (Integer k0 = i0; k0 < send_nrank + i0; k0++) {
      const auto k = (k0 + 1) % send_nrank;
      if (comm_send_data[k].rank + 1 != eigen_myrank) {
        for (Integer j = 0; j < comm_send_data[k].Ndata; j++) {
          const auto lcol = sendrank_list[k].lid[j] / FS_nbrow_max;
          const auto gcol = lcol2gcol_index[lcol];
          const auto lrow = sendrank_list[k].lid[j] % FS_nbrow_max;
          const auto grow = lrow2grow_index[lrow];

          sendbuf[j].GRow = grow;
          sendbuf[j].GCol = gcol;
          sendbuf[j].MatrixValue = q[lcol * ldq + lrow];
        }
        send(comm_send_data[k], sendbuf, eigen_comm);
      }
    }

    // Recv側のWait
    for (Integer k0 = 0; k0 < recv_nrank; k0++) {
      if (comm_recv_data[k0].rank + 1 != eigen_myrank) {
        MPI_Wait(&comm_recv_data[k0].req, MPI_STATUS_IGNORE);
      }
    }

    // 送信データのアンパック
    // 送信先が自分自身であればデータコピー
    for (Integer k0 = 0; k0 < send_nrank; k0++) {
      if (eigen_myrank == comm_send_data[k0].rank + 1) {
        for (Integer i = 0; i < comm_send_data[k0].Ndata; i++) {
          const auto lcol = sendrank_list[k0].lid[i] / FS_nbrow_max;
          const auto gcol = lcol2gcol_index[lcol];
          const auto lrow = sendrank_list[k0].lid[i] % FS_nbrow_max;
          const auto grow = lrow2grow_index[lrow];

          sendbuf[i].GRow = grow;
          sendbuf[i].GCol = gcol;
          sendbuf[i].MatrixValue = q[lcol * ldq + lrow];
        }
        for (Integer i = 0; i < comm_send_data[k0].Ndata; i++) {
          const auto gcol = sendbuf[i].GCol;
          const auto grow = sendbuf[i].GRow;
          const auto lcol = eigen_libs0_wrapper::eigen_translate_g2l<Integer>(
              gcol, eigen_npcol);
          const auto lrow = eigen_libs0_wrapper::eigen_translate_g2l<Integer>(
              grow, eigen_nprow);
          q[lcol * ldq + lrow] = sendbuf[i].MatrixValue;
        }
        break;
      }
    }

#pragma omp parallel for schedule(dynamic, 1)
    for (Integer k = 0; k < recv_nrank; k++) {
      if (comm_recv_data[k].rank + 1 != eigen_myrank) {
        const auto Ndata = comm_recv_data[k].Ndata;
        for (Integer i = 0; i < Ndata; i++) {
          const auto gcol = comm_recv_data[k].bufp[i].GCol;
          const auto grow = comm_recv_data[k].bufp[i].GRow;
          const auto lcol = eigen_libs0_wrapper::eigen_translate_g2l<Integer>(
              gcol, eigen_npcol);
          const auto lrow = eigen_libs0_wrapper::eigen_translate_g2l<Integer>(
              grow, eigen_nprow);
          q[lcol * ldq + lrow] = comm_recv_data[k].bufp[i].MatrixValue;
        }
      }
    }
  } else {
    if (recv_nrank != 0) {
      for (Integer k = 0; k < recv_nrank; k++) {
        MPI_Wait(&comm_recv_data[k].req, MPI_STATUS_IGNORE);
        const auto Ndata = comm_recv_data[k].Ndata;
#pragma omp parallel for
        for (Integer i = 0; i < Ndata; i++) {
          const auto gcol = comm_recv_data[k].bufp[i].GCol;
          const auto grow = comm_recv_data[k].bufp[i].GRow;
          const auto lcol = eigen_libs0_wrapper::eigen_translate_g2l<Integer>(
              gcol, eigen_npcol);
          const auto lrow = eigen_libs0_wrapper::eigen_translate_g2l<Integer>(
              grow, eigen_nprow);
          q[lcol * ldq + lrow] = comm_recv_data[k].bufp[i].MatrixValue;
        }
      }
    }
  }

  etime = MPI_Wtime();
  prof_time[6] = etime - stime;
  stime = etime;
  delete[] comm_send_info;
  etime = MPI_Wtime();
  prof_time[8] = etime - stime;
  stime = etime;

#if TIMER_PRINT
  prof.end(70);
#endif
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::cout << "FS_pdlasrt end." << std::endl;
  }
#endif
  return 0;
}

} // namespace dc2_FS
} // namespace
