#pragma once

#include "FS_const.hpp"
#include <mpi.h>

#include <cstdio>
#include <new>

#include "../eigen_libs0.hpp"
template <class Float>
void eigen_FS(int n, int nvec, int a[], int lda, int w[], int z[], int ldz,
              int m_forward = eigen_libs0::eigen_NB_f,
              int m_backward = eigen_libs0::eigen_NB_b, char mode = 'A') {
  const auto nnod = eigen_libs0_wrapper::eigen_get_procs().procs;

  if ((n <= nnod) || (nnod < 4)) {
    // call eigen_s0
    return;
  }

  bool flag = false; // eigen_get_initialized(flag);
  if (!flag) {
    return;
  }

  if (MPI_COMM_NULL /* == TRD_COMM_WORLD*/) {
    return;
  }

  if (n <= 0) {
    std::printf("Warining: Negative dimesion is invalid!");
  }

  if (nvec == 0) {
    mode = 'N';
  }

  // eigen_get_matdims(n, nm, ny, m_f, m_b)
  int nm, ny;

  if (nm <= 0 || ny <= 0) {
    std::printf(
        "Warining: Problem size is too large for 32bit fortarn integer biry.");
    return;
  }

  const auto hs0 = 0; // eigen_get_wtime();
  Float ret_1 = FS_const::ZERO<Float>;
  Float ret_2 = FS_const::ZERO<Float>;
  Float ret_3 = FS_const::ZERO<Float>;

  const Float d = new (std::nothrow) Float(d[n]);
  const Float e = new (std::nothrow) Float(e[n]);
  if (d == nullptr || e == nullptr) {
    // call eigen_abort( "Memory allocation error [eigen_FS].", 1)
  }

  const auto world_size = -1; // TRD_nnod;
  const auto my_rank = -1;    // TRD_inod - 1;
}