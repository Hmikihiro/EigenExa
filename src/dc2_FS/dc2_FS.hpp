#pragma once
#include <memory>
#include <mpi.h>

#include "../FS_libs/FS_libs.hpp"
#include "../MPI_Datatype_wrapper.hpp"
#include "../cblas_lapacke_wrapper.hpp"
#include "../eigen_dc_interface.hpp"
#include "../eigen_libs0.hpp"
#include "FS_EDC.hpp"
#include "eigen_devel_FS_wrapper.hpp"

#if TIMER_PRINT > 1
#include <iostream>
#endif

namespace {
using eigen_devel_FS_wrapper::FS_eigen_abort;
using eigen_devel_FS_wrapper::FS_eigen_timer_reset;
using eigen_libs0_wrapper::eigen_get_comm;
using eigen_libs0_wrapper::eigen_get_id;
using eigen_libs0_wrapper::eigen_get_procs;

template <class Integer, class Float>
static long long buffer_for_gposition_value =
    3; // GpositionValueの増加分int64と floatの時に確保する値
template <>
static long long buffer_for_gposition_value<long, float> =
    3; // 計算途中のtbufにおいて、使用するこの時、従来int,int,doubleだったものを使っているため、float対応でメモリ半分を想定していない。
template <> static long long buffer_for_gposition_value<int, float> = 2;
template <> static long long buffer_for_gposition_value<int, double> = 1;
template <>
static long long buffer_for_gposition_value<long, double> =
    2; // 1.5倍で十分だが、切り上げて2
template <class Integer, class Float>
Integer dc2_FS(Integer n, Integer nvec, Float d[], Float e[], Float z[],
               Integer ldz, Float *ret) {
  eigen_dc_interface::flops = 0;
  eigen_dc_interface::dgemm_time = 0;
  eigen_dc_interface::p_time0 = 0;
  eigen_dc_interface::p_timer = 0;
  eigen_dc_interface::p_time2 = 0;
  eigen_dc_interface::p_time3 = 0;
  eigen_dc_interface::p_times = 0;
  eigen_dc_interface::p_timez = 0;
  FS_eigen_timer_reset(1, 0, 0, 0);

  const auto eigen_comm = eigen_get_comm().eigen_comm;
#if TIMER_PRINT > 1
  const auto iam = eigen_get_id().id - 1;
#endif

  const auto worksize = FS_libs::FS_WorkSize(n);
  long long lwork_ = worksize.lwork * 3;
  long long liwork_ = worksize.liwork;

  const int FS_COMM_MEMBER = FS_libs::is_FS_comm_member();
  if (!FS_COMM_MEMBER) {
    lwork_ = 0;
    liwork_ = 0;
  }
  long long lwork;
  long long liwork;
  MPI_Allreduce(&lwork_, &lwork, 1, MPI_LONG_LONG, MPI_MAX, eigen_comm);
  MPI_Allreduce(&liwork_, &liwork, 1, MPI_LONG_LONG, MPI_MAX, eigen_comm);

  Integer info_fs_edc = 0;
  try {
    std::unique_ptr<Float[]> work(new Float[lwork]);
    std::unique_ptr<eigen_mathlib_int[]> iwork(new eigen_mathlib_int[liwork]);

#if defined(__INTEL_COMPILER) && USE_MKL
    const auto mkl_mode = mkl_get_Dynamic();
    MKL_Set_Dynamic(0);
#endif

    FS_prof prof = {};
#if TIMER_PRINT

    prof.init();
#endif
    info_fs_edc = FS_EDC<Integer, Float>(n, d, e, z, ldz, work.get(), lwork,
                                         iwork.get(), liwork, &prof);
#if TIMER_PRINT
    prof.finalize();
#endif
#if TIMER_PRINT > 1
    eigen_dc_interface::p_time0 = prof.region_time[21];
    eigen_dc_interface::p_timer = prof.region_time[70];
    eigen_dc_interface::p_time2 = prof.region_time[50];
    eigen_dc_interface::p_time3 = prof.region_time[60];
    eigen_dc_interface::dgemm_time = prof.region_time[67];
    eigen_dc_interface::p_timez = prof.region_time[40];
#endif
#if defined(__INTEL_COMPILER) && USE_MKL
    MKL_Set_Dynamic(mkl_mode);
#endif

#if TIMER_PRINT > 1
    if (iam == 0) {
      std::cout << "FS_EDC     " << prof.region_time[10] << std::endl;
    }
#endif

  } catch (std::bad_alloc) {
    FS_eigen_abort();
    throw;
  }

#if TIMER_PRINT > 1
  if (iam == 0) {
    std::cout << "FS_dividing " << eigen_dc_interface::p_time0 << std::endl;
    std::cout << "FS_pdlasrt  " << eigen_dc_interface::p_timer << std::endl;
    std::cout << "FS_pdlaed2  " << eigen_dc_interface::p_time2 << std::endl;
    std::cout << "FS_pdlaed3  " << eigen_dc_interface::p_time3 << std::endl;
    std::cout << "FS_pdlaedz  " << eigen_dc_interface::p_timez << std::endl;
    std::cout << "DGEMM       " << eigen_dc_interface::dgemm_time << std::endl;
  }
#endif

  MPI_Allreduce(&eigen_dc_interface::flops, ret, 1,
                MPI_Datatype_wrapper::MPI_TYPE<Float>, MPI_SUM, eigen_comm);

  return info_fs_edc;
}
} // namespace
