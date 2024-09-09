#pragma once
#include <memory>
#include <mpi.h>

#include "../MPI_Datatype_wrapper.hpp"
#include "../eigen/eigen_dc_interface.hpp"
#include "../eigen/eigen_libs0.hpp"
#include "FS_EDC.hpp"
#include "FS_libs.hpp"
#include "eigen_devel_FS_wrapper.hpp"

#if TIMER_PRINT > 1
#include <cstdio>
#endif

namespace eigen_FS {
using eigen_devel_FS_wrapper::FS_eigen_abort;
using eigen_devel_FS_wrapper::FS_eigen_timer_reset;
using eigen_libs0_wrapper::eigen_get_comm;
using eigen_libs0_wrapper::eigen_get_id;
using eigen_libs0_wrapper::eigen_get_procs;
using std::unique_ptr;

template <class Integer, class Float>
void dc2_FS(Integer n, Integer nvec, Float d[], Float e[], Float z[],
            Integer ldz, long *info, Float *ret) {
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

  const auto worksize = FS_libs::FS_WorkSize(n);
  long long lwork_ = worksize.lwork;
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

  try {
    unique_ptr<Float[]> work(new Float[lwork]);
    unique_ptr<Integer[]> iwork(new Integer[liwork]);

#if defined(__INTEL_COMPILER) && USE_MKL
    const auto mkl_mode = mkl_get_Dynamic();
    MKL_Set_Dynamic(0);
#endif

    FS_prof prof = {};
#if TIMER_PRINT

    prof.init();
#endif
    *info = FS_EDC::FS_EDC<Integer, Float>(n, d, e, z, ldz, work.get(), lwork,
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
    const auto iam = eigen_get_id().id - 1;
    if (iam == 0) {
      printf("FS_EDC     %f\n", prof.region_time[10]);
    }
#endif

  } catch (std::bad_alloc) {
    FS_eigen_abort();
  }

#if TIMER_PRINT > 1
  if (iam == 0) {
    printf("FS_dividing %f\n", eigen_dc_interface::p_time0);
    printf("FS_pdlasrt  %f\n", eigen_dc_interface::p_timer);
    printf("FS_pdlaed2  %f\n", eigen_dc_interface::p_time2);
    printf("FS_pdlaed3  %f\n", eigen_dc_interface::p_time3);
    printf("FS_pdlaedz  %f\n", eigen_dc_interface::p_timez);
    printf("DGEMM       %f\n", eigen_dc_interface::dgemm_time);
  }
#endif

  MPI_Allreduce(&eigen_dc_interface::flops, ret, 1,
                MPI_Datatype_wrapper::MPI_TYPE<Float>, MPI_SUM, eigen_comm);

  return;
}
} // namespace eigen_FS
