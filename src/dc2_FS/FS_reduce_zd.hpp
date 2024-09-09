#pragma once

#include "../FS_libs/FS_libs.hpp"
#include "../MPI_Allreduce_group.hpp"
#include "../MPI_Datatype_wrapper.hpp"
#include "FS_dividing.hpp"
#include "FS_prof.hpp"

#if defined(_DEBUGLOG)
#include <cstdio>
#endif

namespace eigen_FS {
using FS_dividing::bt_node;
using FS_libs::FS_COMM_WORLD;
using FS_prof::FS_prof;
template <class Integer, class Float>
void FS_reduce_zd(Integer n, const bt_node<Integer, Float> &subtree,
                  Float work[], Float z[], Float d[], FS_prof &prof) {
#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::printf("FS_reduce_zd start.\n");
  }
#endif
#if TIMER_PRINT
  prof.start(45);
#endif
#if TIMER_PRINT
#ifdef PROF_DETAIL
  prof.start(46);
  MPI_Barrier(subtree.MERGE_COMM);
  prof.end(46);
  prof.start(47);
#endif
#endif

  // reduce
  MPI_Group_Allreduce(work, z, n * 2, MPI_Datatype_wrapper::MPI_TYPE<Float>,
                      MPI_SUM, FS_COMM_WORLD, subtree.MERGE_GROUP_);

#pragma omp parallel for
  for (Integer j = 0; j < n; j++) {
    d[j] = z[j + n];
  }

#if TIMER_PRINT
#ifdef PROF_DETAIL
  prof.end(47);
#endif
#endif

#if TIMER_PRINT
  prof.end(45);
#endif

#ifdef _DEBUGLOG
  if (FS_libs::FS_get_myrank() == 0) {
    std::printf("FS_reduce_zd end.\n");
  }
#endif
}

} // namespace eigen_FS
