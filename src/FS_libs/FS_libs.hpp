#pragma once

#include <mpi.h>

#include <cmath>

#include "FS_libs_type.hpp"
#include "eigen_libs_FS_wrapper.hpp"

namespace FS_libs {
extern process_info FS_info;
namespace {
inline void FS_init(MPI_Comm comm = MPI_COMM_WORLD,
                    FS_GRID_MAJOR order = FS_GRID_MAJOR::COLUMN) {
  auto comm0 = comm;
  eigen_libs_FS_wrapper::eigen_init0(comm0, static_cast<char>(order));
  FS_info = process_info(order);
}

inline void FS_free() {
  eigen_libs_FS_wrapper::eigen_free0();
  FS_info.comm_free();
}

inline Nod FS_get_procs() {
  Nod nnod = {};
  const auto FS_node = FS_info.get_node();
  nnod.nod = FS_node.nnod;
  nnod.x = FS_node.x_nnod;
  nnod.y = FS_node.y_nnod;
  return nnod;
}

inline Nod FS_get_id() {
  Nod inod = {};
  const auto FS_node = FS_info.get_node();
  inod.nod = FS_node.inod;
  inod.x = FS_node.x_inod;
  inod.y = FS_node.y_inod;
  return inod;
}

struct matdims {
  int nx;
  int ny;
};

inline matdims FS_get_matdims(int n) {
  const auto nnod = FS_get_procs();
  int n1 = n / nnod.nod;
  if (n % nnod.nod != 0) {
    n1 += 1;
  }
  const auto nx = n1 * (nnod.nod / nnod.x);
  const auto ny = n1 * (nnod.nod / nnod.y);
  return {nx, ny};
}

inline FS_GRID_MAJOR FS_get_grid_major() { return FS_info.get_grid_major(); }

struct fs_worksize {
  long lwork;
  long liwork;
};

inline fs_worksize FS_WorkSize(int n, int real_byte_size) {
  const auto y_nnod = FS_get_procs().y;
  const auto dims = FS_get_matdims(n);
  const auto np = dims.nx;
  const auto nq = dims.ny;

  // FS2eigen_pdlasrtのtbufで使用するGpositionValueのサイズ
  const auto default_size = 16;  // 4 + 4 + 8
  const auto actual_size = real_byte_size == 4 ? 12 : 16;

  const double padding_rate = ((double)sizeof(double) / real_byte_size) *
                              ((double)actual_size / default_size);

  const long lwork =
      std::ceil((1 + 7 * n + 3 * np * nq + nq * nq) * padding_rate);
  const long liwork = 1 + 8 * n + 2 * 4 * y_nnod;
  return {lwork, liwork};
}

inline long FS_byte_data_context(int n, int int_byte_size, int real_byte_size) {
  const auto worksize = FS_WorkSize(n, real_byte_size);
  return worksize.lwork * real_byte_size + worksize.liwork * int_byte_size;
}

inline MPI_Comm FS_get_comm_world() { return FS_info.get_comm_world(); }

inline int FS_get_myrank() { return FS_info.get_my_rank(); }

inline MPI_Group FS_get_group() { return FS_info.get_group(); }

inline bool is_FS_comm_member() { return FS_info.is_comm_member(); }

}  // namespace
}  // namespace FS_libs
