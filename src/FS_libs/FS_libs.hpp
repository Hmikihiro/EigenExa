#pragma once

#include <mpi.h>

#include "../eigen_libs0.hpp"
#include "eigen_libs_FS_wrapper.hpp"
#include <cmath>

namespace FS_libs {

using eigen_libs0_wrapper::eigen_get_comm;
using eigen_libs0_wrapper::eigen_get_id;
using eigen_libs0_wrapper::eigen_get_procs;

extern MPI_Comm FS_COMM_WORLD;
extern int FS_MYRANK;
extern bool FS_COMM_MEMBER;
extern MPI_Group FS_GROUP;

class process_grid {
public:
  int nnod, x_nnod, y_nnod;
  int inod, x_inod, y_inod;
};
extern process_grid FS_node;

extern char FS_GRID_major;

class process_info {
private:
  MPI_Comm FS_COMM_WORLD;
  int FS_MYRANK;
  bool FS_COMM_MEMBER;
  MPI_Group FS_GROUP;
  process_grid FS_node;
  char FS_GRID_major;

  inline process_grid FS_init_cartesian(char GRID_major, int nnod, int inod) {
    auto x_nnod = int(sqrt(double(nnod)));
    int i = 1;
    const auto k = (nnod % i == 0) ? i : 1;

    while (true) {
      if (x_nnod <= k) {
        break;
      }
      if (x_nnod % k == 0 && nnod % x_nnod == 0) {
        break;
      }
      x_nnod -= 1;
    }

    const auto y_nnod = nnod / x_nnod;

    int x_inod, y_inod;
    if (GRID_major == 'R') {
      // row-major
      x_inod = (inod - 1) / y_nnod + 1;
      y_inod = (inod - 1) % y_nnod + 1;
    } else {
      // column-major
      x_inod = (inod - 1) % x_nnod + 1;
      y_inod = (inod - 1) / x_nnod + 1;
    }
    return process_grid{.nnod = nnod,
                        .x_nnod = x_nnod,
                        .y_nnod = y_nnod,
                        .inod = inod,
                        .x_inod = x_inod,
                        .y_inod = y_inod};
  }

public:
  inline MPI_Comm get_comm_world() const { return FS_COMM_WORLD; }
  inline int get_my_rank() const { return FS_MYRANK; }
  inline bool is_comm_member() const { return FS_COMM_MEMBER; }
  inline MPI_Group get_group() const { return FS_GROUP; }
  inline process_grid get_node() const { return FS_node; }
  inline char get_grid_major() const { return FS_GRID_major; }

  inline void comm_free() { MPI_Comm_free(&FS_COMM_WORLD); }

  process_info() = default;

  process_info(char order) {
    FS_GRID_major = (order == 'R') ? 'R' : 'C';
    const auto eigen_comm = eigen_get_comm().eigen_comm;

    // FS_COMM_WORLDの設定
    auto nnod = eigen_get_procs().procs;
    const auto inod = eigen_get_id().id;

    const auto p = static_cast<int>(std::log2(nnod));
    FS_COMM_MEMBER = (inod <= std::pow(2, p));
    const int color = FS_COMM_MEMBER ? 0 : 1;

    MPI_Comm_split(eigen_comm, color, inod, &FS_COMM_WORLD);

    if (FS_COMM_MEMBER) {
      MPI_Comm_rank(FS_COMM_WORLD, &FS_MYRANK);
      MPI_Comm_group(FS_COMM_WORLD, &FS_GROUP);

      MPI_Comm_size(FS_COMM_WORLD, &nnod);
      FS_node = FS_init_cartesian(FS_GRID_major, nnod, FS_MYRANK + 1);
    } else {
      FS_MYRANK = -1;
      FS_GROUP = MPI_GROUP_NULL;
      FS_node = process_grid{
          .nnod = -1,
          .x_nnod = -1,
          .y_nnod = -1,
          .inod = -1,
          .x_inod = -1,
          .y_inod = -1,
      };
    }
  }
};

extern process_info FS_info;

namespace {

inline void FS_init(MPI_Comm comm = MPI_COMM_WORLD, char order = 'C') {
  auto comm0 = comm;
  order = (order == 'R' || order == 'r') ? 'R' : 'C';
  eigen_libs_FS_wrapper::eigen_init0(comm0, order);
  FS_info = process_info(order);
}

inline void FS_free() {
  eigen_libs_FS_wrapper::eigen_free0();
  FS_info.comm_free();
}

struct Nod {
  int nod;
  int x;
  int y;
};

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

inline char FS_get_grid_major() { return FS_info.get_grid_major(); }

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
  const auto default_size = 16; // 4 + 4 + 8
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

inline int FS_get_comm_world() { return FS_info.get_comm_world(); }

inline int FS_get_myrank() { return FS_info.get_my_rank(); }

inline MPI_Group FS_get_group() { return FS_info.get_group(); }

inline bool is_FS_comm_member() { return FS_info.is_comm_member(); }

} // namespace
} // namespace FS_libs
