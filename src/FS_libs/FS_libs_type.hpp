#pragma once

#include <mpi.h>

#include <cmath>

#include "../eigen_libs0.hpp"

namespace FS_libs {
using eigen_libs0_wrapper::eigen_get_comm;
using eigen_libs0_wrapper::eigen_get_id;
using eigen_libs0_wrapper::eigen_get_procs;
enum class FS_GRID_MAJOR { ROW = 'R', COLUMN = 'C' };

// プロセス情報
struct Nod {
  int nod;
  int x;
  int y;
};

struct process_grid {
  int nnod, x_nnod, y_nnod;
  int inod, x_inod, y_inod;
};
class process_info {
 private:
  MPI_Comm FS_COMM_WORLD;
  int FS_MYRANK;
  bool FS_COMM_MEMBER;
  MPI_Group FS_GROUP;
  process_grid FS_node;
  FS_GRID_MAJOR FS_GRID_major;

  inline process_grid FS_init_cartesian(FS_GRID_MAJOR GRID_major, int nnod,
                                        int inod) {
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
    if (GRID_major == FS_GRID_MAJOR::ROW) {
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
  inline const MPI_Comm &get_comm_world() const { return FS_COMM_WORLD; }
  inline int get_my_rank() const { return FS_MYRANK; }
  inline bool is_comm_member() const { return FS_COMM_MEMBER; }
  inline MPI_Group get_group() const { return FS_GROUP; }
  inline process_grid get_node() const { return FS_node; }
  inline FS_GRID_MAJOR get_grid_major() const { return FS_GRID_major; }

  // プロセス情報の解放
  inline void comm_free() { MPI_Comm_free(&FS_COMM_WORLD); }

  process_info() = default;

  process_info(FS_GRID_MAJOR order) {
    FS_GRID_major = order;
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
}  // namespace FS_libs
