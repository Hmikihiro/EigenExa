#pragma once
#include <mpi.h>

namespace eigen_libs0_interface {

extern "C" {
void eigen_get_comm(int &eigen_comm, int &eigen_x_comm, int &eigen_y_comm);
void eigen_get_procs(int &procs, int &x_procs, int &y_procs);
void eigen_get_id(int &id, int &x_id, int &y_id);
char eigen_get_grid_major();
}

} // namespace eigen_libs0_interface

namespace eigen_libs0_wrapper {
struct eigen_comm {
  MPI_Comm eigen_comm;
  MPI_Comm eigen_x_comm;
  MPI_Comm eigen_y_comm;
};

struct eigen_procs {
  int procs;
  int x_procs;
  int y_procs;
};

struct eigen_id {
  int id;
  int x_id;
  int y_id;
};

inline eigen_procs eigen_get_procs() {
  eigen_procs procs = {};
  eigen_libs0_interface::eigen_get_procs(procs.procs, procs.x_procs,
                                         procs.y_procs);
  return procs;
}

inline eigen_comm eigen_get_comm() {
  int icomm, ix_comm, iy_comm;
  eigen_libs0_interface::eigen_get_comm(icomm, ix_comm, iy_comm);
  eigen_comm comm = {};
  comm.eigen_comm = MPI_Comm_f2c(icomm);
  comm.eigen_x_comm = MPI_Comm_f2c(ix_comm);
  comm.eigen_y_comm = MPI_Comm_f2c(iy_comm);
  return comm;
}

inline eigen_id eigen_get_id() {
  eigen_id id = {};
  eigen_libs0_interface::eigen_get_id(id.id, id.x_id, id.y_id);
  return id;
}

inline char eigen_get_grid_major() {
  return eigen_libs0_interface::eigen_get_grid_major();
}

template <class Integer>
Integer eigen_translate_g2l(Integer ictr, Integer nnod) {
  return ictr / nnod;
}
template <class Integer> Integer eigen_owner_node(Integer ictr, Integer nnod) {
  return ictr % nnod;
}

} // namespace eigen_libs0_wrapper

namespace eigen_libs0 {
static int eigen_NB_f = 48;
static int eigen_NB_b = 128;
} // namespace eigen_libs0
