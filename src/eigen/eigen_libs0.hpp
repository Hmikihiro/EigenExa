#pragma once
#ifndef EIGEN_LIBS_HPP
#define EIGEN_LIBS_HPP
#include <mpi.h>
namespace eigen_libs0 {
extern "C" {

namespace eigen_comm_int {
void eigen_init0(int comm, char order);
void eigen_get_comm(int &eigen_comm, int &eigen_x_comm, int &eigen_y_comm);
void eigen_get_procs(int &procs, int &x_procs, int &y_procs);
void eigen_get_id(int &id, int &x_id, int &y_id);
} // namespace eigen_comm_int
void eigen_free0();
char eigen_get_grid_major();
}

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
  eigen_comm_int::eigen_get_procs(procs.procs, procs.x_procs, procs.y_procs);
  return procs;
}

inline eigen_comm eigen_get_comm() {
  int icomm, ix_comm, iy_comm;
  eigen_comm_int::eigen_get_comm(icomm, ix_comm, iy_comm);
  eigen_comm comm = {};
  comm.eigen_comm = MPI_Comm_f2c(icomm);
  comm.eigen_x_comm = MPI_Comm_f2c(ix_comm);
  comm.eigen_y_comm = MPI_Comm_f2c(iy_comm);
  return comm;
}

inline eigen_id eigen_get_id() {
  eigen_id id = {};
  eigen_comm_int::eigen_get_id(id.id, id.x_id, id.y_id);
  return id;
}

inline void eigen_init0(MPI_Comm comm, char order) {
  eigen_comm_int::eigen_init0(MPI_Comm_c2f(comm), order);
}

inline int eigen_translate_g2l(int ictr, int nnod) { return ictr / nnod; }
inline int eigen_owner_node(int ictr, int nnod) { return ictr % nnod; }

} // namespace eigen_libs0
#endif
