#pragma once
#include "fortran_c_glue_int.hpp"
#include <mpi.h>

namespace eigen_libs0_interface {

extern "C" {
void eigen_get_comm(glue_int &eigen_comm, glue_int &eigen_x_comm,
                    glue_int &eigen_y_comm);
void eigen_get_procs(glue_int &procs, glue_int &x_procs, glue_int &y_procs);
void eigen_get_id(glue_int &id, glue_int &x_id, glue_int &y_id);
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
  glue_int procs;
  glue_int x_procs;
  glue_int y_procs;
};

struct eigen_id {
  glue_int id;
  glue_int x_id;
  glue_int y_id;
};

inline eigen_procs eigen_get_procs() {
  eigen_procs procs = {};
  eigen_libs0_interface::eigen_get_procs(procs.procs, procs.x_procs,
                                         procs.y_procs);
  return procs;
}

inline eigen_comm eigen_get_comm() {
  glue_int icomm, ix_comm, iy_comm;
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

inline glue_int eigen_translate_g2l(glue_int ictr, glue_int nnod) {
  return ictr / nnod;
}
inline glue_int eigen_owner_node(glue_int ictr, glue_int nnod) {
  return ictr % nnod;
}

} // namespace eigen_libs0_wrapper

namespace eigen_libs0 {
static glue_int eigen_NB_f = 48;
static glue_int eigen_NB_b = 128;
} // namespace eigen_libs0