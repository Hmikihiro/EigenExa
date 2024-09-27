#pragma once

#include <mpi.h>

namespace eigen_libs_interface {
extern "C" void FS_eigen_init0(int comm, char order);
extern "C" void FS_eigen_free0();
}  // namespace eigen_libs_interface

namespace eigen_libs_FS_wrapper {
inline void eigen_free0() { eigen_libs_interface::FS_eigen_free0(); }

inline void eigen_init0(MPI_Comm comm, char order) {
  eigen_libs_interface::FS_eigen_init0(MPI_Comm_c2f(comm), order);
}
}  // namespace eigen_libs_FS_wrapper
