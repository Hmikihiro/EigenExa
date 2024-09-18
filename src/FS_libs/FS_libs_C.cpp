#include <mpi.h>

#include "FS_libs.hpp"

namespace FS_libs {
MPI_Comm FS_COMM_WORLD = MPI_COMM_WORLD;
int FS_MYRANK = 0;
bool FS_COMM_MEMBER = false;
MPI_Group FS_GROUP = MPI_GROUP_NULL;

process_grid FS_node = {};
char FS_GRID_major = 'C';
} // namespace FS_libs

namespace FS_libs_interface {
extern "C" void FS_init(int comm, char order) {
  FS_libs::FS_init(MPI_Comm_f2c(comm), order);
}
extern "C" void FS_free() { FS_libs::FS_free(); }

extern "C" void FS_get_matdims(int n, int &nx, int &ny) {
  const auto dims = FS_libs::FS_get_matdims(n);
  nx = dims.nx;
  ny = dims.ny;
}
extern "C" int FS_get_myrank() { return FS_libs::FS_MYRANK; }

extern "C" long FS_byte_data_context(int n, int int_size, int real_size) {
  return FS_libs::FS_byte_data_context(n, int_size, real_size);
}
} // namespace FS_libs_interface
