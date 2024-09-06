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

namespace FS_libs_C_interface {
extern "C" void FS_init(int comm, char order) {
  FS_libs::FS_init(MPI_Comm_f2c(comm), order);
}
extern "C" void FS_free() { FS_libs::FS_free(); }

extern "C" void FS_get_matdims(int n, int &nx, int &ny) {
  FS_libs::FS_get_matdims(n, nx, ny);
}
extern "C" int FS_get_myrank() { return FS_libs::FS_MYRANK; }
} // namespace FS_libs_C_interface
