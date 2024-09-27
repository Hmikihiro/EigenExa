#include <mpi.h>

#include "FS_libs.hpp"

namespace FS_libs {
process_info FS_info;
}  // namespace FS_libs

namespace FS_libs_interface {
extern "C" void FS_init(int comm, char order) {
  const auto FS_order = (order == 'R') ? FS_libs::FS_GRID_MAJOR::ROW
                                       : FS_libs::FS_GRID_MAJOR::COLUMN;
  FS_libs::FS_init(MPI_Comm_f2c(comm), FS_order);
}
extern "C" void FS_free() { FS_libs::FS_free(); }

extern "C" void FS_get_matdims(int n, int &nx, int &ny) {
  const auto dims = FS_libs::FS_get_matdims(n);
  nx = dims.nx;
  ny = dims.ny;
}
extern "C" int FS_get_myrank() { return FS_libs::FS_info.get_my_rank(); }

extern "C" long FS_byte_data_context(int n, int int_size, int real_size) {
  return FS_libs::FS_byte_data_context(n, int_size, real_size);
}
}  // namespace FS_libs_interface
