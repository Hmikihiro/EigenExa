#include <mpi.h>

#define _DEBUGLOG

#include "dc2_FS.hpp"
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#define int_type eigen_int64
#else
#define int_type eigen_int
#endif

extern "C" {
void dc2_FS_f64(int n, int nvec, double d[], double e[], double z[], int ldz,
                long *info, double *ret) {
  dc2_FS<int_type, double>(n, nvec, d, e, z, ldz, info, ret);
}

void dc2_FS_f32(int n, int nvec, float d[], float e[], float z[], int ldz,
                long *info, float *ret) {
  dc2_FS<int_type, float>(n, nvec, d, e, z, ldz, info, ret);
}
}