#include <mpi.h>

#include "dc2_FS.hpp"

extern "C" {
void dc2_FS_f64(int n, int nvec, double d[], double e[], double z[], int ldz,
                long *info, double *ret) {
  const auto result = dc2_FS<int64_t, double>(n, nvec, d, e, z, ldz);
  *ret = result.ret;
  *info = result.info;
}

void dc2_FS_fp32(int n, int nvec, float d[], float e[], float z[], int ldz,
                 long *info, float *ret) {
  const auto result = dc2_FS<int64_t, float>(n, nvec, d, e, z, ldz);
  *ret = result.ret;
  *info = result.info;
}
}