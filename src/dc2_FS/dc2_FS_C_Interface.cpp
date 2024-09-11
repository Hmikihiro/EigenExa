#include <mpi.h>

#define _DEBUGLOG

#include "dc2_FS.hpp"

extern "C" {
void dc2_FS_f64(int n, int nvec, double d[], double e[], double z[], int ldz,
                long *info, double *ret) {
  *info = dc2_FS<long, double>(n, nvec, d, e, z, ldz, ret);
}

void dc2_FS_f32(int n, int nvec, float d[], float e[], float z[], int ldz,
                long *info, float *ret) {
  *info = dc2_FS<long, float>(n, nvec, d, e, z, ldz, ret);
}
}