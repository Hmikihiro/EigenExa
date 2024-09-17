#include <memory>
#include <mpi.h>

#include "dc2_FS.hpp"

extern "C" {
void dc2_FS_f64(int n, int nvec, double d[], double e[], double z[], int ldz,
                long *info, double *ret) {
  const auto result = dc2_FS<int64_t, double>(n, nvec, d, e, z, ldz);
  *ret = result.ret;
  *info = result.info;
}

void dc2_FS_fp32(int n, int nvec, double d[], double e[], double z[], int ldz,
                 int ny, long *info, double *ret) {
  std::unique_ptr<float[]> d_fp32(new float[n]);
  std::unique_ptr<float[]> e_fp32(new float[n - 1]);
  std::unique_ptr<float[]> z_fp32(new float[ldz * ny]);
#pragma omp parallel
  {
#pragma omp for
    for (size_t i = 0; i < (size_t)ldz * ny; i++) {
      z_fp32[i] = z[i];
    }
#pragma omp for
    for (size_t i = 0; i < (size_t)n; i++) {
      d_fp32[i] = d[i];
    }
#pragma omp for
    for (size_t i = 0; i < (size_t)n - 1; i++) {
      e_fp32[i] = e[i];
    }
  }
  const auto result = dc2_FS<int64_t, float>(n, nvec, d_fp32.get(),
                                             e_fp32.get(), z_fp32.get(), ldz);
  *ret = result.ret;
  *info = result.info;
#pragma omp parallel
  {
#pragma omp for
    for (size_t i = 0; i < (size_t)ldz * ny; i++) {
      z[i] = z_fp32[i];
    }
#pragma omp for
    for (size_t i = 0; i < (size_t)n; i++) {
      d[i] = d_fp32[i];
    }
#pragma omp for
    for (size_t i = 0; i < (size_t)n - 1; i++) {
      e[i] = e_fp32[i];
    }
  }
}
}