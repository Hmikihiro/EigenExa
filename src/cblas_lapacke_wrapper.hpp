#pragma once
#ifndef FS_BLAS_HPP
#define FS_BLAS_HPP

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)

#include <mkl.h>

typedef MKL_INT eigen_int;

typedef MKL_INT64 eigen_int64;

#else
#include <cblas.h>
#include <plasma.h>

#include <core_dblas.h>
#include <core_sblas.h>
#include <lapacke.h>

typedef int eigen_int;

extern "C" {
double dlaed4_(eigen_int *n, eigen_int *i, double d[], double z[],
               double delta[], double *rho, double *dlam, eigen_int *info);
float slaed4_(eigen_int *n, eigen_int *i, float d[], float z[], float delta[],
              float *rho, float *dlam, eigen_int *info);

double dlanst_(char *norn, eigen_int *n, const double *D, const double *E);
float slanst_(char *norn, eigen_int *n, const float *D, const float *E);

double dlamc3_(double *x, double *y);
float slamc3_(float *x, float *y);
}

#endif

namespace lapacke {
const CBLAS_LAYOUT FS_layout = CBLAS_LAYOUT::CblasColMajor;
#define FS_LAPACKE_LAYOUT LAPACK_COL_MAJOR

template <class Integer, class Float>
inline Integer stedc(char compz, Integer n, Float *d, Float *e, Float *z,
                     Integer ldz, Float *work, Integer lwork, Integer *iwork,
                     Integer liwork) {
  static_assert(false, "unexpected type set");
};

template <>
inline eigen_int stedc<eigen_int, double>(char compz, eigen_int n, double *d,
                                          double *e, double *z, eigen_int ldz,
                                          double *work, eigen_int lwork,
                                          eigen_int *iwork, eigen_int liwork) {
  return LAPACKE_dstedc_work(FS_LAPACKE_LAYOUT, compz, n, d, e, z, ldz, work,
                             lwork, iwork, liwork);
}

template <>
inline eigen_int stedc<eigen_int, float>(char compz, eigen_int n, float *d,
                                         float *e, float *z, eigen_int ldz,
                                         float *work, eigen_int lwork,
                                         eigen_int *iwork, eigen_int liwork) {
  return LAPACKE_sstedc_work(FS_LAPACKE_LAYOUT, compz, n, d, e, z, ldz, work,
                             lwork, iwork, liwork);
}

template <>
inline eigen_int64 stedc<eigen_int64, double>(
    char compz, eigen_int64 n, double *d, double *e, double *z, eigen_int64 ldz,
    double *work, eigen_int64 lwork, eigen_int64 *iwork, eigen_int64 liwork) {
  return LAPACKE_dstedc_work_64(FS_LAPACKE_LAYOUT, compz, n, d, e, z, ldz, work,
                                lwork, iwork, liwork);
}

template <>
inline eigen_int64 stedc<eigen_int64, float>(
    char compz, eigen_int64 n, float *d, float *e, float *z, eigen_int64 ldz,
    float *work, eigen_int64 lwork, eigen_int64 *iwork, eigen_int64 liwork) {
  return LAPACKE_sstedc_work_64(FS_LAPACKE_LAYOUT, compz, n, d, e, z, ldz, work,
                                lwork, iwork, liwork);
}

template <class Integer, class Float>
inline Integer lascl(char type, Integer kl, Integer ku, Float cfrom, Float cto,
                     Integer m, Integer n, Float *a, Integer lda) {
  static_assert(false, "lascl unexpected type set");
};

template <>
inline eigen_int lascl<eigen_int, double>(char type, eigen_int kl, eigen_int ku,
                                          double cfrom, double cto, eigen_int m,
                                          eigen_int n, double *a,
                                          eigen_int lda) {
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
  return LAPACKE_dlascl_work(FS_LAPACKE_LAYOUT, type, kl, ku, cfrom, cto, m, n,
                             a, lda);
#else
  if (type == 'G') {
    return CORE_dlascl(PlasmaGeneral, kl, ku, cfrom, cto, m, n, a, lda);
  }
  return -1;
#endif
}

template <>
inline eigen_int lascl<eigen_int, float>(char type, eigen_int kl, eigen_int ku,
                                         float cfrom, float cto, eigen_int m,
                                         eigen_int n, float *a, eigen_int lda) {
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
  return LAPACKE_slascl_work(FS_LAPACKE_LAYOUT, type, kl, ku, cfrom, cto, m, n,
                             a, lda);
#else
  if (type == 'G') {
    return CORE_slascl(PlasmaGeneral, kl, ku, cfrom, cto, m, n, a, lda);
  }
  return -1;
#endif
}

template <>
inline eigen_int64
lascl<eigen_int64, double>(char type, eigen_int64 kl, eigen_int64 ku,
                           double cfrom, double cto, eigen_int64 m,
                           eigen_int64 n, double *a, eigen_int64 lda) {
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
  return LAPACKE_dlascl_work_64(FS_LAPACKE_LAYOUT, type, kl, ku, cfrom, cto, m,
                                n, a, lda);
#else
  return -1;
#endif
}

template <>
inline eigen_int64
lascl<eigen_int64, float>(char type, eigen_int64 kl, eigen_int64 ku,
                          float cfrom, float cto, eigen_int64 m, eigen_int64 n,
                          float *a, eigen_int64 lda) {
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
  return LAPACKE_slascl_work_64(FS_LAPACKE_LAYOUT, type, kl, ku, cfrom, cto, m,
                                n, a, lda);
#else
  return -1;
#endif
}

template <class Integer, class Float>
inline Float lanst(char norm, Integer n, const Float *D, const Float *E) {
  static_assert(false, "lanst unexpected type set");
};

template <>
inline double lanst<eigen_int, double>(char norm, eigen_int n, const double *D,
                                       const double *E) {
  return dlanst_(&norm, &n, D, E);
}

template <>
inline float lanst<eigen_int, float>(char norm, eigen_int n, const float *D,
                                     const float *E) {
  return slanst_(&norm, &n, D, E);
}

template <>
inline double lanst<eigen_int64, double>(char norm, eigen_int64 n,
                                         const double *D, const double *E) {
  return dlanst_64(&norm, &n, D, E);
}

template <>
inline float lanst<eigen_int64, float>(char norm, eigen_int64 n, const float *D,
                                       const float *E) {
  return slanst_64(&norm, &n, D, E);
}

template <class Float> inline Float lapy2(Float x, Float y) {
  static_assert(false, "lapy2 unexpected type set");
};

template <> inline double lapy2<double>(double x, double y) {
  return LAPACKE_dlapy2(x, y);
}

template <> inline float lapy2<float>(float x, float y) {
  return LAPACKE_slapy2(x, y);
}

template <class Integer, class Float>
inline void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const Integer M, const Integer N, const Integer K,
                 const Float alpha, const Float *A, const Integer lda,
                 const Float *B, const Integer ldb, const Float beta, Float *C,
                 const Integer ldc) {
  static_assert(false, "gemm unexpected type set");
};

template <>
inline void gemm<eigen_int, double>(
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const eigen_int M, const eigen_int N, const eigen_int K, const double alpha,
    const double *A, const eigen_int lda, const double *B, const eigen_int ldb,
    const double beta, double *C, const eigen_int ldc) {
  cblas_dgemm(FS_layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta,
              C, ldc);
}

template <>
inline void gemm<eigen_int, float>(
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const eigen_int M, const eigen_int N, const eigen_int K, const float alpha,
    const float *A, const eigen_int lda, const float *B, const eigen_int ldb,
    const float beta, float *C, const eigen_int ldc) {
  cblas_sgemm(FS_layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta,
              C, ldc);
}

template <>
inline void gemm<eigen_int64, double>(const CBLAS_TRANSPOSE TransA,
                                      const CBLAS_TRANSPOSE TransB,
                                      const eigen_int64 M, const eigen_int64 N,
                                      const eigen_int64 K, const double alpha,
                                      const double *A, const eigen_int64 lda,
                                      const double *B, const eigen_int64 ldb,
                                      const double beta, double *C,
                                      const eigen_int64 ldc) {
  cblas_dgemm_64(FS_layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                 beta, C, ldc);
}

template <>
inline void gemm<eigen_int64, float>(
    const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
    const eigen_int64 M, const eigen_int64 N, const eigen_int64 K,
    const float alpha, const float *A, const eigen_int64 lda, const float *B,
    const eigen_int64 ldb, const float beta, float *C, const eigen_int64 ldc) {
  cblas_sgemm_64(FS_layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                 beta, C, ldc);
}

template <class Integer, class Float>
inline void copy(Integer n, const Float *X, Integer incX, Float *Y,
                 Integer incY) {

  static_assert(false, "copy unexpected type set");
};

template <>
inline void copy<eigen_int, double>(eigen_int n, const double *X,
                                    eigen_int incX, double *Y, eigen_int incY) {
  cblas_dcopy(n, X, incX, Y, incY);
}

template <>
inline void copy<eigen_int, float>(eigen_int n, const float *X, eigen_int incX,
                                   float *Y, eigen_int incY) {
  cblas_scopy(n, X, incX, Y, incY);
}

template <>
inline void copy<eigen_int64, double>(eigen_int64 n, const double *X,
                                      eigen_int64 incX, double *Y,
                                      eigen_int64 incY) {
  cblas_dcopy_64(n, X, incX, Y, incY);
}

template <>
inline void copy<eigen_int64, float>(eigen_int64 n, const float *X,
                                     eigen_int64 incX, float *Y,
                                     eigen_int64 incY) {
  cblas_scopy_64(n, X, incX, Y, incY);
}

template <class Integer, class Float>
inline Integer iamax(Integer n, Float dx[], Integer incx) {
  static_assert(false, "i_amax unexpected type set");
};

template <>
inline eigen_int iamax<eigen_int, double>(eigen_int n, double dx[],
                                          eigen_int incx) {
  return cblas_idamax(n, dx, incx);
}

template <>
inline eigen_int iamax<eigen_int, float>(eigen_int n, float dx[],
                                         eigen_int incx) {
  return cblas_isamax(n, dx, incx);
}

template <>
inline eigen_int64 iamax<eigen_int64, double>(eigen_int64 n, double dx[],
                                              eigen_int64 incx) {
  return cblas_idamax_64(n, dx, incx);
}

template <>
inline eigen_int64 iamax<eigen_int64, float>(eigen_int64 n, float dx[],
                                             eigen_int64 incx) {
  return cblas_isamax_64(n, dx, incx);
}

template <class Integer, class Float>
inline void scal(Integer n, Float alpha, Float x[], Integer incx) {
  static_assert(false, "scal unexpected type set");
};

template <>
inline void scal(eigen_int n, double alpha, double x[], eigen_int incx) {
  cblas_dscal(n, alpha, x, incx);
}

template <>
inline void scal(eigen_int n, float alpha, float x[], eigen_int incx) {
  cblas_sscal(n, alpha, x, incx);
}

template <>
inline void scal(eigen_int64 n, double alpha, double x[], eigen_int64 incx) {
  cblas_dscal_64(n, alpha, x, incx);
}

template <>
inline void scal(eigen_int64 n, float alpha, float x[], eigen_int64 incx) {
  cblas_sscal_64(n, alpha, x, incx);
}

template <class Integer, class Float>
inline Float nrm2(Integer n, Float X[], Integer incX) {
  static_assert(false, "nrm2 unexpected type set");
};

template <> inline double nrm2(eigen_int n, double X[], eigen_int incX) {
  return cblas_dnrm2(n, X, incX);
}

template <> inline float nrm2(eigen_int n, float X[], eigen_int incX) {
  return cblas_snrm2(n, X, incX);
}

template <> inline double nrm2(eigen_int64 n, double X[], eigen_int64 incX) {
  return cblas_dnrm2_64(n, X, incX);
}

template <> inline float nrm2(eigen_int64 n, float X[], eigen_int64 incX) {
  return cblas_snrm2_64(n, X, incX);
}

template <class Integer, class Float>
inline Integer laed4(Integer n, Integer i, Float d[], Float z[], Float delta[],
                     Float rho, Float &dlam) {
  static_assert(false, "laed4 unexpected type set");
};

template <>
inline eigen_int laed4(eigen_int n, eigen_int i, double d[], double z[],
                       double delta[], double rho, double &dlam) {
  eigen_int info;
  dlaed4_(&n, &i, d, z, delta, &rho, &dlam, &info);
  return info;
}

template <>
inline eigen_int laed4(eigen_int n, eigen_int i, float d[], float z[],
                       float delta[], float rho, float &dlam) {
  eigen_int info;
  slaed4_(&n, &i, d, z, delta, &rho, &dlam, &info);
  return info;
}

template <>
inline eigen_int64 laed4(eigen_int64 n, eigen_int64 i, double d[], double z[],
                         double delta[], double rho, double &dlam) {
  eigen_int64 info;
  dlaed4_64(&n, &i, d, z, delta, &rho, &dlam, &info);
  return info;
}

template <>
inline eigen_int64 laed4(eigen_int64 n, eigen_int64 i, float d[], float z[],
                         float delta[], float rho, float &dlam) {
  eigen_int64 info;
  slaed4_64(&n, &i, d, z, delta, &rho, &dlam, &info);
  return info;
}

template <class Float> inline Float lamc3(Float x, Float y) {
  static_assert(false, "unexpected type set");
};

template <> inline double lamc3(double x, double y) { return dlamc3_(&x, &y); }
template <> inline float lamc3(float x, float y) { return slamc3_(&x, &y); }

template <class Integer, class Float>
inline void rot(Integer N, Float *X, Integer incX, Float *Y, Integer incY,
                Float c, Float s) {
  static_assert(false, "rot unexpected type set");
};

template <>
inline void rot(eigen_int N, double *X, eigen_int incX, double *Y,
                eigen_int incY, double c, double s) {
  cblas_drot(N, X, incX, Y, incY, c, s);
}

template <>
inline void rot(eigen_int N, float *X, eigen_int incX, float *Y, eigen_int incY,
                float c, float s) {
  cblas_srot(N, X, incX, Y, incY, c, s);
}

template <>
inline void rot(eigen_int64 N, double *X, eigen_int64 incX, double *Y,
                eigen_int64 incY, double c, double s) {
  cblas_drot_64(N, X, incX, Y, incY, c, s);
}

template <>
inline void rot(eigen_int64 N, float *X, eigen_int64 incX, float *Y,
                eigen_int64 incY, float c, float s) {
  cblas_srot_64(N, X, incX, Y, incY, c, s);
}
} // namespace lapacke
#endif
