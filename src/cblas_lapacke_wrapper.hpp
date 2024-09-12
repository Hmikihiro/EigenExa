#pragma once

#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)

#include <mkl.h>

typedef MKL_INT eigen_mathlib_int;

#else
#include <cblas.h>
#include <lapacke.h>

typedef FJ_MATHLIB_INT eigen_mathlib_int;

extern "C" {
double dlaed4_(eigen_mathlib_int *n, eigen_mathlib_int *i, const double d[],
               const double z[], double delta[], double *rho, double *dlam,
               eigen_mathlib_int *info);
float slaed4_(eigen_mathlib_int *n, eigen_mathlib_int *i, const float d[],
              const float z[], float delta[], float *rho, float *dlam,
              eigen_mathlib_int *info);

double dlanst_(char *norn, eigen_mathlib_int *n, const double *D,
               const double *E);
float slanst_(char *norn, eigen_mathlib_int *n, const float *D, const float *E);

double dlamc3_(double *x, double *y);
float slamc3_(float *x, float *y);

void dlascl_(const char *type, const eigen_mathlib_int *kl,
             const eigen_mathlib_int *ku, const double *cfrom,
             const double *cto, const eigen_mathlib_int *m,
             const eigen_mathlib_int *n, double *a,
             const eigen_mathlib_int *lda, eigen_mathlib_int *info);

void slascl_(const char *type, const eigen_mathlib_int *kl,
             const eigen_mathlib_int *ku, const float *cfrom, const float *cto,
             const eigen_mathlib_int *m, const eigen_mathlib_int *n, float *a,
             const eigen_mathlib_int *lda, eigen_mathlib_int *info);
}

#endif

namespace lapacke {
const CBLAS_LAYOUT FS_layout = CBLAS_LAYOUT::CblasColMajor;
#define FS_LAPACKE_LAYOUT LAPACK_COL_MAJOR

template <class Float>
inline eigen_mathlib_int
stedc(char compz, eigen_mathlib_int n, Float *d, Float *e, Float *z,
      eigen_mathlib_int ldz, Float *work, eigen_mathlib_int lwork,
      eigen_mathlib_int *iwork, eigen_mathlib_int liwork);

template <>
inline eigen_mathlib_int
stedc<double>(char compz, eigen_mathlib_int n, double *d, double *e, double *z,
              eigen_mathlib_int ldz, double *work, eigen_mathlib_int lwork,
              eigen_mathlib_int *iwork, eigen_mathlib_int liwork) {
  return LAPACKE_dstedc_work(FS_LAPACKE_LAYOUT, compz, n, d, e, z, ldz, work,
                             lwork, iwork, liwork);
}

template <>
inline eigen_mathlib_int
stedc<float>(char compz, eigen_mathlib_int n, float *d, float *e, float *z,
             eigen_mathlib_int ldz, float *work, eigen_mathlib_int lwork,
             eigen_mathlib_int *iwork, eigen_mathlib_int liwork) {
  return LAPACKE_sstedc_work(FS_LAPACKE_LAYOUT, compz, n, d, e, z, ldz, work,
                             lwork, iwork, liwork);
}

template <class Float>
inline eigen_mathlib_int lascl(char type, eigen_mathlib_int kl,
                               eigen_mathlib_int ku, Float cfrom, Float cto,
                               eigen_mathlib_int m, eigen_mathlib_int n,
                               Float *a, eigen_mathlib_int lda);

template <>
inline eigen_mathlib_int
lascl<double>(char type, eigen_mathlib_int kl, eigen_mathlib_int ku,
              double cfrom, double cto, eigen_mathlib_int m,
              eigen_mathlib_int n, double *a, eigen_mathlib_int lda) {
  eigen_mathlib_int info = 0;
  dlascl_(&type, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);
  return info;
}

template <>
inline eigen_mathlib_int
lascl<float>(char type, eigen_mathlib_int kl, eigen_mathlib_int ku, float cfrom,
             float cto, eigen_mathlib_int m, eigen_mathlib_int n, float *a,
             eigen_mathlib_int lda) {
  eigen_mathlib_int info = 0;
  slascl_(&type, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);
  return info;
}

template <class Float>
inline Float lanst(char norm, eigen_mathlib_int n, const Float *D,
                   const Float *E);

template <>
inline double lanst<double>(char norm, eigen_mathlib_int n, const double *D,
                            const double *E) {
  return dlanst_(&norm, &n, D, E);
}

template <>
inline float lanst<float>(char norm, eigen_mathlib_int n, const float *D,
                          const float *E) {
  return slanst_(&norm, &n, D, E);
}

template <class Float> inline Float lapy2(Float x, Float y);

template <> inline double lapy2<double>(double x, double y) {
  return LAPACKE_dlapy2(x, y);
}

template <> inline float lapy2<float>(float x, float y) {
  return LAPACKE_slapy2(x, y);
}

template <class Float>
inline void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const eigen_mathlib_int M, const eigen_mathlib_int N,
                 const eigen_mathlib_int K, const Float alpha, const Float *A,
                 const eigen_mathlib_int lda, const Float *B,
                 const eigen_mathlib_int ldb, const Float beta, Float *C,
                 const eigen_mathlib_int ldc);

template <>
inline void
gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
             const eigen_mathlib_int M, const eigen_mathlib_int N,
             const eigen_mathlib_int K, const double alpha, const double *A,
             const eigen_mathlib_int lda, const double *B,
             const eigen_mathlib_int ldb, const double beta, double *C,
             const eigen_mathlib_int ldc) {
  cblas_dgemm(FS_layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta,
              C, ldc);
}

template <>
inline void gemm<float>(const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const eigen_mathlib_int M,
                        const eigen_mathlib_int N, const eigen_mathlib_int K,
                        const float alpha, const float *A,
                        const eigen_mathlib_int lda, const float *B,
                        const eigen_mathlib_int ldb, const float beta, float *C,
                        const eigen_mathlib_int ldc) {
  cblas_sgemm(FS_layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta,
              C, ldc);
}

template <class Float>
inline void copy(eigen_mathlib_int n, const Float *X, eigen_mathlib_int incX,
                 Float *Y, eigen_mathlib_int incY);

template <>
inline void copy<double>(eigen_mathlib_int n, const double *X,
                         eigen_mathlib_int incX, double *Y,
                         eigen_mathlib_int incY) {
  cblas_dcopy(n, X, incX, Y, incY);
}

template <>
inline void copy<float>(eigen_mathlib_int n, const float *X,
                        eigen_mathlib_int incX, float *Y,
                        eigen_mathlib_int incY) {
  cblas_scopy(n, X, incX, Y, incY);
}

template <class Float>
inline eigen_mathlib_int iamax(eigen_mathlib_int n, Float dx[],
                               eigen_mathlib_int incx);

template <>
inline eigen_mathlib_int iamax<double>(eigen_mathlib_int n, double dx[],
                                       eigen_mathlib_int incx) {
  return cblas_idamax(n, dx, incx);
}

template <>
inline eigen_mathlib_int iamax<float>(eigen_mathlib_int n, float dx[],
                                      eigen_mathlib_int incx) {
  return cblas_isamax(n, dx, incx);
}

template <class Float>
inline void scal(eigen_mathlib_int n, Float alpha, Float x[],
                 eigen_mathlib_int incx);

template <>
inline void scal(eigen_mathlib_int n, double alpha, double x[],
                 eigen_mathlib_int incx) {
  cblas_dscal(n, alpha, x, incx);
}

template <>
inline void scal(eigen_mathlib_int n, float alpha, float x[],
                 eigen_mathlib_int incx) {
  cblas_sscal(n, alpha, x, incx);
}

template <class Float>
inline Float nrm2(eigen_mathlib_int n, Float X[], eigen_mathlib_int incX);

template <>
inline double nrm2(eigen_mathlib_int n, double X[], eigen_mathlib_int incX) {
  return cblas_dnrm2(n, X, incX);
}

template <>
inline float nrm2(eigen_mathlib_int n, float X[], eigen_mathlib_int incX) {
  return cblas_snrm2(n, X, incX);
}

template <class Float>
inline eigen_mathlib_int laed4(eigen_mathlib_int n, eigen_mathlib_int i,
                               const Float d[], const Float z[], Float delta[],
                               Float rho, Float &dlam);

template <>
inline eigen_mathlib_int laed4(eigen_mathlib_int n, eigen_mathlib_int i,
                               const double d[], const double z[],
                               double delta[], double rho, double &dlam) {
  eigen_mathlib_int info;
  dlaed4_(&n, &i, d, z, delta, &rho, &dlam, &info);
  return info;
}

template <>
inline eigen_mathlib_int laed4(eigen_mathlib_int n, eigen_mathlib_int i,
                               const float d[], const float z[], float delta[],
                               float rho, float &dlam) {
  eigen_mathlib_int info;
  slaed4_(&n, &i, d, z, delta, &rho, &dlam, &info);
  return info;
}

template <class Float> inline Float lamc3(Float x, Float y);

template <> inline double lamc3(double x, double y) { return dlamc3_(&x, &y); }
template <> inline float lamc3(float x, float y) { return slamc3_(&x, &y); }

template <class Float>
inline void rot(eigen_mathlib_int N, Float *X, eigen_mathlib_int incX, Float *Y,
                eigen_mathlib_int incY, Float c, Float s);

template <>
inline void rot(eigen_mathlib_int N, double *X, eigen_mathlib_int incX,
                double *Y, eigen_mathlib_int incY, double c, double s) {
  cblas_drot(N, X, incX, Y, incY, c, s);
}

template <>
inline void rot(eigen_mathlib_int N, float *X, eigen_mathlib_int incX, float *Y,
                eigen_mathlib_int incY, float c, float s) {
  cblas_srot(N, X, incX, Y, incY, c, s);
}
} // namespace lapacke
