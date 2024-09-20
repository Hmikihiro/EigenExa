#pragma once

#include <limits>

#if (defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER))

#include <mkl.h>

typedef MKL_INT eigen_mathlib_int;

#else
#include <cblas.h>
#include <lapacke.h>

#if defined(CBLAS_INT)             // netlib-blas
typedef CBLAS_INT eigen_mathlib_int;
#elif defined(OPENBLAS_CONST)      // openblas
typedef blasint eigen_mathlib_int;
#elif defined(FJ_MATHLIB_TYPE_INT) // fujitsu
typedef FJ_MATHLIB_TYPE_INT eigen_mathlib_int;
extern "C" {
double dlanst_(char *norn, eigen_mathlib_int *n, const double *D,
               const double *E);
float slanst_(char *norn, eigen_mathlib_int *n, const float *D, const float *E);
}
#endif

extern "C" {
double dlaed4_(eigen_mathlib_int *n, eigen_mathlib_int *i, const double d[],
               const double z[], double delta[], double *rho, double *dlam,
               eigen_mathlib_int *info);
float slaed4_(eigen_mathlib_int *n, eigen_mathlib_int *i, const float d[],
              const float z[], float delta[], float *rho, float *dlam,
              eigen_mathlib_int *info);

double dlamc3_(double *x, double *y);
float slamc3_(float *x, float *y);
}

#endif

namespace lapacke {
const auto FS_layout = CblasColMajor;

template <class Real>
inline eigen_mathlib_int
stedc(char compz, eigen_mathlib_int n, Real *d, Real *e, Real *z,
      eigen_mathlib_int ldz, Real *work, int64_t lwork,
      eigen_mathlib_int *iwork, eigen_mathlib_int liwork);

template <>
inline eigen_mathlib_int
stedc<double>(char compz, eigen_mathlib_int n, double *d, double *e, double *z,
              eigen_mathlib_int ldz, double *work, int64_t lwork,
              eigen_mathlib_int *iwork, eigen_mathlib_int liwork) {
  if (lwork > std::numeric_limits<eigen_mathlib_int>::max()) {
    lwork = std::numeric_limits<eigen_mathlib_int>::max();
  }

  return LAPACKE_dstedc_work(FS_layout, compz, n, d, e, z, ldz, work, lwork,
                             iwork, liwork);
}

template <>
inline eigen_mathlib_int
stedc<float>(char compz, eigen_mathlib_int n, float *d, float *e, float *z,
             eigen_mathlib_int ldz, float *work, int64_t lwork,
             eigen_mathlib_int *iwork, eigen_mathlib_int liwork) {
  if (lwork > std::numeric_limits<eigen_mathlib_int>::max()) {
    lwork = std::numeric_limits<eigen_mathlib_int>::max();
  }
  return LAPACKE_sstedc_work(FS_layout, compz, n, d, e, z, ldz, work, lwork,
                             iwork, liwork);
}

template <class Real>
inline eigen_mathlib_int lascl(char type, eigen_mathlib_int kl,
                               eigen_mathlib_int ku, Real cfrom, Real cto,
                               eigen_mathlib_int m, eigen_mathlib_int n,
                               Real *a, eigen_mathlib_int lda);

template <>
inline eigen_mathlib_int
lascl<double>(char type, eigen_mathlib_int kl, eigen_mathlib_int ku,
              double cfrom, double cto, eigen_mathlib_int m,
              eigen_mathlib_int n, double *a, eigen_mathlib_int lda) {
  eigen_mathlib_int info = 0;
#if defined(LAPACK_dlascl)
  LAPACK_dlascl(&type, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);
#else
  dlascl_(&type, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);
#endif
  return info;
}

template <>
inline eigen_mathlib_int
lascl<float>(char type, eigen_mathlib_int kl, eigen_mathlib_int ku, float cfrom,
             float cto, eigen_mathlib_int m, eigen_mathlib_int n, float *a,
             eigen_mathlib_int lda) {
  eigen_mathlib_int info = 0;
#if defined(LAPACK_slascl)
  LAPACK_slascl(&type, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);
#else
  slascl_(&type, &kl, &ku, &cfrom, &cto, &m, &n, a, &lda, &info);
#endif
  return info;
}

template <class Real>
inline Real lanst(char norm, eigen_mathlib_int n, const Real *D, const Real *E);

template <>
inline double lanst<double>(char norm, eigen_mathlib_int n, const double *D,
                            const double *E) {
#if defined(LAPACK_dlanst)
  return LAPACK_dlanst(&norm, &n, D, E);
#else
  return dlanst_(&norm, &n, D, E);
#endif
}

template <>
inline float lanst<float>(char norm, eigen_mathlib_int n, const float *D,
                          const float *E) {
#if defined(LAPACK_slanst)
  return LAPACK_slanst(&norm, &n, D, E);
#else
  return slanst_(&norm, &n, D, E);
#endif
}

template <class Real> inline Real lapy2(Real x, Real y);

template <> inline double lapy2<double>(double x, double y) {
  return LAPACKE_dlapy2(x, y);
}

template <> inline float lapy2<float>(float x, float y) {
  return LAPACKE_slapy2(x, y);
}

template <class Real>
inline void gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const eigen_mathlib_int M, const eigen_mathlib_int N,
                 const eigen_mathlib_int K, const Real alpha, const Real *A,
                 const eigen_mathlib_int lda, const Real *B,
                 const eigen_mathlib_int ldb, const Real beta, Real *C,
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

template <class Real>
inline void copy(eigen_mathlib_int n, const Real *X, eigen_mathlib_int incX,
                 Real *Y, eigen_mathlib_int incY);

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

template <class Real>
inline eigen_mathlib_int iamax(eigen_mathlib_int n, Real dx[],
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

template <class Real>
inline void scal(eigen_mathlib_int n, Real alpha, Real x[],
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

template <class Real>
inline Real nrm2(eigen_mathlib_int n, Real X[], eigen_mathlib_int incX);

template <>
inline double nrm2(eigen_mathlib_int n, double X[], eigen_mathlib_int incX) {
  return cblas_dnrm2(n, X, incX);
}

template <>
inline float nrm2(eigen_mathlib_int n, float X[], eigen_mathlib_int incX) {
  return cblas_snrm2(n, X, incX);
}

template <class Real>
inline eigen_mathlib_int laed4(eigen_mathlib_int n, eigen_mathlib_int i,
                               const Real d[], const Real z[], Real delta[],
                               Real rho, Real &dlam);

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

template <class Real> inline Real lamc3(Real x, Real y);

template <> inline double lamc3(double x, double y) { return dlamc3_(&x, &y); }
template <> inline float lamc3(float x, float y) { return slamc3_(&x, &y); }

template <class Real>
inline void rot(eigen_mathlib_int N, Real *X, eigen_mathlib_int incX, Real *Y,
                eigen_mathlib_int incY, Real c, Real s);

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
