#pragma once
#include <mpi.h>

namespace MPI_Datatype_wrapper {
template <typename T>
static const MPI_Datatype MPI_TYPE = 0;
template <>
static const MPI_Datatype MPI_TYPE<int> = MPI_INT;
template <>
static const MPI_Datatype MPI_TYPE<long> = MPI_LONG;
template <>
static const MPI_Datatype MPI_TYPE<long long> = MPI_LONG_LONG;
template <>
static const MPI_Datatype MPI_TYPE<double> = MPI_DOUBLE;
template <>
static const MPI_Datatype MPI_TYPE<float> = MPI_FLOAT;
}  // namespace MPI_Datatype_wrapper
