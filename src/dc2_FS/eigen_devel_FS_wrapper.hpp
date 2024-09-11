#pragma once

#include "../fortran_c_glue_int.hpp"
namespace eigen_devel_FS_wrapper {

extern "C" void FS_eigen_abort();

extern "C" void FS_eigen_timer_reset(glue_int bcast, glue_int reduce,
                                     glue_int redist, glue_int gather);

extern "C" double FS_eigen_timer_print();

} // namespace eigen_devel_FS_wrapper
