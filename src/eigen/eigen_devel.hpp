#pragma once
#ifndef EIGEN_DEVEL_HPP
#define EIGEN_DEVEL_HPP

namespace eigen_devel {

extern "C" void FS_eigen_abort();

extern "C" void FS_eigen_timer_reset(int bcast, int reduce, int redist,
                                     int gather);

extern "C" double FS_eigen_timer_print();

} // namespace eigen_devel

#endif
