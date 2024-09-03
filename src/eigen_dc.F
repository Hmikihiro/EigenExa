!--------*---------*---------*---------*---------*---------*---------*-*
!     
! Module: eigen_dc_mod
!
! Purpose
! =======
!
! eigen_dc_mod manages the modules related for the DC methods.
!
!--------*---------*---------*---------*---------*---------*---------*-*

      module eigen_dc_mod
      use, intrinsic :: iso_c_binding
      implicit none

      real(c_double),bind(C) :: flops
      real(c_double),bind(C) :: dgemm_time
      real(c_double),bind(C) :: dgemm_dummy(1:2)
      real(c_double),bind(C) :: p_time0 = 0d0
      real(c_double),bind(C) :: p_time2 = 0d0
      real(c_double),bind(C) :: p_time3 = 0d0
      real(c_double),bind(C) :: p_times = 0d0
      real(c_double),bind(C) :: p_timez = 0d0
      real(c_double),bind(C) :: p_timer = 0d0


      end module eigen_dc_mod
