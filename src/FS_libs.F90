!>
!> @file   FS_libs.F90
!> @brief  module FS_libs_mod
!!>
!
!
!
!> module FS_libs_mod
!> @brief  @n
!> Purpose @n
!> ======= @n
!> FS_libs_mod include common variables and subroutines
!>
module FS_libs_mod
  use, intrinsic :: iso_c_binding
  use mpi
  implicit none

  public :: FS_init
  public :: FS_free
  public :: FS_get_matdims
  public ::FS_get_myrank

  ! comm world
  !integer :: FS_COMM_WORLD = MPI_COMM_WORLD  !< comm world
  !integer :: FS_MYRANK = 0                   !< rank No.
  !logical :: FS_COMM_MEMBER = .FALSE.        !< comm member fla
  
  !integer :: FS_GROUP = MPI_UNDEFINED !< FS_COMM_WORLD group

  type  process_grid 
     integer :: nnod,x_nnod,y_nnod
     integer :: inod,x_inod,y_inod
  end type process_grid
  type(process_grid) :: FS_node

  !> type version_t
  !> @brief version info
  type, public :: version_t
     integer       :: Major_Version !< Major version
     integer       :: Minor_Version !< Minor version
     integer       :: Patch_Level   !< Patchlevel 0=none,1=a,2=b,...
     character(32) :: date          !< Release date
     character(32) :: vcode         !< Version code name
  end type  version_t

  !> version info
  type(version_t) :: FS_Version & ! 1.0
       = version_t (            &
       1, 1, 0,                 & ! Major, Minor, Patchlevel
       'Mar 31, 2019',          & ! Release date
       'FS proto'               & ! Version code
       )

  !> grid major
  !character(1) :: FS_GRID_major = 'C'

#ifdef WRITE_INPUT_VEC
  !> matrix type
  integer :: mat_type
#endif

  interface

       subroutine eigen_FS(n, nvec, a, lda, w, z, ldz, &
          m_forward, m_backward, mode, precision)
       integer,   intent(in)           :: n
       integer,   intent(in)           :: nvec
       real(8),   intent(inout)        :: a(lda,*)
       integer,   intent(in)           :: lda
       real(8),   intent(out)          :: w(1:n)
       real(8),   intent(out)          :: z(ldz,*)
       integer,   intent(in)           :: ldz
       integer,   intent(in), optional :: m_forward
       integer,   intent(in), optional :: m_backward
       character(*), intent(in), optional :: mode
       integer,   intent(in), optional :: precision
       end subroutine eigen_FS

       subroutine FS_init(comm,order) bind(c, name="FS_init")
         use, intrinsic :: iso_c_binding
     
         integer(c_int), intent(in), value :: comm
         character(c_char), intent(in), value:: order
       end subroutine

       subroutine FS_free() bind(c, name="FS_free")
         use, intrinsic :: iso_c_binding
       end subroutine 

       subroutine FS_get_matdims(n, nx, ny) bind(c, name="FS_get_matdims")
         use, intrinsic :: iso_c_binding
         integer(c_int), intent(in), value :: n
         integer(c_int), intent(out)       :: nx, ny
       end subroutine

       integer(c_int) function FS_get_myrank() bind(c, name="FS_get_myrank")
         use, intrinsic ::iso_c_binding
       end function

      integer(c_int) function FS_byte_data_context(N, int_size, real_size) &
         bind(c, name="FS_byte_data_context")
        use, intrinsic :: iso_c_binding
        integer(c_int), intent(in), value :: N, int_size, real_size
      end function

  end interface

contains

  !--------*---------*---------*---------*---------*---------*---------*-*
  !> subroutine FS_get_version
  !> @brief get version information
  !> @param[out] version   version
  !> @param[out] date      date
  !> @param[out] vcode     vcode
  subroutine FS_get_version(version, date, vcode)

    integer,      intent(out)           :: version
    character(*), intent(out), optional :: date
    character(*), intent(out), optional :: vcode


    version = FS_Version%Major_Version * 100 &
         + FS_Version%Minor_Version * 10  &
         + FS_Version%Patch_Level

    if (present(date)) then
       date = FS_Version%date
    end if

    if (present(vcode)) then
       vcode = FS_Version%vcode
    end if

    return

  end subroutine FS_get_version

  !--------*---------*---------*---------*---------*---------*---------*-*
  !> subroutine FS_show_version
  !> @brief print version information
  subroutine FS_show_version()

    use eigen_libs0_mod,only : eigen_get_id
    character(256) :: version
    character(1  ) :: patchlevel
    integer         :: i
    integer         :: id, x_id, y_id

    call eigen_get_id(id, x_id, y_id)

    i = min(26, FS_Version%Patch_Level) + 1
    patchlevel = (" abcdefghijklmnopqrstuvwxyz*" (i:i))

    write(version, '(I1,A,I1,A)')     &
         FS_Version%Major_Version, &
         '.',FS_Version%Minor_Version, trim(patchlevel)

    if (id == 1) then
       print*, "## FS version (", trim(version), &
            ") / (", trim(FS_Version%date),         &
            ") / (", trim(FS_Version%vcode), ")"
    end if

    return
  end subroutine FS_show_version
end module FS_libs_mod
