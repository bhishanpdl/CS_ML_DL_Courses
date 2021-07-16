!Bhishan Poudel
!Oct 14, 2015

! cmd: clear; gfortran dsyev_test.f90  -framework vecLib && ./a.out
! cmd: clear; gfortran dsyev_test.f90  -framework vecLib && ./a.out > hw7qn1edsy.dat

!cmd: clear; f95 dsyev_test.f90  -llapack && ./a.out
!cmd: clear; f95 dsyev_test.f90  -llapack && ./a.out > dsyev.dat

program test_dsyev
implicit none

    integer,parameter:: n=3
    integer,parameter:: lda=n
    integer,parameter:: lwmax=1000
    
    integer:: info,lwork,kwrite
    
    double precision:: a(lda,n)
    double precision:: w(n)
    double precision:: work(lwmax)

    data kwrite/6/
    
!     input matrix A
      data a / &
       1.d0, -4.d0,   2.d0,   &
      -4.d0,  1.d0,  -2.d0,   &
       2.d0, -2.d0,  -2.d0/

!   printing input matrix
    
    call print_matrix('Input Matrix', n,n,a,lda)
         

!   Query the optimal workspace.

    lwork = -1
    call dsyev('V','U',n,a,lda,w,work,lwork,info)
    lwork = min(lwmax, int(work(1)))

      
!    Solve eigenproblem.

     call dsyev('V','U',n,a,lda,w,work,lwork,info)

!    Check for convergence.


    if(info /=0) then
      write(kwrite,*) 'info=  ', info
      stop
    end if

!   Print eigenvalues.

    call print_matrix('Eigenvalues', 1,n,w,1)

!   Print eigenvectors.


    call print_matrix('Eigenvectors (columns)', n,n,a,lda)
    stop
    end program

!===============================================================================
!  This subroutine prints the matrix eigenvalues and eigenvectors
    subroutine print_matrix(desc,m,n,a,lda)
    
    character*(*),intent(in) :: desc ! description
    integer,intent(in) :: m,n,lda
    double precision,intent(inout):: a(lda,*)
    integer :: i,j
    
    write(*,*)
    write(*,*) desc
    
    do i=1,m
      write(*,10000) (a(i,j), j=1,n )
    end do 
    10000 format(5(:,2x,f6.3))  ! format works upto n=5 matrix
    return
    end subroutine
!===============================================================================    

!*  Purpose
!*  =======
!*
!*  DSYEV computes all eigenvalues and, optionally, eigenvectors of a
!*  real symmetric matrix A.
!*
!*  Arguments
!*  =========
!*
!*  JOBZ    (input) CHARACTER*1
!*          = 'N':  Compute eigenvalues only;
!*          = 'V':  Compute eigenvalues and eigenvectors.
!*
!*  UPLO    (input) CHARACTER*1
!*          = 'U':  Upper triangle of A is stored;
!*          = 'L':  Lower triangle of A is stored.
!*
!*  N       (input) INTEGER
!*          The order of the matrix A.  N >= 0.
!*
!*  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N)
!*          On entry, the symmetric matrix A.  If UPLO = 'U', the
!*          leading N-by-N upper triangular part of A contains the
!*          upper triangular part of the matrix A.  If UPLO = 'L',
!*          the leading N-by-N lower triangular part of A contains
!*          the lower triangular part of the matrix A.
!*          On exit, if JOBZ = 'V', then if INFO = 0, A contains the
!*          orthonormal eigenvectors of the matrix A.
!*          If JOBZ = 'N', then on exit the lower triangle (if UPLO='L')
!*          or the upper triangle (if UPLO='U') of A, including the
!*          diagonal, is destroyed.
!*
!*  LDA     (input) INTEGER
!*          The leading dimension of the array A.  LDA >= max(1,N).
!*
!*  W       (output) DOUBLE PRECISION array, dimension (N)
!*          If INFO = 0, the eigenvalues in ascending order.
!*
!*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
!*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
!*
!*  LWORK   (input) INTEGER
!*          The length of the array WORK.  LWORK >= max(1,3*N-1).
!*          For optimal efficiency, LWORK >= (NB+2)*N,
!*          where NB is the blocksize for DSYTRD returned by ILAENV.
!*
!*          If LWORK = -1, then a workspace query is assumed; the routine
!*          only calculates the optimal size of the WORK array, returns
!*          this value as the first entry of the WORK array, and no error
!*          message related to LWORK is issued by XERBLA.
!*
!*  INFO    (output) INTEGER
!*          = 0:  successful exit
!*          < 0:  if INFO = -i, the i-th argument had an illegal value
!*          > 0:  if INFO = i, the algorithm failed to converge; i
!*                off-diagonal elements of an intermediate tridiagonal
!*                form did not converge to zero.
!*
!*  =====================================================================

!Wolfram Alpha
!eigenvectors {{1,-4,2},{-4,1,-2},{2,-2,-2}}

!v1= 2,-2,1
!v2= -1,0,2
!v3= 1,1,0

!l1= 6
!l2= -3
!l3= -3

