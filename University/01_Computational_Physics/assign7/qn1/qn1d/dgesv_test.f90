! Bhishan Poudel
! Oct 12, 2015

! cmd: clear; gfortran dgesv_test.f90  -framework vecLib && ./a.out
! cmd: clear; gfortran dgesv_test.f90  -framework vecLib && ./a.out > hw7qn1d.dat

! cmd: clear; f95 dgesv_test.f90  -llapack && ./a.out
! cmd: clear; f95 dgesv_test.f90  -llapack && ./a.out > hw7qn1d.dat



! Usage: to solve linear equations using subroutine 'dgesv'


      program dgesv_test
      implicit none

      integer,parameter :: kwrite=6
      integer,parameter :: n=3, m1=1
      integer :: ipiv, info
      integer :: i, j

      double precision :: a,aa
      double precision :: b,bb

      dimension a(n,n),aa(n,n)
      dimension b(n),bb(n)
      dimension ipiv(n)


!     input matrix
      data a / &
               4.d0,  3.d0, 2.d0,  &
               -2.d0, 6.d0, 1.d0,  &
               1.d0, -4.d0, 8.d0   /
               
               
!     input rhs vector B
      data b /4.d0,-10.d0,22.d0/
      
!     backup data
      aa=a; bb=b


!     print inputs

      write(kwrite,*) 'This program solves X for AX=B'
      write(kwrite,*)
     
      write (kwrite,*) 'Matrix A :'
      do i=1,n
        write (kwrite,10000) a(i,1), a(i,2), a(i,3)
      end do
      
      write (kwrite,*)
      write (kwrite,*) 'Column vector B:'
      write (kwrite,10000) (b(j),j=1,n)

      
!     call the subroutine dgesv for column vector B1
      call dgesv (n,m1,a,n,ipiv,b,n,info)

!     check for convergence
      if ( info .ne. 0 ) then
      write (kwrite,*) ' info  = ',info
      endif

!         write the results

      write (kwrite,*)
      write (kwrite,*) 'Result X for column vector B:'
      write (kwrite,10000) (b(j),j=1,n)

!     testing results
      write(kwrite,*)
      write(kwrite,*) 'The first row of matrix A'
      write(kwrite,*) aa(1,1), aa(1,2), aa(1,3)
      
      write(kwrite,*)
      write(kwrite,*) 'The first value of column vector B is:'
      write(kwrite,*) aa(1,1)*b(1)+ aa(1,2)*b(2) + aa(1,3)*b(3)


10000 format (3(1x,f10.3))

      stop
      end
