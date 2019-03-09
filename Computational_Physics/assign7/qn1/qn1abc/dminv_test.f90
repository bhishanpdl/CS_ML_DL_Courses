!Bhishan Poudel
!cmd: clear; gfortran dminv_test.f90  -framework vecLib && ./a.out > hw7qn1.dat
!cmd: clear; f95 dminv_test.f90 && ./a.out

!cmd: clear; f95 dminv_test.f90 -llapack && ./a.out

! Purpose:
! This program is to test the suboutine dminv- a lapack routine to find the inverse of the general matrix
! Required Programs: dminv.f90
! we will use: subroutine dminv (a,n,d,l,m)

! input matrix      inverse matrix is (from wolfram alpha)
! 4  -2  1          1     52   17  2
! 3   6  -4        --     -32  30  19 
! 2   1   8        263    -9   -8  30  


      program dminv_test  
      implicit none 
     

      integer,parameter :: n = 3 ! order of matrix A
      integer,parameter :: kwrite=6
      integer :: i,j

      double precision :: a,b   ! input matrix (2d array)
      double precision :: d     ! resultant determinant
      integer :: lwork ! work vector of length n (1d array)
      integer :: work  ! work vector of length n (1d array)

      
      dimension :: lwork(n)
      dimension :: work(n)
      dimension :: a(n,n),b(n,n)


!     input matrix
      data a / &
               4.d0,  3.d0, 2.d0,  &
               -2.d0, 6.d0, 1.d0, &
               1.d0, -4.d0, 8.d0 /
               
!     storing original a in b and printing input matrix              
      b=a      
      call print_matrix('Input  A', n,n,a,n)

! call the  subroutine dminv
! original is: subroutine dminv (a,n,d,l,m)
      call dminv(a,n,d,lwork,work)
      
!     determinant
      write(kwrite,*)
      write(kwrite,*)'Determinant:',d
      
!     inverse matrix         
      call print_matrix('Inverse of A', n,n,a,n)
      
!     inverse divided by determinant      
      write(kwrite,*)
      write(kwrite,*) 'Inverse divided by determinant is:'
      do i=1,n
         write(kwrite,10000) (a(i,j)*263.d0,  j= 1,n)
      end do
      
!    input multiplied by inverse
      write(kwrite,*)
      write(kwrite,*) 'input multiplied by inverse:'
      do i=1,n
         write(kwrite,20000) (b(i,j)*a(i,j),  j= 1,n)
      end do
      
!    inverse multiplied by input
      write(kwrite,*)
      write(kwrite,*) 'inverse multiplied by input:'
      do i=1,n
         write(kwrite,20000) (a(i,j)*b(i,j),  j= 1,n)
      end do
   
      
      
10000 format(3(2x,f8.0))
20000 format(3(2x,f14.2))


      end program
