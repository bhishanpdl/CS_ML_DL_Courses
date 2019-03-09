!Bhishan Poudel
!Oct 15, 2015

! cmd: clear; gfortran dgeev_test.f90  -framework vecLib && ./a.out

!cmd: clear; f95 dgeev_test.f90  -llapack && ./a.out
!cmd: clear; f95 dgeev_test.f90  -llapack && ./a.out > hw7qn1f.dat

! hw7 1n1f for non symmetric matrix

       integer,parameter:: kwrite=6
       integer,parameter:: n=3
       integer,parameter:: lda=n,ldvl=n,ldvr=n
       integer,parameter:: lwmax=1000
       
       integer:: info,lwork,i,j

      double precision:: a(lda,n),vl(ldvl,n), vr(ldvr,n)
      double precision:: wr(n),  wi(n),  work(lwmax)


!     input matrix A
      DATA             A/ &
      -2.d0, 2.d0, -1.d0, &
       2.d0, 1.d0, -2.d0, &
      -3.d0,-6.d0, 0.d0 /


!     print input matrix
      call print_matrix('input matrix A',n,n,a,n)
                       

!     Query the optimal workspace.
      lwork = -1
      call dgeev( 'Vectors', 'Vectors', n,a,lda,  wr,wi,   vl,ldvl,  vr,ldvr,   work,lwork, info )
      lwork = min( lwmax, int( work( 1 ) ) )
      

!     Solve eigenproblem.
      call dgeev( 'Vectors', 'Vectors', n,a,lda,  wr,wi,   vl,ldvl,  vr,ldvr,   work,lwork, info )

!     Check for convergence.
      if( info/=0 ) then
         write(*,*) 'info=  ', info
      end if

!     Printing Eigen Values

      call print_matrix('Real eigenvalues',n,1,wr,n)
!      write(kwrite,*)
!      write(kwrite,*)'Real Eigen values wr:'
!      write(kwrite,10000)(wr(j), j=1,n)
      
      call print_matrix('Imaginary eigenvalues',n,1,wi,n)
!      write(kwrite,*)
!      write(kwrite,*)'Imaginary Eigen values wi:'
!      write(kwrite,10000)(wi(j),j=1,n)
!      write(kwrite,*)

!     Printing Left Eigen vectors

       call print_matrix('Left eigenvectors',n,n,vl,n)
!      write(kwrite,*)'Left Eigen Vector:'
!      do i=1,n
!      write(kwrite,10000)(vl(i,j), j=1,n)
!      end do
      
!     Printing Right Eigen Vectors

      call print_matrix('Right eigenvectors',n,n,vr,n)
!      write(kwrite,*)'Right Eigen Vector:'
!      do i=1,n
!      write(kwrite,10000)(vr(i,j), j=1,n)
!      end do

10000 format (3(1x,f6.2))
      
      stop
      end program
      
!===============================================================================     
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
    10000 format(5(:,2x,f14.2))  ! format works upto n=5 matrix
    return
    end subroutine
!===============================================================================  

