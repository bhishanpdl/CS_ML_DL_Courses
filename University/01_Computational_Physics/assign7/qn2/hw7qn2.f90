!Bhishan Poudel
!Oct 16, 2015

!bp: clear; gfortran -framework vecLib hw7qn2.f90 && ./a.out
!bp: clear; gfortran -framework vecLib hw7qn2.f90 && ./a.out > matrix3.dat

!bp: clear; f95 -llapack hw7qn2.f90 && ./a.out > hw7qn2.dat

program hw7qn2
implicit none

    external potential
    real(8) :: potential
    integer, parameter:: nstep=100, rmin=-10, rmax=10, kwrite=6
    real(8):: h,hinv,x,const1, const2
    real(8):: xk(0:nstep),  vk(0:nstep)
    real(8):: dk(1:nstep-1), ek(1:nstep-1)   
    real(8), dimension(nstep,nstep) :: a 
    integer                 :: i, j
 
    
    ! variables for subroutine dsyev
    integer,parameter:: lda=nstep
    integer,parameter:: lwmax=1000    
    integer:: info,lwork  
    double precision:: w(nstep)
    double precision:: work(lwmax)    

    
!   giving values to step size h    
    h= dble(rmax)-dble(rmin)
    h = h/dble(nstep)
    hinv = 1.d0 / h 
       
!   printing initial parameters     
    write(kwrite,*) '    rmin     rmax   nstep     h        hinv '
    write(kwrite,100) rmin,rmax,nstep, h, hinv
    write(kwrite,*)

    
!   printing x values and potentials
    write(kwrite,*) '  i      xk(i)      vk(i)    '
    x=rmin
    do i= 0,nstep
        xk(i) = rmin + dble(i) * h
        vk(i) = potential(xk(i))
        write(kwrite,10000) i,xk(i), vk(i)        
    end do
    
!   printing h values and inverse of h    
    const1= 2.d0/(h*h)
    const2= -1.d0/(h*h)
    write(*,*)
    write(kwrite,*) '     h        2/h^2   -1/h^2'
    write(kwrite,1000) h,const1, const2
    write(*,*)
    
!   printing diagonal and non-diagonal elements    
    write(kwrite,*) '  i       dk(i)      ek(i)    '
    do i= 1,nstep
        dk(i) = 2.d0 * hinv * hinv + vk(i)  ! dk = (2/ h^2) + Vk
        ek(i) = const2
        write(kwrite,10000) i,dk(i), ek(i)
        
    end do
    
!initialize matrix a(i,j) to all zeros   
    do i = 1, nstep      ! iteration i = 1 is first column
      do j = 1, nstep    
          a(i, j) =0.d0    
      end do
   end do
    
! filling diagonal elements    
    do i = 1, nstep      ! iteration i = 1 is first column
      do j = 1, nstep  
      
          if(i.eq.j) then
             a(i, j) = dk(i)
          end if
      end do
   end do
 
! filling elements right and below the diagonal elements
   const2= -1.d0/(h*h)     
   do i = 1, nstep      ! iteration i = 1 is first column
       if(i/=nstep) then
               a(i, i+1) = const2  ! filling right
               a(i+1, i) = const2  ! filling below
       end if
   end do

!    printing matrix A  
     call print_matrix('input matrix A', nstep,nstep,a,nstep)
    
!    Solve eigenproblem.
     lwork=lwmax
     call dsyev('V','U',nstep,a,lda,w,work,lwork,info)    
    
!   Print eigenvalues.
    call print_matrix('Eigenvalues', 1,nstep,w,1)
    
!   print first three eigenvalues
    call print_matrix('First three eigenvalues', 1,3,w,1)
        
!   Print eigenvectors.
    !call print_matrix('Eigenvectors (columns)', nstep,nstep,a,lda)     
    
!   print first three eigenvector
    open(kwrite,file='u123.dat',status='unknown')
    write(kwrite,*)
    do i=1,nstep
          write(kwrite,20000) i, xk(i), a(i,1),a(i,2),a(i,3)
    end do
    close(kwrite)
    
100  format(3i8, 2f10.2)    
1000 format(3f10.2)
10000 format(i4,2x,2f10.3)
20000 format(i4,4f10.3)  
end program    
   
    
!*******************************************************************************
! harmonic oscillator potential
real(8) function potential(x)
implicit none
    real(8),intent(in)::x
    
    potential= x*x

end function potential

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
    10000 format(1000(:,1x,f7.3)) 
    return
    end subroutine
!===============================================================================
