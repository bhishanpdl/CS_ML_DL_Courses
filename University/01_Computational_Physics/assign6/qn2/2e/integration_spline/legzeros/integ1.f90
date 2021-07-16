!bp: clear; f90 integ1.f90 && ./a.out >legzeros.dat


include 'gauleg.f90'
      program integ1
!
!
      implicit none

      integer :: nxmt,nxms,ng,nsxx,ngp
      integer :: kread, kwrite
      integer :: i,nxmx,nxmx1,nxmx2
      double precision :: pi,sxx,wxx,u,w,err1,err2,err3
      double precision :: xival,dx,sum1,sum2,sum3,d43,d23
 
!        integration points
 
      parameter (nsxx=2000)
      parameter (ngp=64)
      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
 
      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngp),w(ngp)
!     write (kwrite,*) '------------------------------------------------&
!       &-------------------------------' 
!      do nxmt=2,100
      ng=18
      xival=200.d0
!
!        gauss integration
!
!        get gauss points and weights
!
      call gauleg (0.d0,xival,u,w,ng)  ! integrating from 0 to 200
 
 
 
      do i=0,ng
      sxx(i)=u(i)
      wxx(i)=w(i)

      write(kwrite,10002) u(i),w(i)
      end do
!
!
!       integrate
!
!      sum3=0.d0
!      do  i=1,ng
!      sum3=sum3 + ( 3*(sxx(i))**(3) )*wxx(i)
!      end do
 
      
!      write (kwrite,10000) nxmx,sum3
 
10000 format(1x,i3,1x,f9.4,3(1x,f19.15)) 
10001 format(2x,A,7x,A,10x,A,14x,A,14x,A)
10002 format(2x,f9.2,3x,f8.2)
!      end do 
      stop 
      end



