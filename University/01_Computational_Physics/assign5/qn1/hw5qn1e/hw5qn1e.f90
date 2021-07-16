!Template: integ1.f90, gauleg.f90 were provided by Dr. Elster

! Author: Bhishan Poudel
! Date  : Sep 27,2015

! cmd: clear; f90 hw5qn1e.f90 && ./a.out


  program hw5qn1e
!
!        integrate  sin(x) from 0 to xival 
!        with different integration schemes:
!        Trapezoial, Simpson, Gauss
!
      implicit none

      integer :: nxmt,nxms,ng,nsxx,ngp
      integer :: kread, kwrite
      integer :: i,nxmx,nxmx1,nxmx2
      double precision :: pi,sxx,wxx,u,w, upper, lower
      double precision :: xival,dx,sum,d43,d23
 

   ! integration points

 
      parameter (nsxx=2000)
      parameter (ngp=64)
      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
 
      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngp),w(ngp)
 
      xival=1.0

      open(kwrite, File = 'hw5qn1e.dat', Status = 'Unknown')
      write(kwrite,*)'# N           Trapezoidal-Value'
      
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!       Integration with Trapezoidal rule
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!       set up grid for trapezoidal rule

      nxmt=3
      sum = 1.d0
      lower = 0.d0
      upper = 100.d0
      do while ((abs(sum-lower)).gt. (epsilon(1d-14)))
      nxmx=nxmt/2
      nxmx=nxmx*2+1
 
      dx=xival/dfloat(nxmx-1)
 
      wxx(1)=dx*0.5d0
      wxx(nxmx)=dx*0.5d0
      
      sxx(1)=0.d0
      sxx(nxmx)=xival
 
      nxmx1=nxmx-1
 
      do 10 i=1,nxmx1
      sxx(i)=dfloat(i-1)*dx
   10 wxx(i)=dx
!
!       integrate
!
      sum=0.d0
      do 20 i=1,nxmx
   20 sum=sum +  (((sxx(i))**4)*((1-(sxx(i))**2)**4)/(((sxx(i))**2)+1))   *wxx(i)


      write(kwrite,10000) nxmx,sum
      lower = upper
      upper=sum
      nxmt = nxmt+2
      end do
  
      10000 format(t1, i3, 4x,f17.8)      

      close(kwrite)
      stop 'Data saved in the hw5qn1e.dat'
      end program
