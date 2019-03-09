!Templates were provied by Dr. Elster
!cmd: clear;f90 hw5qn2b.f90 && ./a.out


! usage    : program for the integration  of sin^2 (x) between the limits 0 to pi
!            equivalent to 2* integral of 0 to 1 
! methods  : Trapeziodal, Simpson, Gauss Methods
! precision: double precision


program hw5qn2b

      !implicit none

      integer :: nxmt,nxms,ng,nsxx,ngp
      integer :: kread, kwrite,j
      integer :: i,nxmx,nxmx1,nxmx2,k
      double precision :: pi,sxx,wxx,u,w, exact
      double precision :: xival,dx,d43,d23
      double precision ::sum1,sum2,sum3,e1,e2,e3   
      
      

      
!      integration points
!      parameter (nxmt=30,nxms=20)
!      parameter (ng=8)
 
      parameter (nsxx=1000)
      parameter (ngp=1000)
      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
 
      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngp),w(ngp)
 
      xival= pi
      


      open(kwrite, File = 'hw5qn2b.dat', Status = 'Unknown')
      
      write (kwrite,10000) '#N','h','T-rule,eT','S-rule,eS','G-Quad,eG'

      exact = 1.57079632679
      
      


j=1
do k=1, 200
    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!       Integration with Trapezoidal rule
!
!
!       set up grid for trapezoidal rule
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      nxmt=(4.d0*j)+1
      nxmx=nxmt/2
      nxmx=nxmx*2+1
 
      dx=xival/float(nxmx-1)
 
      wxx(1)=dx*0.5d0
      wxx(nxmx)=dx*0.5d0
      
      sxx(1)=0.d0     
      sxx(nxmx)=xival
 
      nxmx1=nxmx-1
 
      do 10 i=1,nxmx1
      sxx(i)=float(i-1)*dx    
10    wxx(i)=dx
      

!       integrate
!
      sum1=0.d0
      do 20 i=1,nxmx
20       sum1 = sum1 + ((sin(sxx(i)))**2)     *wxx(i)
         
  
 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!       Simpson integration
!
!       set up grid for Simpson integration
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      nxms=(4.d0*j)+1
      nxmx=nxms/2
      nxmx = (nxmx*2.d0) + 1
 
      dx=xival/float(nxmx-1)
 
      d43=4.d0/3.d0
      d23=2.d0/3.d0
 
      wxx(1)=dx/3.d0
      wxx(nxmx)=dx/3.d0
      
      sxx(1)=0.d0
      sxx(nxmx)=xival
 
      nxmx1=nxmx-1
      nxmx2=nxmx-2
 
      do  i=2,nxmx1,2
      sxx(i)=float(i-1)*dx
      wxx(i)=d43*dx
      end do
 
      do  i=3,nxmx2,2
      sxx(i)=float(i-1)*dx
      wxx(i)=d23*dx
      end do
!
!       integrate
!
      sum2 = 0.d0
      do  i=1,nxmx
         sum2 = sum2 +((sin(sxx(i)))**2)     *wxx(i)
         end do
     
 


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!        gauss integration
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!        get gauss points and weights

      ng = (4.0*j)+1
      call gauleg (0.d0,xival,u,w,ng)
	
 
 
      do i=1,ng
      sxx(i)=u(i)
      wxx(i)=w(i)
      end do
!
!
!       integrate
!
      sum3 = 0.d0
      do  i=1,ng
      sum3 = sum3 +((sin(sxx(i)))**2)     *wxx(i)
   end do
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   ! Relative errors
   
   e1 = abs((sum1-exact)/exact)
   e2 = abs((sum2-exact)/exact)
   e3 = abs((sum3-exact)/exact)
   
   
      write(kwrite,10001) nxmx, dx,  sum1,sum2,sum3, e1,e2,e3 
      
     
      j = j+1
    
   end do
   
   10000 format(A3, 4(A26,3x))
   10001 format(I3, 4(E26.12,3x), /t33, 3(E26.12,3x)/)  
   
   close(kwrite)
   stop 'data saved in hw5qn2b.dat'
   end program
 
 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  subroutine gauleg (x1,x2,x,w,n) 
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!        calculate gauss points for gauss-legendre integration
!                 (numerical recipes)  
!!
      implicit double precision (a-h,o-z) 
!
      parameter (eps=3.d-14) 
      dimension x(n),w(n)
!
      m=(n+1)/2 
      xm=0.5d0*(x2+x1) 
      xl=0.5d0*(x2-x1) 

      do 12 i=1,m 
        z=cos(3.141592654d0*(i-.25d0)/(n+.5d0)) 
   1    continue 
          p1=1.d0 
          p2=0.d0 
          do 11 j=1,n 
            p3=p2 
            p2=p1 
            p1=((2.d0*j-1.d0)*z*p2-(j-1.d0)*p3)/j 
  11   continue 
       pp=n*(z*p1-p2)/(z*z-1.d0) 
       z1=z 
       z=z1-p1/pp 

       if (abs(z-z1).gt.eps) go to 1 

       x(i)=xm-xl*z 
       x(n+1-i)=xm+xl*z 
       w(i)=2.d0*xl/((1.d0-z*z)*pp*pp) 
       w(n+1-i)=w(i) 
  12  continue 

      return 
      end 
