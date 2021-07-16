!cmd: clear; gfortran -Wall hw9qn2d.f90 && ./a.out 

!cmd: clear; f95 hw9qn2d.f90 && ./a.out


program hw9qn2d


      implicit none
      external f1,f2,f3

      integer :: ng,nsxx,ngmax
      integer :: kread, kwrite
      integer :: i
      double precision :: pi,sxx,wxx,u,w,lower,upper ! pi is defined below
      double precision :: sum1,sum2,sum3
      double precision :: true1,true2,true3
      double precision :: err1,err2,err3
      double precision :: f1,f2,f3

      ! parameters
      parameter (nsxx=9000,ngmax=9000)  ! maximum no. of iterations

      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngmax),w(ngmax)

      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      open(unit=kwrite,file='hw9qn2d.dat',status='unknown')
      write(kwrite,10000)'gauss points','final', &
                       & '1st','1st error','2nd', '2nd error', '3rd','3rd error'
      do 10 ng= 1,200  !  **** CHANGE
   
      
      lower = 0.d0   ! *** CHANGE
      upper = 1.d0
      
      !!!!!!!!!!!!!!!!! Calling subroutine !!!!!!!!!!!!
      !    subroutine gauleg (x1,x2,x,w,n)
      !    x(n) and w(n) are 1d arrays of n elements output
      !    x1,x2,n are inputs
      
      call gauleg (lower,upper,u,w,ng)
 
      do i=1,ng
      sxx(i)=u(i)
      wxx(i)=w(i)
      end do

      ! integrate

      sum1=0.d0
      sum2=0.d0
      sum3=0.d0
      do i=1,ng 
      sum1 = sum1 + f1(sxx(i))*wxx(i)
      sum2 = sum2 + f2(sxx(i))*wxx(i)
      sum3 = sum3 + f3(sxx(i))*wxx(i)
      end do
      
      true1 = 0.78540d0
      true2 = 0.31606d0
      true3 = 1.49365d0
      
      !write(kwrite,10000)'gauss points','final', &
      !                 & '1st','1st error','2nd', '2nd error', '3rd','3rd error'
      write(kwrite,20000) ng,sum1*sum2*sum3, &
                        &  sum1,(sum1-true1)/true1, &
                        &  sum2,(sum2-true2)/true2, &
                        &  sum3,(sum3-true3)/true3 

      10000 format(A12, A10,   3(a14,a14))
      20000 format(I12, f10.3, 3(f14.5,e14.3)) 

      10 end do
      close(kwrite)
      
      stop'data saved in hw9qn2d.dat' 
      end program
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
double precision function f1(x)

       implicit none
       double precision x

       f1 = 1.d0/(1.d0+ x*x)
 
       return
       end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
double precision function f2(y)

       implicit none
       double precision y

       f2 = y*exp(-y*y)
 
       return
       end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
double precision function f3(z)

       implicit none
       double precision z

       f3 = exp(-z)/sqrt(z)
 
       return
       end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine gauleg (x1,x2,x,w,n) 
!
!        calculate gauss points for gauss-legendre integration
!                 (numerical recipes)  
!
      implicit double precision (a-h,o-z) 
!
      parameter (eps=3.d-14) 
      dimension x(n),w(n) ! x(n) and w(n) are 1d arrays of n elements
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
      end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!time ./a.out 
!real	0m0.098s
!user	0m0.094s
!sys	0m0.004s


