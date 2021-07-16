!cmd: clear; f95 euler.f90 && ./a.out; rm -f *fort* *~

!Description: This program solves the differential equation y'(x) = 2(y+1), -2<x<2
!             using Euler's method for different step sizes h = 0.05,0.10,0.15.0.20

Program euler
implicit none
      
      integer          :: i,n
      double precision :: y,x,f
      double precision :: a,b,h,exact,error
      integer          :: kwrite,eps
      
      data eps/1e-16/, kwrite/6/
      
      
      !write outputs
      open(unit=kwrite,file='euler875.dat',status='unknown')
      
      !x varies from 0 to 2 in both + and - direction
      a=0.d0
      b=2.d0
  
          ! for negative values from a to -b    
          h=0.875d0
          n = int((b-a)/h)
          
          write(kwrite,*)'#                Euler method to solve the differential equation'
          write(kwrite,100)'#              step size h = ',h
          write(kwrite,*)  '#                         '
          write(kwrite,*)  '#           x                        y               exact               error'         
          write(kwrite,*)  '#                         '

          x=a   
          y=0.d0  

          do 20 i=1,n+1
              exact=exp(2.d0*x)-1.d0  ! f =y'=2y+2, exact = exp(2x) -1
              
              !avoid floating point exception when exact = 0.d0
              if (abs(exact).gt.eps) then
              error=abs(y-exact)/abs(exact)
              else
              error = 0.d0
              end if
 
              write(kwrite,10000)x,y,exact,error
      
              y=y-h*f(y)
              x=x-h
          20 end do
      
          
          !for positive values from a to b
          x=a
          y=0.d0
          do 30 i=1,n+1
          
              exact=exp(2.d0*x)-1 ! f =y'=2y+2, exact = exp(2x) -1
              
              !avoid floating point exception when exact = 0.d0
              if (abs(exact).gt.eps) then
              error=abs(y-exact)/abs(exact)
              else
              error = 0.d0
              end if

              write(kwrite,10000)x,y,exact,error
      
              y=y+h*f(y)
              x=x+h
          30 end do

      
      100 format(a30,f5.2)
      10000 format(4(f20.6))
      stop 'data is saved' 
      end program

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! function
double precision function f(yy)
implicit none
      double precision::yy

      f=2.d0*yy+2.d0  ! f =y'=2y+2, exact = exp(2x) -1

end function f
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

