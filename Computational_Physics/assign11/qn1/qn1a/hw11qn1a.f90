!cmd: clear; f95 hw11qn1a.f90 && ./a.out; rm -f *fort* *~

!Description: This program solves the differential equation y'(x) = 2(y+1), -2<x<2
!             using Euler's method for different step sizes h = 0.05,0.10,0.15.0.20

Program euler
implicit none
      
      integer          ::i,j,n,m
      double precision ::y,x,f
      double precision ::a,b,h,exact,error,eps
      
      data eps/1e-14/
      
      
      !write outputs
      open(unit=1,file='euler05.dat',status='unknown')
      open(unit=2,file='euler10.dat',status='unknown')
      open(unit=3,file='euler15.dat',status='unknown')
      open(unit=4,file='euler20.dat',status='unknown')
      
      !x varies from 0 to 2 in both + and - direction
      a=0.d0
      b=2.d0

      ! make a loop in steps of 0.05, 0.10, 0.15, 0.20
      m=5
      do 10 j=1,4
          n=int(200/m)
          h=(b-a)/dfloat(n)
          
          write(j,*)'#                Euler method to solve the differential equation'
          write(j,100)'#              step size h = ',h
          write(j,*)  '#                         '
          write(j,*)  '#           x                        y               exact               error'         
          write(j,*)  '#                         '

          x=a   
          y=0.d0  

          do 20 i=1,n+1
              exact=exp(2.d0*x)-1.d0
              
              if (abs(exact-eps).gt.eps) then
              error=abs(y-exact)/abs(exact)
              end if
 
              write(8,10000)x,y,exact,error
              write(j,10000)x,y,exact,error
      
              y=y-h*f(y)
              x=x-h
          20 end do
      
          x=a
          y=0.d0
          do 30 i=1,n+1
              exact=exp(2.d0*x)-1
              if (abs(exact-eps).gt.eps) then
              error=abs(y-exact)/abs(exact)
              end if

              write(8,10000)x,y,exact,error
              write(j,10000)x,y,exact,error
      
              y=y+h*f(y)
              x=x+h
          30 end do

          m=m+5

      10 end do
      
      100 format(a30,f5.2)
      10000 format(4(f20.6))
      stop 'data is saved' 
      end program euler
      
!get rid of floating point exception
! if (abs(exact-eps).gt.eps) then
! error=abs(y-exact)/abs(exact)
! end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! function
double precision function f(yy)
implicit none
      double precision::yy

      f=2.d0*yy+2.d0

end function f
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

