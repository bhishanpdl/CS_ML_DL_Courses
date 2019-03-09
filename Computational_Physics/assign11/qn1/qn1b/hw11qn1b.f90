!cmd: clear; f95 hw11qn1b.f90 && ./a.out; rm -f *~ *.o *fort*

program differentiation_methods
implicit none

    external f1
    integer          ::i,j,n,m,nn
    double precision ::y,x,f,y1,y2,x2,yout,dydx,dydx1
    double precision ::a,b,h,exact,error1,error2,error3
    integer          :: kwrite
    
    data kwrite/6/

    open(unit=kwrite,file='hw11qn1b.dat',status='unknown')

    !range of x
    a=0.d0
    b=2.d0

    n=20  ! to make h = 0.1
    nn=1  ! first order diff eq for rk4

    h=(b-a)/dfloat(n)
    
    write(kwrite,*)  '#step size h = ',h
    write(kwrite,200)'#x','exact','Euler','error','Imp_Euler','error','rk4','error'

    x=a 

    y=0.d0  
    y1=0.d0
    y2=0.d0

    ! Modified Euler's Method

    do i=1,n+1
        exact=exp(2.d0*x)-1.d0
 
        ! errors for three methods
        error1=abs(y-exact)
        error2=abs(y1-exact)
        error3=abs(y2-exact)
 
        write(kwrite,10000)x,exact,y,error1,y1,error2,y2,error3
        y=y+h*f(y)
        y1=y1+(h/2.d0)*(f(y1)+f(y1+h*f(y1)))
  
        ! rk4 method
        call f1(x,y2,dydx)   
        call rk4(y2,dydx,nn,x,h,yout,f1)
  
        x=x+h
        y2=yout
        end do
        
    !For negative values
    x=a 

    y=0.d0
    y1=0.d0
    y2=0.d0

    do i=1,n+1
        exact=exp(2.d0*x)-1.d0

        ! errors for three methods
        error1=abs(y-exact)
        error2=abs(y1-exact)
        error3=abs(y2-exact)

        write(kwrite,10000)x,exact,y,error1,y1,error2,y2,error3
        y=y-h*f(y)
        y1=y1-(h/2.d0)*(f(y1)+f(y1-h*f(y1)))

        ! rk4 method
        call f1(x,y2,dydx)
        call rk4(y2,dydx,nn,x,-h,yout,f1)

        x=x-h
        y2=yout
   end do


10000 format(8(f14.6))
200   format(8a14) 

end program
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! function 

  double precision function f(yy)
  implicit none

      double precision::yy

      f=2.d0*yy+2.d0

end function f

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine f1 (x,y,dydx)
      implicit none

      double precision x
      double precision y,dydx

           dydx=2.d0*y+2.d0  !dy/dx = 2y + 2, exact = exp(2x) -1 and y(0) = 0

      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine rk4(y,dydx,n,x,h,yout,derivs)

!        from numerical recipes
!        Given values for the variable y and their derivatives dxdy known at x, use
!        the fourth-order Runge-Kutta method to andvance the solution over an interval h
!        and return the incremented variables as yout, 
!        which need not be a distinct array from y.
!        The user supplies the subroutine derivs(x,y,dydx), 
!        which returns the derivatives dydx at x.
!        n is the number of equations

      implicit none

      external derivs

      integer nmax
      parameter (nmax=50)

      integer n, i
      double precision  h,x,dydx(n),y(n),yout(n)
      double precision  h6,hh,xh,dym(nmax),dyt(nmax),yt(nmax)

      hh=h*0.5
      h6=h/6.
      xh=x+hh
      do i=1,n
        yt(i)=y(i)+hh*dydx(i)
      end do

      call derivs(xh,yt,dyt)

      do  i=1,n
        yt(i)=y(i)+hh*dyt(i)
      end do

      call derivs(xh,yt,dym)

      do  i=1,n
        yt(i)=y(i)+h*dym(i)
        dym(i)=dyt(i)+dym(i)
      end do

      call derivs(x+h,yt,dyt)

      do i=1,n
        yout(i)=y(i)+h6*(dydx(i)+dyt(i)+2.*dym(i))
      end do

      return
      end

!       Copr. 1986-92 Numerical Recipes Software 2721[V3.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
