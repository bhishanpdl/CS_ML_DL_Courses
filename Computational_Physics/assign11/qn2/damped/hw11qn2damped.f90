! clear; f95 hw11qn2damped.f90 && ./a.out


   program hw11qn2damped
      implicit none
      external f2

      double precision x,h,hq
      double precision y(22),dy(22),yout(22)
      double precision xinitial, xfinal
      double precision pih,pi
      double precision rkq
      double precision d12,fd12,den

      double precision xx(1250),vv(1250)
      double precision yy(1250)
      integer i, ix, n, ntot

      integer kread, kwrite
      data kread/5/, kwrite/6/
  

      data pih/1.570796326794897d0/
      pi = 2*acos(0.d0)
   

      n=2
      xinitial = 0.0d0
      xfinal   = 10.d0    
      h=0.05d0
      
      ! writing output
      open(kwrite,File ='overdamped.dat', Status = 'unknown')
      
          write (kwrite,*)    '#  Runge-Kutta for damped harmonic oscillator'
          write (kwrite,1002) '#x','y','analytical','Relative-error'
          write (kwrite,*)

      !  initialize for rk4

      x=xinitial
      !x= -0.05d0
      y(1)=1.d0      
      y(2)=0.d0

      ix=0

 100  continue

      call f2 (x,y,dy)
      call rk4 (y,dy,n,x,h,yout,f2)

      ix=ix+1
      xx(ix)=x+h
      yy(ix)=yout(1)
     ! vv(ix) = yout(2)

      x=x+h
      y(1)=yout(1)
      y(2)=yout(2)

      if (x.le.xfinal) go to 100
 
      do i=1,ix
          !E(i)= 0.5*4*pi**2*yy(i)**2 + 0.5*vv(i)**2
          write (kwrite,1001) xx(i),yy(i), sin(2*pi*xx(i)),abs((sin(2.d0*pi*xx(i))-yy(i))/sin(2*pi*xx(i)))
      end do

    1001 format(6d18.6)
    1002 format(6A18)
    
    close(kwrite)
    stop 'data is saved in hw11qn2damped.dat'
    end  program

!---------------------------------------------------------------
subroutine f2 (xl,y,dy)
!
!       function y''= -4*pi*pi*y    
!
      implicit none

      double precision xl,pi,b
      double precision y(22),dy(22)
      
    pi = 2.d0 * acos(0.d0)
    !b  = pi  !under damped
    !b  = 4.d0 * pi  !critical damped 4 PI
    b  = 15.d0 * pi  !over damped


      dy(1) = y(2)
      dy(2) = -4.d0*pi**2*y(1) - b*y(2)

      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
 subroutine f1 (xl,y,dy)
!
!       function y'=-xy
!
      implicit none

      double precision xl
      double precision y,dy

      dy= 2.d0*(y+1.d0)

      return
      end 
