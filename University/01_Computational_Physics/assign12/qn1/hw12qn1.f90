! clear; f95 hw12qn1.f90 && ./a.out; rm *~

     
      program legendre
      implicit none

      double precision x,h, y(2),dy(2),yout(2)
      double precision xinitial,xfinal,d(100),c(100)
      double precision xx(15000), yy(15000), eps
      integer i, ix, n,l
      integer kread, kwrite,kwrite1,order
      data kread/5/, kwrite/6/,kwrite1/10/
      
      n=2 		!number of differential equations

      !Limits of Legendre Polynomial
      xinitial = -0.999999d0
      xfinal   =  0.999999d0
      h=0.001d0
      eps = (10.d0**(-6.d0)) !precision
      
   
    order = 5                                      !*** change
    open(kwrite, FILE='n5.dat', Status='Unknown')  !*** change n=3,value: 3*4=12
    open(kwrite1, FILE='n5compare.dat', Status='Unknown')
    write(kwrite,*)'#     x','               ', 'legendre p(',order,',x)'


    ! Initial guess for derivatives of 'P'
	d(1)=2.d0
	d(2)=8.d0
	c(1)=1.d0
	
	do l=1,25		
	    x=xinitial
	    y(1)=-1.d0	      
	    y(2)=d(l)


	     ix=0

         100 continue

	     call f (x,y,dy)

	     call rk4 (y,dy,n,x,h,yout,f)

	    ix=ix+1
	    xx(ix)=x+h
	    yy(ix)=yout(1)

	    x=x+h
	    y(1)=yout(1)
	    y(2)=yout(2)
 
	    if (x.le.xfinal) go to 100 
      
  
        ! secant method
		if(l .ge. 2) then
		    c(l)=yy(ix)
		    d(l+1)=d(l)-(c(l)-1.d0)*(d(l)-d(l-1))/(c(l) -c(l-1))
	    end if

	    if ((abs(yout(1)-1.d0)) .le. eps) exit
	end do

    !printing results 
	do i=1,ix
	    write (kwrite,10000) xx(i),yy(i)
	end do
	
	!printing results
    write(kwrite1,*)'#     x','               ', 'legendre p(',order,',x)' 
    do i=1,ix,100
        write (kwrite1,10000) xx(i),yy(i)
    end do


    10000 format(2x, f8.6,2x, f22.9)
    close(kwrite)
    close(kwrite1)

 stop 'Data is saved'
 end program

!*****************************************************************
! For the two functions of two different differential equations
!*****************************************************************

      
      subroutine f(xl,y,dy)

      implicit none

      double precision xl
      double precision y(22),dy(22)
	
      dy(1) = y(2)
      dy(2) = (1.d0/(1.d0-xl*xl)*(2.d0*xl*y(2)-30.0d0*y(1))) !!** change

      return
      end

!*******************************************************
! RK4 method for solving differential equations
!*******************************************************

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
