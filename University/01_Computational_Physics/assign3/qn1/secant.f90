!cmd : clear; gfortran -Wall secant.f90 && ./a.out > secant2.dat
!cmd : clear; f90 secant.f90 && ./a.out > secant1.dat      for 0.1 and 1.0
!cmd : clear; f90 secant.f90 && ./a.out > secant2.dat      for 5.1 and 10.0


!***********************************************
!*        Main Program to calculate root       *
!***********************************************

      program secant_main
      implicit none

      double precision x
      double precision Y
      double precision xinitial,xfinal ! these are given to bisect1
      double precision root            ! root is obtained from bisect1
      double precision delta
      double precision tolerance   ! tolerance number

      double precision xx(50)  ! array for x values
      double precision fxx(50) ! array for Y(x) values
      integer iteration, itermax  
      
      integer kread, kwrite
      data kread/5/, kwrite/6/
      data itermax/30/
      
      ! print*, fxx(0) array starts with 1 in fortran
      ! NOTE: we should exclude zero, log(0) is not defined
      ! Y(0.1) = 1.60258509299  and Y(1) = -6 ,sign is changed,so root lies in between 0.1 and 1.0
      

      !FIRST ROOT
      xinitial = 0.1d0  
      xfinal   = 1.0d0
       
      ! SECOND ROOT
      !xinitial = 5.1d0  
      !xfinal   = 10.0d0        
      
      delta    = 0.1d0
      tolerance    = 1.d-10

      write (kwrite,*) '#Search Root via Newton-Raphson Secant Method : '
      write (kwrite,36) '#iteration', 'root(x)', 'Y(x)'
      36 format(A10,A15,A10)
 
!        do newton search
!        starting from xinitial

      xx(1)  = xinitial  ! xx first element is 0.1
      fxx(1) = Y(xx(1))  !fxx(1) is  Y(0.1)
      
      xx(2)  = xx(1) + delta      ! xx(2) is now (first element + delta) = 0.1 + 0.1 = 0.2
      fxx(2) = Y(xx(2))  ! Y(0.2) value

      iteration     = 1 ! iteration is iteration, will be updated just below.

 61   iteration=iteration+1    ! now, iteration = 2, this is like do loop
 
          if (iteration.gt.itermax) stop  ! itermax = max iterations = 30
      
		  ! when iteration < itermax this is calculated
		  ! xx(iteration) and and its function value
          write (kwrite,55) iteration-1, xx(iteration-1),fxx(iteration-1)   
     
          ! fxx(1), fxx(2), xx(1), xx(2),delta
          call secant (fxx(iteration-1),fxx(iteration),xx(iteration-1),xx(iteration),xx(iteration+1),delta) 
      
          fxx(iteration+1) = Y(xx(iteration+1))  ! fxx(2) updated

          ! if we get f(x) < tolerance we are done.
          if (abs(fxx(iteration+1)).lt.tolerance) then
          
              write(kwrite,*)
              write(kwrite,100) '#the value of Y(x) = ',fxx(iteration),'  at x = ',xx(iteration)
              100 format (A25, E20.5, A10, F10.5)

              ! if f(x) > tolerance we go back to label 61
              else 
              go to 61
           endif
           55 format(T2,I4, T15,F10.5, T25, E15.5)
 
      stop 
      end 

      
      
      

!***********************************************
!*        Function Y(x)                        *
!***********************************************
real*8 Function Y(x) 
  real*8 x
  Y = x*x - 7*x - LOG(x)
end

!***********************************************************************************
!* 	subroutine secant (f1       ,f2     ,a       ,b     ,c       ,delta)
!	call       secant (fxx(iteration-1),fxx(iteration),xx(iteration-1),xx(iteration),xx(iteration+1),delta)
!
!	Purpose		: This subroutine searches for the roots of an equation Y(m) = 0;
!            	
!	Arguments 	: delta, a,b and Y(a) = f1 and Y(b) = f2  and intent in 
!				  c is intent out.

!   Note 		: This program first, search where Y(m) changes sign, then
!                 search for the zero with the Newton-Raphson method.
!!**************************************************************************************
subroutine secant (f1,f2,a,b,c,delta)
      implicit none

      double precision,intent(in)   :: f1,f2,a,b,delta  ! f1 is Y(a) 
      double precision,intent(out)  :: c
      double precision              :: sign1,sign2      ! local variables
      integer,save                  :: flag = 0         ! saving value to ensure that variables used within
       							                        !a procedure preserve their values between
       							                        ! successive calls to the procedure
       
!     looking for change in sign

      if (flag.eq.0) then
          sign1=sign(1.d0,f1)  ! sign1 = + 1.000 if f1 is +ve and so on
          sign2=sign(1.d0,f2)  ! note: arguments must be of same type for transfer of sign function
          
 
      
          if (sign1.eq.sign2) then ! flag = 0 and f1,f2 has same sign i.e. Y (a) Y (b) > 0
              c=b+delta          ! c=b+delta is returned
              return
      
          else               ! when flag=0, and f1,f3 has opposite signs i.e. Y (a) Y (b) < 0, we use secant method
              c= (f2 *a - f1*b) / (f2-f1)
              flag=1
          endif
      
 
        !Secant method
    else if (flag.eq.1) then 
        c= (f2*a - f1*b) / (f2-f1)
 
    endif
 
    return
end
!   from: Morten HJ fall 2012 page 105
!   ===================================
!	A variation of the secant method is the so-called false
!	position method (regula falsi from Latin) where the interval [a,b] 
!	is chosen so that Y (a) Y (b) < 0 , else there is no solution.

!	we determine c by the point where a straight line
!	from the point (a, Y (a)) to (b, Y (b)) crosses the x − axis , that is
!	c = Y(b) a − Y(a) b    = (f2 *a - f1*b) / (f2-f1)
!	    ________________
!		Y(b) − Y(a)
!
!
!   Newton Raphson equation 4.6 
!	x_(n+1) = x_n − Y(x_n) / Y′ (x_n)
!
!   Secant method eq 4.37
!
!   x_(n+1) = xn - Y(xn) * (xn - xn1))
!             	    ===================  eq 4.37
!                   Y(xn) -Y(xn-1)
!
!           = Y(xn)x(n-1) - Y(xn-1)xn
!             ========================   eq 4.38
!             Y(xn) - Y(xn-1)
!
!
!      
      









