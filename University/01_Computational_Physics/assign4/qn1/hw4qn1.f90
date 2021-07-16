!cmd: clear; gfortran hw4qn1.f90 && ./a.out
!cmd: clear; f90 hw4qn1.f90 && ./a.out

! this program calculates forward,central,and extrapolated differentiation of given functions


      program hw4qn1
      implicit none
      
!      real,parameter    :: x = 0.1d0  ! **** CHANGE value of x and filename
!      real,parameter    :: tol = 1d-7  ! tolerance
!      real    :: fcos,fexp,fsq, h, hd2, hd4
!      real    :: result(9)
!      integer :: kwrite = 6
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      
      double precision,parameter :: x = 30.0d0   ! **** CHANGE value of x and filename
      double precision,parameter :: tol = 1d-14  ! tolerance *** CHANGE
      double precision           :: fcos,fexp,fsq, h, hd2, hd4, hstep
      double precision           :: result(9)
      integer                    :: kwrite

      
!!!!!!!!!!!!for cosine x i.e. fcos !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      open(1000, File = 'cos30dp.dat', Status = 'Unknown')  ! **** CHANGE
      
      h = 40.0d0       ! h can not be made parameter since h = h*0.1 is changing

      
      
      write(1000,1000)   "#h", "forward_diff",  "rel_error", &
                           &     " central_diff", "rel_error", &
                           &     "extrapol_diff", "rel_error"
      
      do while (h.gt.tol)
     
      hd2 = h*0.5d0  ! h/2
      hd4 = h*0.25d0 ! h/4
      result(1) = (fcos(x+h) - fcos(x))/h
      result(2) = (fcos(x+hd2) - fcos(x-hd2))/h
      result(3) = (8.d0 *(fcos(x+hd4)-fcos(x-hd4)) - &
     &                 (fcos(x+hd2)-fcos(x-hd2)))/(3.d0*h)

      ! relative_error    = |(true - observed) / true |
      ! here, d/dx (cosx) = -sinx
      
      write (1000,1001) h,   result(1), abs((result(1)+sin(x))/sin(x)), &
                          &  result(2), abs((result(2)+sin(x))/sin(x)), &
	                      &  result(3), abs((result(3)+sin(x))/sin(x))
	
      h = h*0.8d0 ! decreasing value of step size h
      
      end do

     1000 format (7(A16))
     1001 format(7(E16.6))

      close(1000)

!!!!!!!!!!!!for exp x i.e. fexp !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      open(2000, File = 'exp30dp.dat', Status = 'Unknown')  ! **** CHANGE
            
      h = 40.d0

      
      write(2000,2000)   "#h", "forward_diff", "rel_error", " central_diff", "rel_error", "extrapol_diff", "rel_error"
      
      do while (h.gt.tol)
     

		  hd2 = h*0.5d0  ! h/2
          hd4 = h*0.25d0 ! h/4
		  
		  result(4) = (fexp(x+h) - fexp(x))/h
		  result(5) = (fexp(x+hd2) - fexp(x-hd2))/h
		  result(6) = (8.d0 *(fexp(x+hd4)-fexp(x-hd4)) -   (fexp(x+hd2)-fexp(x-hd2)))/(3.d0*h)


      !error = |(true - observed) / true |
      ! here, d/dx (exp x) = exp x
      
      write (2000,2001) h, result(4), abs((result(4)-exp(x))/(exp(x))),result(5), abs((result(5)-exp(x))/(exp(x))), &
	& result(6), abs((result(6)-exp(x))/(exp(x)))
	
      h = h*0.8d0
      
      end do
      
     2000 format (7(A16))
     2001 format(7(E16.6))
 

      close(2000)
      
!!!!!!!!!!!!for sqrt x i.e. fsq !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      open(3000, File = 'sq30dp.dat', Status = 'Unknown')  ! **** CHANGE
      
      h = 0.1d0 ! should be greater than 0

      
      
      write(3000,3000) "#h", "forward_diff", "rel_error", " central_diff", "rel_error", "extrapol_diff", "rel_error"
      
      do while (h.gt.tol)
     

		  hd2 = h*0.5d0  ! h/2
          hd4 = h*0.25d0 ! h/4
          
		  result(7) = (fsq(x+h)   - fsq(x))     /h
		  result(8) = (fsq(x+hd2) - fsq(x-hd2)) /h
		  result(9) = (8.d0 *(fsq(x+hd4)-fsq(x-hd4)) -   (fsq(x+hd2)-fsq(x-hd2)))/(3.d0*h)


      !error = |(true - observed) / true |
      ! here, d/dx (sqrt x) = 1/(2* sqrt(x)) = 0.5/sqrt(x)
      
	
	write (3000,3001) h, result(7), abs((result(7)-.5d0/sqrt(x))/(.5d0/sqrt(x))),result(8),&
	& abs((result(2)-.5d0/sqrt(x))/(.5d0/sqrt(x))),  result(9), abs((result(9)-.5d0/sqrt(x))/(.5d0/sqrt(x)))
	
      h = h*0.8d0
      
      end do

     3000 format (7(A16))
     3001 format(7(E16.6))
 

      close(3000)
!-------------------------------------------------------------------------------     
      
      end program hw4qn1
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

      double precision function fcos(x)
!      real function fcos(x)  !fcos is function and x is argument

      implicit none

      double precision ::  x
!     real ::  x

      fcos = cos(x)

      return
      end
!-------------------------------------------------------------------------------
      double precision function fexp(x)
!      real function fexp(x)

      implicit none

      double precision ::  x
!      real ::  x

      fexp = exp(x)

      return
      end
!-------------------------------------------------------------------------------

      double precision function fsq(x)
!      real function fsq(x)

      implicit none

      double precision ::  x
!      real ::  x

      fsq = sqrt(x)

      return
      end
!-------------------------------------------------------------------------------
