! clear; f90 hw4qn2.f90 && ./a.out 

!        calculate forward, central, and extrapolated differentiation
!        of a given function



      program cosine_diff

      implicit none

!      double precision :: fcos, h, hd2, hd4, x
!      double precision :: result(3)

      real    :: fcos, h, hd2, hd4, x
      real    :: result(3), errfor, errcent
      real,parameter :: tolerance = 1d-6
      integer :: kwrite,i

      data kwrite/6/


      x  = real(30.0)  ! ***** CHANGE X = 0.1d0, 1.0d0, 30.0d0  also change outputfile
      
      open(unit=kwrite,file='hw4qn2_30.dat',status='unknown')

      write (kwrite,100) '#x=',x
      write (kwrite,101) '    #h','3-point','Error','5-point','Error'
      
      h = 0.314d0
      
      do while(h.gt.tolerance) ! upto machine precision for single precision
    
      hd2 = h* real(0.5)
      hd4 = h* real(0.25)
      
      
      ! grouping similar numbers together to reduce roundoff error
      result(1) = (fcos(x-h) + fcos(x+h))/h**2 - (2*fcos(x) )/h**2
      
      result(2) = (   -fcos(x-2*h)   - fcos(x+2*h) &
                    & + 16*fcos(x-h) + 16*fcos(x+h) &
                    & - 30*fcos(x)    ) &
                  & /  (12*h**2)


      !     error = abs(true - calc)/ abs(true)
      !     d/dx cosx = -sinx
      !     d2/dx2 cosx = -cosx
    
      errfor = abs(-cos(x) - result(1))/abs(-cos(x))     
      errcent = abs(-cos(x) - result(2))/abs(-cos(x)) 


      write (kwrite,102) h, result(1), errfor, result(2), errcent
      h = h*0.95 
      end do
      close(kwrite)
          

100 format(T4,A, T8, F6.2)      
101 format( T10,A,  T24,A10,  T34,A10,   T54,A10,  T69,A10,  T86,A10)
102 format(4x,E14.4,2x,E14.4,2x,E14.6,2x,E14.2,2x,E14.2) 
      close(kwrite)

      stop 
      
      end program cosine_diff
!--------------------------------------------------------------------------

!      double precision function fcos(x)
      real function fcos(x)

      implicit none

!      double precision ::  x
      real ::  x

      fcos = cos(x)

      return
      end
!--------------------------------------------------------------------------

!      double precision function fexp(x)
      real function fexp(x)

      implicit none

!      double precision ::  x
      real ::  x

      fexp = exp(x)

      return
      end
!--------------------------------------------------------------------------

!      double precision function fsq(x)
      real function fsq(x)

      implicit none

!      double precision ::  x
      real ::  x

      fsq = sqrt(x)

      return
      end
