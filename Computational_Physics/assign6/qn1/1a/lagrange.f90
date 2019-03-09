!cmd: clear; gfortran -Wall lagrange.f90 && ./a.out


      program lagrange

!        Langrange interpolation of given data set

      implicit none

      integer :: i, e, datmax
      integer :: kwrite,kread

      parameter (datmax=9)

      double precision :: interl, x, xin(datmax), yin(datmax)

      double precision dummy,xstart,xend,xstep

      data kwrite/6/,kread/5/

      e = 9


!        output file
      open(kwrite, File = 'lagrangeout.dat', Status = 'Unknown')

!        read intput data 
      
      open (kread, File = 'crossX2.dat', Status = 'old')

      do i=1,datmax 
      read (kread,10001) xin(i),yin(i),dummy
      end do

10001 format (1x,3f10.2)

      xstart = xin(1)
      xend   = xin(9)
      xstep  = 5.d0

      do x = xstart,xend,xstep

      write (kwrite, 10001) x, interl (xin, yin, e, x)
      end do

      close (kread)
      close (kwrite)

      Stop 'data saved in lagrangeout.dat'
      end program lagrange

! ------------------------------------------------------------

      double precision function interl (xin, yin, e, x)

!        evaluate interpolation function 
!        input:  xin, yin  = x and y values of the function
!                e  =  degree of interpolating polynomial
!                x  =  x-value at which inter(x) is calculated
  
      implicit none
      integer :: i, j, e, ndmax

      parameter (ndmax=9)

      double precision :: lambda(9), x
      double precision :: xin(ndmax), yin(ndmax)

      interl = 0.d0

      do i = 1, e
      lambda(i) = 1.d0
       do j = 1, e
        if (i.ne.j) then
        lambda(i) = lambda(i) * ((x - xin(j))/(xin(i) - xin(j)))  ! landau 1E eqn 5.4
        endif
       end do
      interl = interl + (yin(i) * lambda(i)) ! landau 1E eqn 5.3
      end do

      return
      end
