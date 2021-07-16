!bp: clear; f90 splineNR.f90 && ./a.out


include 'spline.f90'
      program splineNR

!        Spline interpolation of given data set
!        Use icsccu and icsevu

      implicit none

      integer :: i, datmax
      integer :: kwrite,kread

      parameter (datmax=9)

      double precision ::  xin(datmax), yin(datmax)
      double precision ::  y2(datmax), yp1, ypn
      double precision ::  x, y

      double precision dummy,xstart,xend,xstep

      data kwrite/6/,kread/5/


!        output file
      open(kwrite, File = 'splineNRout.dat', Status = 'Unknown')

!        read intput data 
      
      open (kread, File = 'crossX2.dat', Status = 'old')

      do i=1,datmax 
      read (kread,10001) xin(i),yin(i),dummy
      end do

10001 format (1x,3f10.2)
10002 format (1x,2f10.2,1x,d15.5)


      xstart = xin(1)
      xend   = xin(datmax)
      xstep  = 2.d0


!        prepare interpolation

      yp1 = 0.d0
      ypn = 0.d0

      call spline (xin,yin,datmax,yp1,ypn,y2)

!        interpolate

       do x = xstart,xend,xstep

       call splint (xin,yin,y2,datmax,x,y)

       write (kwrite, 10002) x, y
       end do


      close (kread)
      close (kwrite)

      Stop 'data saved in splineNRout.dat'
      end program splineNR

