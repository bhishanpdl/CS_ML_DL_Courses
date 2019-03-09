!bp: clear; f90 splinerecipe.f90 && ./a.out


include 'spline.f90'
      program splineNR

!        Spline interpolation of given data set
!        Use icsccu and icsevu

      implicit none

      integer :: i, datmax
      integer :: kwrite,kread

      parameter (datmax=6)  ! ** CHANGE (21,11,6)

      double precision ::  xin(datmax), yin(datmax)
      double precision ::  y2(datmax), yp1, ypn
      double precision ::  x, y,f,error

      double precision dummy,xstart,xend,xstep
      character(len=50) :: outputfile,inputfile

      data kwrite/7/,kread/5/
      
      outputfile = 'recipe40.dat'         !*** CHANGE
      inputfile  = 'experimental40.dat'   ! *** CHANGE


      ! output file
      open(kwrite, File =outputfile , Status = 'Unknown')

      !read intput data      
      open (kread, File =inputfile, Status = 'old')
      
      ! write title for output file
      write(kwrite,10000) '#E','recipe','exp','error'

      do i=1,datmax 
      read (kread,10001) xin(i),yin(i),dummy
      end do

10000 format (4A10)
10001 format (1x,3f10.2)
10002 format (4F10.2)


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
       
       f = (68658.34d0)/( (x-75)**2 + 822.26)
       error = abs(y-f)

       write (kwrite, 10002) x, y,f,error
       end do


      close (kread)
      close (kwrite)

      print*, 'output data saved in  ', outputfile
      end program splineNR

