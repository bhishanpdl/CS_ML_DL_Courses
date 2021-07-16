!bp: clear; f90 splinehuber.f90 && ./a.out


include 'cubhermdh.f90'
      program splinecbh

!        Spline interpolation of given data set
!        Use icsccu and icsevu

      implicit none

      integer :: i, datmax
      parameter (datmax=6)               ! ** CHANGE (total rows in inputfile 21,11,6)
      integer :: index(4), i1,i2,i3,i4
      integer :: kwrite,kread
      double precision ::  xin(datmax), yin(datmax)
      double precision ::  spl(4)
      double precision ::  x, y, summ,f,error

      double precision dummy,xstart,xend,xstep
      character(len=50) :: outputfile,inputfile
      
      

      data kwrite/7/,kread/5/
      
      
      outputfile = 'huber40.dat'         !*** CHANGE
      inputfile  = 'experimental40.dat'  ! *** CHANGE
      
      ! output file
      open(kwrite, File =outputfile, Status = 'Unknown')

      !read intput data      
      open (kread, File = inputfile, Status = 'old')
      
      ! write title for output file
      write(kwrite,10000) '#E','huber','exp','error'

      do i=1,datmax 
      read (kread,10001) xin(i),yin(i)
      end do
      
10000 format (4A10)
10001 format (1x,3f10.2)
10002 format (4F10.2)


      xstart = xin(1)
      xend   = xin(datmax)
      xstep  = 5.d0




!        interpolate

       do x = xstart,xend,xstep

       call cubhermdh (xin,datmax,x,1,spl,index)
!       summ =0.d0 

        i1=index(1)
        i2=index(2)
        i3=index(3)
        i4=index(4)
   
        y = spl(1)*yin(i1)+spl(2)*yin(i2)+spl(3)*yin(i3)+spl(4)*yin(i4)
        f = (68658.34d0)/( (x-75)**2 + 822.26)
        error = abs(y-f)

       write (kwrite, 10002) x, y,f,error
       end do


      close (kread)
      close (kwrite)

      print*, 'output data saved in  ', outputfile
      end program splinecbh

