!bp: clear; f90 splinecbh.f90 && ./a.out

include 'cubhermdh.f90'
      program splinecbh

!        Spline interpolation of given data set
!        Use icsccu and icsevu

      implicit none

      integer :: i, datmax
      integer :: index(4), i1,i2,i3,i4
      integer :: kwrite,kread

      parameter (datmax=9)

      double precision ::  xin(datmax), yin(datmax)
      double precision ::  spl(4)
      double precision ::  x, y, summ, f, error

      double precision dummy,xstart,xend,xstep
      character(len=50) :: outputfile

      data kwrite/6/,kread/5/


      !output file
      outputfile='spline.dat'
      open(kwrite, File = outputfile, Status = 'Unknown')

      !read intput data 
      
      open (kread, File = 'crossX2.dat', Status = 'old')

      do i=1,datmax 
      read (kread,10001) xin(i),yin(i),dummy
      end do

10001 format (1x,3f10.2)
10002 format (1x,2f10.2,1x,d15.5)


      xstart = xin(1)
      xend   = xin(datmax)
      
      xstep  = 0.01d0  !*** CHANGE 0.01,10,20


!        interpolate

       do x = xstart,xend,xstep

       call cubhermdh (xin,datmax,x,1,spl,index)
!       summ =0.d0 

        i1=index(1)
        i2=index(2)
        i3=index(3)
        i4=index(4)
   
        y = spl(1)*yin(i1)+spl(2)*yin(i2)+spl(3)*yin(i3)+spl(4)*yin(i4)
        f = (68658.72)/( (x-75)**2 + 822.26)      
        error = abs(f-y)/abs(f)
       write (kwrite, 10002) x, y
       end do


      close (kread)
      close (kwrite)

      print*, 'data saved in output file  ', outputfile
      end program splinecbh

