!Bhishan Poudel
!Oct 19, 2015

!bp: clear; f95 randtest.f90 && ./a.out

      program randtest
!
!        program to test built-in Random Number Generator
!           (SunStudio 12 intrinsic functions)
!
      implicit none

      integer :: np, nplot
      parameter (np=20, nplot=800)
!
      integer :: seed, iflag

      double precision :: ran
      double precision :: drand
      double precision :: a(np), p1(nplot), p2(nplot)

      integer :: i, irand

      integer :: kread, kwrite, kplot1, kplot2

      data kread/5/, kwrite/6/
      data kplot1/11/, kplot2/12/

!-----------------------------------------------
!       First built-in Random Number Generator 'rand'
!         
!        needs seed and call to srand to set up, then call
!        as often as numbers are needed

!        plant seed
      seed = 760013

!        or generate the seed with irand
!      iflag = 1
!      seed = irand(iflag)
    
      write (kwrite,*) ' seed = ',seed
      write (kwrite,*)


!        generate and print 20 random numbers

      do i=1,np
        a(i)= ran(seed)
      end do

      do i=1,np
        !write (kwrite,*) a(i)
      end do 

!      write (kwrite,*)
!      write (kwrite,*)


!        generate and print 800 random numbers for plot


      seed = 760056

      open (kplot1, File = 'plot1.dat', Status = 'Unknown')
      
      do i=1,nplot
        p1(i)= ran(seed)
      end do

      do i=1,nplot-1
        write (kplot1,10000) p1(i),p1(i+1)
      end do

10000 format(2d18.7)

!-----------------------------------------------
!       Second built-in Random Number Generator 'drand'
!          for iflag.ne.0  a seed it planted and
!          a new sequence of random number is generated
!          to get the next random number, iflag has to be zero

      iflag=1

!      write (kwrite,*)  'plant seed '
!      write (kwrite,*) ' iflag = ',iflag
!      write (kwrite,*)

!        plant seed

      a(1) = drand(iflag)

!        set flag to zero to get next random number

      iflag=0

      do i=2,np
        a(i)=drand(iflag)
      end do

      do i=1,np
        !write (kwrite,*) a(i)
      end do 


!        generate and print 800 random numbers for plot


      open (kplot2, File = 'plot2.dat', Status = 'Unknown')
      
      iflag=1

      p2(1) = drand(iflag)

      iflag=0

      do i=2,nplot
        p2(i)=drand(iflag)
      end do

      do i=1,nplot-1
        write (kplot2,10000) p2(i),p2(i+1)
      end do
     

      close (kplot1)
      close (kplot2) 
      stop 'data saved in plot1.dat & plot2.dat' 
      end

