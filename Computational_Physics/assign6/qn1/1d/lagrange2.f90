!bp: clear; gfortran -Wall lagrange2.f90 && ./a.out
!bp: clear; f90 lagrange2.f90 && ./a.out 
 
include 'polint.f90' 
      program lagrange2

!        Langrange interpolation of given data set
!        Use Numerical Recipe routine polint

      implicit none

      integer :: i, j, datmax, pdeg
      integer :: kwrite,kread

      parameter (datmax=9)

      double precision ::  xin(datmax), yin(datmax)
      double precision ::  x, y, dy

      double precision dummy,xstart,xend,xstep

      data kwrite/7/,kread/5/

      pdeg=3


!        output file
      open(kwrite, File = 'polintout.dat', Status = 'Unknown')

!        read intput data 
      
      open (kread, File = 'crossX2.dat', Status = 'old')

      do i=1,datmax 
      read (kread,10001) xin(i),yin(i),dummy
      end do

10001 format (1x,3f10.2)
10002 format (1x,2f10.2,1x,d15.5)


      do j=1,7,2    ! we have 9 data when j=1: j+2 =3 so 1,2,3 values are used
                    ! when j=7, j+2 = 9 : we will use 7,8,9 indexed values

        xstart = xin(j)  ! when j=1, this is first x value ( when j changes, x values changes)
        xend   = xin(j+2) ! when j=1, this is third x value
        xstep  = 5.d0

        do x = xstart,xend,xstep

       !bp: for lagrange1, input was xin, which was 9 numbers array.
       
       !print*, j, x, xin(j:j+2) ! bp: we will get 3 values for xin
       
       
       ! xin, yin,pdeg,x are inputs
!       !bp: y,dy are outputs



!             subroutine polint (xa,ya,n,x,y,dy)

!!       given arrays xa and ya, each of length n, and given value of x,
!!       this routine returns a value y, and an error estimate dy.
!!       If P(x) is the polynomial of degree N-1 sich that 
!!       P(xa_i)=ya_i, i=1,...,n, then the returned value is y=P(x).
!!

! lagrange1 uses interl routine for which xin takes all 9 elements of the array
! lagrange2 uses polint routine which takes only 3 points of array at one time

       call polint (xin(j:j+2),yin(j:j+2),pdeg,x,y,dy)  !bp: we are taking 3 points each time

       write (kwrite, 10002) x, y, dy
       !print*, j,x,y,dy
       end do

      end do
      !bp: when j=1, polint was called with xin 1,2,3
      !    when j=2, polint was called with xin 2,3,4

      close (kread)
      close (kwrite)

      Stop 'data saved in polintout.dat'
      end program lagrange2

