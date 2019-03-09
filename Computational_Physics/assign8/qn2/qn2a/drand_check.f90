!bp: clear; f95 drand_check.f90 && ./a.out > drand_check.dat

      program drand_check
      implicit none

      integer :: npmax
      parameter (npmax=100000)
      
      integer :: iflag
      integer :: ku,nwork

      double precision :: drand
      double precision :: ax(npmax),awork(npmax)
      double precision :: sum, ucheck

      integer :: i,ii,np,jj

      integer :: kread, kwrite
      integer :: ku_array(3), np_array(3)

      data kread/5/, kwrite/6/
      
!     intialize
      ku_array = (/1, 3,7 /)
      np_array = (/100,10000,100000/)

      

      do 10 ii = 1,3
      ku = ku_array(ii)
      
      do 20 jj=1,3
      np = np_array(jj)

!-----------------------------------------------
!       Second built-in Random Number Generator 'drandm'
!          for iflag.ne.0  a seed it planted and
!          a new sequence of random number is generated
!          to get the next random number, iflag has to be zero


!        plant seed

      iflag = 1

      ax(1) = drand(iflag)

!        set flag to zero to fill ax(np)

      iflag=0

      do i=2,np
        ax(i)=drand(iflag)
      end do

!        check for uniformity

      do i=1,np
        awork(i)=ax(i)**ku
      end do

      sum = 0.d0

      do i=1,np
      sum = sum + awork(i)
      end do

      sum = sum/dfloat(np) 

      ucheck = sqrt(dfloat(np))*abs(sum - 1/(dfloat(ku)+1))

      write (kwrite,10000) 'ku=', ku, 'np=', np, 'sum =',sum,' ucheck = ', ucheck
      10000 format (2(a10,i6),2(a10,f10.6))
      
      20 end do
      write(kwrite,*)
      10 end do 

      stop 'data saved in output file' 
      end

