!Bhishan Poudel
!Oct 29,2015

!original program: int_10d.f90

!cmd: clear; gfortran -Wall hw9qn2a.f90 && ./a.out

!cmd: clear; f95 hw9qn2a.f90 && ./a.out; rm -f *~

!note: for gfortran use ran, for f95 use ddrand

      program int10d

      implicit none

      integer :: nmax
      parameter (nmax=10000)
   
      integer :: i, k, p, n, ntrial, iflag
      parameter (ntrial=16)
      double precision :: drand, y, seed, ys, sum1,x1,x2,x3

      integer :: kread, kwrite,kplot
      data kread/5/, kwrite/6/, kplot/8/


      open(kwrite, File='hw9qn2a.dat', Status='Unknown')
      
      write(kwrite,*) '#1/sqrt(n)', '      result', '        n'


!        initialize
      do p=1,13

      sum1=0.d0

      n=2**p

   do k=1,ntrial
      
      iflag = k
      seed  = drand(iflag)
      iflag = 0

!        Outer loops determines the number of trials = accuracy
         
      y=0.d0



      do i=1, n

         x1 = 0.d0
         x2 = 0.d0
         x3 = 0.d0

      
      x1 = x1 + drand(iflag)
      x2 = x2 + drand(iflag)
      x3 = x3 + drand(iflag)
      
      x1 = 1.d0/(1.d0 + x1 * x1)
      x2 = x2 * exp(-x2 * x2)
      x3 = exp(-x3) / sqrt(x3)
      
      y = y+ x1 * x2 * x3
      

      end do

      ys=y/dble(n)
      
      sum1=sum1+ys

   end do

     write (kwrite,20000) 1.d0/sqrt(dble(n)), sum1/float(ntrial),n

20000 format(f10.4,4x,f10.2,i8)

end do
 
      close(kwrite)
      
      stop 'data saved in output file'
      end
      
!time ./a.out 
!real	0m0.265s
!user	0m0.051s
!sys	0m0.011s

