!Bhishan Poudel
!Oct 29,2015

!original program: int_10d.f90

!cmd: clear; gfortran -Wall hw9qn1.f90 && ./a.out
!cmd: clear; f95 hw9qn1.f90 && ./a.out

      program int10d

      implicit none

      integer :: nmax
      parameter (nmax=10000)
   
      integer :: i, j, k, p, n, ntrial, iflag
      parameter (ntrial=16)
      double precision :: rand, x, y, seed, ys, sum1,true

      integer :: kread, kwrite,kplot
      data kread/5/, kwrite/6/, kplot/8/


      open(kwrite, File='hw9qn1b.dat', Status='Unknown')
      open(kplot,  File='hw9qn1c.dat',  Status='Unknown')
      
      write (kwrite,*) '-------------------------------------------------'
      write(kwrite,*)'Sample size    Answer      true         error'
      write (kwrite,*) '-------------------------------------------------'
      true = 155.d0/6.d0

!        initialize
      do p=1,13

      sum1=0.d0

      n=2**p

   do k=1,ntrial
      
      iflag = k
      seed = rand(iflag)
      iflag=0

!        Outer loops determines the number of trials = accuracy
         
      y=0.d0



      do i=1, n

         x=0.d0

!
!        Add up ten random numbers x1 .. x10

        do  j=1,10
          x=x+rand(iflag)
        end do 

!        square and add up

        y=y+x*x

      end do

      ys=y/dble(n)
      
      sum1=sum1+ys

   end do

     write (kwrite,20000) n, sum1/float(ntrial),true, abs(sum1/float(ntrial)-155.d0/6.d0)
     
!    plot error versus 1/sqrt(n)
     write (kplot, 30000) 1.d0/sqrt(float(n)), abs(sum1/float(ntrial)-155.d0/6.d0)

20000 format(i10,4x,2(f8.4,4x),es10.2)
30000 format(2f10.4)
end do
 
      write (kwrite,*) '-------------------------------------------------'
      close(kwrite)
      
      stop 'data saved in output file'
      end
