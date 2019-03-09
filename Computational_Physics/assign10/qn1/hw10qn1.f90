!Bhishan Poudel
!Nov 2, 2015

!cmd: clear; f95 hw10qn1.f90 && ./a.out; rm -f a.out *~

!cmd: clear; gfortran hw10qn1.f90 && ./a.out

!Topic: Metropolis Algorithm to sample 2x

!Extracting first 1000 xvalues: sed -n '1,1000 p' xvalues.dat > xvalues1.dat     (add 3 to last value)
!                               sed -n '1003,2002 p' xvalues.dat > xvalues2.dat  
!                               sed -n '2005,3004 p' xvalues.dat > xvalues3.dat
!                               sed -n '3007,4006 p' xvalues.dat > xvalues4.dat

program hw10qn1
implicit none

integer,parameter :: nwalk=1000, nstep=100, kwrite=6, kwrite1=8,kwrite2=10,kwrite3=15
integer           :: iflag, i, j, k
double precision  :: delta, q
double precision  :: x(nwalk), g(nwalk), x_next, g_next, xmin, xmax
double precision  :: drand, seed
double precision  :: ibin(21), exact(21)

delta=0.1d0

! plant seed
iflag=1
seed=drand(iflag)
iflag=0

open(kwrite, File='hw10qn1.dat', Status='Unknown')
open(kwrite2,File='xvalues.dat', Status='Unknown')

open(kwrite1,File='initial.dat', Status='Unknown')
open(kwrite3,File='initialbinvalues.dat',Status='Unknown')

!   printing initial distribution
    write(kwrite1,*) 'Initial distribution'
    write(kwrite1,*)

!   Initialize loop
    do 1 i=1,nwalk
  
        x(i)=drand(iflag)      ! random number
        g(i)=2.d0*x(i)         ! our function is 2x
    1   write(kwrite1,*) g(i)


!   using 20 bins
    do 4 j=1,20
    4   ibin(j)=0.d0

!   using 1000 walkers
    do 5 i=1,nwalk
        j=20.d0*x(i)+1.d0
    5   ibin(j)=ibin(j)+1.d0

!   find the exact value x_max^2 - x_min^2
    do 6 j=1,20
        xmin=(j-1)*0.05d0
        xmax=j*0.05d0
    6   exact(j)=nwalk*(xmax**2-xmin**2)

!   printing bin values
    write(kwrite3,*) 'Initial Bin values'
    write(kwrite3,*) 'j         ibin(j)         exact(j)'

    do j=1,20
        write(kwrite3,10000) j,ibin(j),exact(j)
    end do
    10000  format (i2,2x,f15.8,2x,f15.8)
    write(kwrite3,*)
    write(kwrite3,*)
    !write(kwrite,*) '--------------------------------------------------------'


!   taking nsteps    
    do 2 k=1,nstep
         
        do 3 i=1,nwalk

            !setting new values to x (here, delta = 0.1 )
            !xvalues are between 0 and 1
            !immediate next x-values is within +-0.1 of current x value 
            x_next=x(i)+delta*(2.d0*drand(iflag)-1.d0)
            
            !taking x values between 0 and 1
            if (x_next.gt.1.d0) goto 3
            if (x_next.lt.0.d0) goto 3
              
            g_next=2.d0*x_next
            q=g_next/g(i)

            !rejection algorithm
            if (q.le.drand(iflag)) goto 3 

            x(i)=x_next
            g(i)=g_next

        3   continue

    ! Bin every 25 steps
    if (mod(k,25).ne.0) goto 2

    !initialize ibin
    do 14 j=1,20
    14   ibin(j)=0.d0

    !printing step number and x values(between 0 and 1)
    !write(kwrite2,*) 'step number is', k
    !write(kwrite2,*) 'x values'

    do 15 i=1,nwalk
         j=20.d0*x(i)+1.d0
         write(kwrite2,*) x(i)
    15 ibin(j)=ibin(j)+1.d0
    write(kwrite2,*)
    write(kwrite2,*)
    !write(kwrite2,*) '--------------------------------------------------------'
    
    
!   printing bin values
    !write(kwrite,*) 'j         ibin(j)         exact(j)'
    do j=1,20
         write(kwrite,10000) j,ibin(j),exact(j)
    end do
    write(kwrite,*)
    write(kwrite,*)
    !write(kwrite,*) '--------------------------------------------------------'    


2  continue            

 close(kwrite)
 close(kwrite1)
 close(kwrite2)
 close(kwrite3)
 stop 'data saved in hw10qn1.dat and initial.dat,initialbinvalues.dat and xvalues.dat'

end program 
