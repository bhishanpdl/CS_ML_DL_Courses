!Template: integ3.f90, gaulag.f90 and gauleg.f90 were provided by Dr. Elster


! Author: Bhishan Poudel
! Date  : Sep 27,2015


!cmd: clear; gfortran -Wall hw5qn3b.f90 && ./a.out
!cmd: clear; f90 hw5qn3b.f90 && ./a.out

include 'gauleg.f90'


      program gaussleg
      implicit none

      integer :: nxmt,nxms,ng,nsxx,ngmax !ng is no. of gauss points 
      integer :: kread, kwrite
      integer :: i
      character(len=50):: filename
      double precision :: pi  ! parameter
      double precision :: lower,upper,sum,exact,error3
      double precision :: sxx,wxx,u,w ! 1d arrays sxx(i) and wxx(i) will be obtained from u(i) and w(i)
                                      ! sxx is like x and wxx is like w(x), we shall again loop from 1 to ng
      
 
!        integration points
      parameter (nxmt=30,nxms=30) 
      parameter (nsxx=2000)
      parameter (ngmax=64) ! max. no. of gauss points
      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
 
      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngmax),w(ngmax)
      
      ! lower and upper limits
      lower = 0.d0
      upper = 1.d0  ! limit 0 to infinity is mapped to 0 to 1 by substituing x = tan(pi*y/2)
      
      !exact value  Pi^4/15
      exact = pi**4.d0/15.d0
      exact = 6.49393940227d0
      
      !filename='hw5qn3b.dat'   ! *** CHANGE filename,title,write,format
      filename='hw5qn3bplot.dat'
      
      open(kwrite,file=filename,status='unknown')
      
           
      !write (kwrite,10000) '#n-point','exact', 'Gauss-Legendre','absolute error'  ! ** CHANGE
      write (kwrite,10000) '#log10(N)','exact', 'Gauss-Legendre','log10(E)'             
 
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!        gauss integration
!
!        get gauss points and weights
!        inputs: lower,uppper and ng
!        out   : u(i) and w(i) arrays, which we shall again loop upto i = 1 to ng
!
!subroutine gauleg (x1,x2,x,w,n)  ! we have included this subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       do 45 ng=2,10,2
       call gauleg (lower,upper,u,w,ng) ! limits are mapped to 0 to 1
                                        ! sxx(i), wxx(i) will be obtained from u(i) and w(i)
 
 
      
      do i=1,ng
      sxx(i)=u(i)
      wxx(i)=w(i)
      end do
!
!
!       integrate
!
      sum=0.d0
      do  i=1,ng
      sum=sum + ((pi/2.d0)*tan((pi/2.d0)*sxx(i))**3*wxx(i))/ &
           &    ((cos((pi/2.d0)*sxx(i))**2)*(exp(tan((pi/2)*sxx(i)))-1))
      end do
      
      
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !!!!!!! Writing the outputs
      
      error3= abs(sum-exact)/ abs(exact)
 
      !write (kwrite,20000) ng,exact,sum,error3  ! *** CHANGE
      
      if(error3 /=0.0000) then
      write (kwrite,20000) log10(dfloat(ng)),exact,sum,log10(error3)
      end if
      
      
      45 end do
      
      
 
      !10000 format(A10,  T12,A14,  T30,A14,  T45,A14) ! *** CHANGE
      !20000 format(i10,1x,f14.6,f14.6,1x,E14.6)
      
      10000 format(A10,  T12,A14,  T30,A14,  T45,A14)
      20000 format(F10.5,1x,f14.6,f14.6,1x,E14.6) 
 
      close(kwrite)
      print*, 'filename is ',filename
      stop 'data saved in filename' 
      end



