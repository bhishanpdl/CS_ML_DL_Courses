!Template: integ3.f90, gaulag.f90 and gauleg.f90 were provided by Dr. Elster


! Author: Bhishan Poudel
! Date  : Sep 27,2015


!cmd: clear; gfortran -Wall hw5qn3a.f90 && ./a.out
!cmd: clear; f90 hw5qn3a.f90 && ./a.out


! find integral of x^3/(e^x-1) from 0 to infinity
! I = 0 to infinity x^alpha exp(-x) f(x) dx
! here, alpha = 3, and f(x) = 1/ (1- exp(-x))

include 'gaulag.f90'




      program hw5qn3a
      implicit none

      integer :: ng,nsxx,ngp    !ng is no. of gauss points, ngp is paramter (max n-points)
      integer :: kread, kwrite
      integer :: i
      double precision :: pi,sxx,wxx,u,w
      double precision :: sum3,error3,exact,alpha
      character(len=50):: filename
      
      
 
!        integration points

      parameter (nsxx=2000)
      parameter (ngp=64)
      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
 
      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngp),w(ngp)
      
      !exact value  Pi^4/15
      exact = pi**4.d0/15.d0
      exact = 6.49393940227
      
      !filename='hw5qn3a.dat'        ! *** CHANGE filename,title,write,and format
      filename='hw5qn3aplot.dat'
      
     open(kwrite,file=filename,status='Unknown') 
     
      
      
     !write (kwrite,10000) '#n-point','exact', 'Gauss-Laguerre','absolute error'   ! *** CHANGE     
     write (kwrite,10000) '#log10(N)','exact', 'Gauss-Laguerre','log10(E)'                  
     write (kwrite,*)
     
     
       do 41 ng=2,10,2
 
!
!
!        gauss integration
!
!        get gauss points and weights
!
! SUBROUTINE gaulag (x,w,n,alf)

        ! for given function to integrate alpha = 3.d0
        ! generalized gauss laguerre quadrature
        ! integral 0 to infinity  x^alpha  e^-x   f(x) dx
        ! integral 0 to infinity  x^3      e^-x   1/ (1- e^-x )
        alpha = 3.d0
        call gaulag (u,w,ng,alpha)
 
 
 
      do i=1,ng
      sxx(i)=u(i)
      wxx(i)=w(i)
      end do
!
!
!       integrate
!
      sum3=0.d0
      do  i=1,ng
      sum3=sum3 + 1/(1- exp(-sxx(i)) )*wxx(i)  !f(x)
      error3= abs(sum3-exact)/ abs(exact)
      end do
 
 
      !write (kwrite,20000) ng,exact,sum3,error3  ! *** CHANGE
      
      
      if(error3 /=0.0000) then                                          ! *** CHANGE
      write (kwrite,20000) log10(dfloat(ng)),exact,sum3,log10(error3)
      end if
 
!      10000 format(A10,  T12,A14,  T30,A14,  T45,A14)
!      20000 format(i10,1x,f14.6,f14.6,1x,E14.6)
!      
      
      10000 format(A10,  T12,A14,  T30,A14,  T45,A14)
      20000 format(F10.5,1x,f14.6,f14.6,1x,E14.6)  
      
      41 end do
      close(kwrite)
      
      print*, 'filename is ', filename
      stop 'data is saved in filename'     
      
      end program



