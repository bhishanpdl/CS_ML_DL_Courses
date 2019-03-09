! Bhishan Poudel
! Dec 5, 2015 Sat

! clear; f95 legendre_compare.f90 && ./a.out; rm *~ a.out

! This program checks the accuracy of Pn(x) obtained from the function pl(x,n)
! using orthogonality condition. It uses subroutine 'gauleg' to integrate the
! required integrand and takes 30 number of gauss points.


program legendre_compare


      implicit none
      external f1,pl

      integer :: ng,nsxx,ngmax
      integer :: kread, kwrite
      integer :: i,n
      double precision :: pi,sxx,wxx,u,w,lower,upper ! pi is defined below
      double precision :: sum1,orthogonality
      double precision :: f1

      ! parameters
      parameter (nsxx=9000,ngmax=9000)  ! maximum no. of iterations

      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngmax),w(ngmax)

      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



      open(unit=kwrite,file='legendre_compare.dat',status='unknown')
      write(kwrite,10000) '#n', 'integral of Pn(x)^2 from -1 to 1', 'value from orthogonality 2/(2n+1)'
      
      do n = 0,8
      orthogonality = 2.d0 / (2.d0*real(n)+1.d0)
      
      ng= 30         ! no. of gauss points
   
      
      lower = 0.d0   ! *** CHANGE
      upper = 1.d0
      
      !!!!!!!!!!!!!!!!! Calling subroutine !!!!!!!!!!!!
      !    subroutine gauleg (x1,x2,x,w,n)
      !    x(n) and w(n) are 1d arrays of n elements output
      !    x1,x2,n are inputs
      
      call gauleg (lower,upper,u,w,ng)
 
      do i=1,ng
      sxx(i)=u(i)
      wxx(i)=w(i)
      end do

      ! integrate

      sum1=0.d0
      do i=1,ng 
      sum1 = sum1 + f1(sxx(i),n) * wxx(i)
      end do
      
      
      
      write(kwrite,20000) n,2.d0*sum1,orthogonality
      
      end do

      10000 format(A4, 2A40)
      20000 format(I4,2f40.7) 

      
      close(kwrite)      
      stop'data saved in legendre_compare.dat' 
      end program
      

      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     function pl(x,n)
!======================================
! calculates Legendre polynomials Pn(x)
! using the recurrence relation
! if n > 100 the function retuns 0.0
!======================================
double precision pl
double precision x
double precision pln(0:n)
integer n, k

pln(0) = 1.d0
pln(1) = x

if (n <= 1) then
  pl = pln(n)
  else
  do k=1,n-1
    pln(k+1) = ((2.d0*k+1.d0)*x*pln(k) - real(k)*pln(k-1))/(real(k+1))
  end do
  pl = pln(n)
end if
return
end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
double precision function f1(x,n)
       external pl
       implicit none
       integer n
       double precision x,pl

       f1 = pl(x,n) * pl(x,n)
 
       return
       end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine gauleg (x1,x2,x,w,n) 
!
!        calculate gauss points for gauss-legendre integration
!                 (numerical recipes)  
!
      implicit double precision (a-h,o-z) 
!
      parameter (eps=3.d-14) 
      dimension x(n),w(n) ! x(n) and w(n) are 1d arrays of n elements
!
      m=(n+1)/2 
      xm=0.5d0*(x2+x1) 
      xl=0.5d0*(x2-x1) 

      do 12 i=1,m 
        z=cos(3.141592654d0*(i-.25d0)/(n+.5d0)) 
   1    continue 
          p1=1.d0 
          p2=0.d0 
          do 11 j=1,n 
            p3=p2 
            p2=p1 
            p1=((2.d0*j-1.d0)*z*p2-(j-1.d0)*p3)/j 
  11   continue 
       pp=n*(z*p1-p2)/(z*z-1.d0) 
       z1=z 
       z=z1-p1/pp 

       if (abs(z-z1).gt.eps) go to 1 

       x(i)=xm-xl*z 
       x(n+1-i)=xm+xl*z 
       w(i)=2.d0*xl/((1.d0-z*z)*pp*pp) 
       w(n+1-i)=w(i) 
  12  continue 

      return 
      end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
