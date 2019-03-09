
!cmd: clear; gfortran -Wall pr1qn1c.f90 && ./a.out
!cmd: clear; f90 pr1qn1c.f90 && ./a.out
! change alpha, and f(x)

      program main_gaulag
      implicit none
      external f
      
      real(kind=16)::f,sxx
      integer :: ng,nsxx,ngp    !ng is no. of gauss points, ngp is paramter (max n-points)
      integer :: kread, kwrite
      integer :: i
      double precision :: pi,wxx,u,w
      double precision :: sum3,alpha
      character(len=50):: filename
      
      
 
!        integration points

      parameter (nsxx=2000)
      parameter (ngp=6400)
      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
 
      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngp),w(ngp)
      
      
      
      filename='pr1qn1c.dat'
      open(unit=kwrite,file=filename,status='unknown')
        !ng = 100
        
        do 10 ng=20,100
        alpha = 2.d0  !** CHANGE
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
      !sum3=sum3 + 1/(1- exp(-sxx(i)) )*wxx(i)  !f(x)
      sum3=sum3 + f(sxx(i))  *wxx(i)  !f(x)
      end do
 
 
      write(kwrite,10000) ng, sum3
      10000 format (i4, f10.5)
      
      
      10 end do
      close(kwrite)
      
      stop 'data saved in outputfile'
      
      end program

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

real(kind=16) function f(r)


       implicit none
       integer,parameter:: p=16
       real(kind=p)::  A(0:100)
       real(kind=p)::  B(0:100,0:100)
       integer     ::  n,i,factorial
       real(kind=16),parameter :: pi=3.141592654
       
       real(kind=16) r,hnx
       
       
     ! initializine A(k) values
     do i=1,100
     A(i) = 0._p
     end do

     n = 5 ! degree of Hermite Polynomial
     call Hermite_Coeff(n,A,B)
     
     
!          !!checking coefficients
!          do i=0,5
!          print*, i,A(i)
!          end do
! note: change f and alpha
       
     hnx=0._p
     do i=0,n
     hnx= hnx+A(i)* (r**i)
     end do
     
     
     
     f = 2._p*exp(r-r*r)*hnx*hnx/( 2._p**n * factorial(n) * sqrt(pi) )
     
     !hnx = A(0)+A(1)* (r**1) + A(2)* (r**2) + A(3) * (r**3) + A(4) * (r**4) + A(5) * (r**5) + A(6) * (r**6)
     
     !f = exp(r-r*r) * ((2._p*r*r-1)**2._p) /sqrt(pi)                   ! for n=2, exact = 2.5, alpha = 2
     !f = 2._p * exp(r-r*r) * ((2._p*r*r-3)**2._p) / (3._p * sqrt(pi))  ! for n=3, exact = 3.5, alpha = 4
     !f = exp(r-r*r) * ((4._p*r**4 - 20._p*r*r +15._p)**2._p) / (30._p * sqrt(pi))  ! for n=5, exact = 5.5, alf=4
     
 
       return
       end
       
INTEGER FUNCTION factorial(n) 
IMPLICIT NONE 
 
INTEGER, INTENT(IN) :: n 
INTEGER :: i, Ans 

Ans = 1 
DO i = 1, n 
Ans = Ans * i 
END DO 

factorial = Ans 
END FUNCTION factorial

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      SUBROUTINE gaulag (x,w,n,alf)

!        author:  Numerical Recipes

      implicit none

      INTEGER n,MAXIT

      double precision :: alf,w(n),x(n)
      double precision :: EPS
      parameter (EPS=3.D-14,MAXIT=10)

      double precision nx

!       USES gammln
     
!        Give alf (=parameter alpha of the Laguerre polynomials) the routine returns
!        arrays x(1:n) and w(1:n) containing the abscissas as weights of the n-point 
!        Gauss-Laguerre quadrature formula. The smallest abscissa is returned in x(1)
!        and the largest in x(n)

      integer :: i,its,j
      double precision :: ai,gammln

      double precision :: p1,p2,p3,pp,z,z1

      do 13 i=1,n
        if(i.eq.1)then
          z=(1.+alf)*(3.+.92*alf)/(1.+2.4*n+1.8*alf)
        else if(i.eq.2)then
          z=z+(15.+6.25*alf)/(1.+.9*alf+2.5*n)
        else
          ai=i-2
          z=z+((1.+2.55*ai)/(1.9*ai)+1.26*ai*alf/(1.+3.5*ai))* &
     &(z-x(i-2))/(1.+.3*alf)
        endif

        do 12 its=1,MAXIT
          p1=1.d0
          p2=0.d0
          do 11 j=1,n
            p3=p2
            p2=p1
            p1=((2*j-1+alf-z)*p2-(j-1+alf)*p3)/j
 11        continue
          pp=(n*p1-(n+alf)*p2)/z
          z1=z
          z=z1-p1/pp
          if(abs(z-z1).le.EPS)goto 1
 12      continue

!        pause 'too many iterations in gaulag'

 1      x(i)=z

        nx=float(n)
 
        w(i)=-exp(gammln(alf+nx)-gammln(nx))/(pp*nx*p2)

 13    continue
      return
      end

!  (C) Copr. 1986-92 Numerical Recipes Software 2721[V3.
!---------------------------------------------------------------

      function gammln(xx)

!        function calculates the value ln[Gamma(xx)] for xx>0
!        internal arithmetic carried out in double precision
!        (numerical recipes)

      implicit none

      integer :: j
      double precision :: gammln,xx

      double precision :: cof(6),stp,half,one,fpf,x,y,tmp,ser

      save cof,stp

      data cof,stp/76.18009173d0,-86.50532033d0,24.01409822d0,   &
     &    -1.231739516d0,.120858003d-2,-.536382d-5,2.50662827465d0/
      data half,one,fpf/0.5d0,1.0d0,5.5d0/


      x=xx
      y=x     
!      x=xx-one
      tmp=x+fpf
      tmp=(x+half)*log(tmp)-tmp
      ser=one

      do  j=1,6
        y=y+one
        ser=ser+cof(j)/y
      end do

      gammln=tmp+log(stp*ser/x)

      return

      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Subroutine Hermite_Coeff(n,A,B)
  
  integer           :: i,j,n
  integer,parameter :: p=16
  real(kind=p)      :: A(0:10), B(0:10,0:10)


  B(0,0)=1._p ; B(1,0)=0._p ; B(1,1)=2._p


  if (n>1) then
    do i = 2, n
      B(i,0)=-2._p*(i-1)*B(i-2,0)
      do j = 1, i
      
        !recursion relation
        B(i,j)=2._p*B(i-1,j-1)  -  2._p*(i-1)*B(i-2,j)
      end do
    end do
    do i = 0, n
      A(i)=B(n,i)
    end do
  end if
  return
end


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    

