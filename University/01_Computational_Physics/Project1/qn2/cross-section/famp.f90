
!bp: clear; gfortran -w famp.f90 && ./a.out
!bp: clear; f90 famp.f90 && ./a.out


subroutine famp(theta,sum3)
implicit none
  external ftheta
  double precision:: ftheta,theta

      integer :: ng,nsxx,ngp    !ng is no. of gauss points, ngp is paramter (max n-points)
      integer :: kread, kwrite
      integer :: i
      double precision :: pi,sxx,wxx,u,w
      double precision :: sum3,error3,exact,alpha
      character(len=50):: filename
      
      
 
!        integration points

      parameter (nsxx=2000)
      parameter (ngp=6400)
      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
 
      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngp),w(ngp)
      
        !theta = 0.020d0
      
        ng = 90
        alpha = 0.d0
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
      sum3=sum3 + ftheta(theta,sxx(i))  *wxx(i)  !f(x)
      end do
 
 
      !write(kwrite,10000) theta,sum3

      !10000 format(f10.3,e20.6)

return
end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


double precision function ftheta(theta,r)

       implicit none
       double precision theta,r

       ftheta = -0.4726d0 * exp(r-7.559d0*r) * sin(16.d0 * r * sin(theta/2.d0)) / sin(theta/2.d0)
 
       return
       end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

