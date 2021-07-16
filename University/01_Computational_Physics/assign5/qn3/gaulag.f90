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
