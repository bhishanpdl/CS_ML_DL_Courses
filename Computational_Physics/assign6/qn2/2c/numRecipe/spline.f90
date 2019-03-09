      subroutine spline (x,y,n,yp1,ypn,y2)

      implicit none

      integer ::  n,NMAX
      double precision ::  yp1,ypn,x(n),y(n),y2(n)

      parameter (NMAX=500)
      integer :: i,k

      double precision ::  p,qn,sig,un,u(NMAX)

      if (yp1.gt..99d30) then
        y2(1)=0.d0
        u(1)=0.d0
      else
        y2(1)=-0.5d0
        u(1)=(3.d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do 11 i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.d0
        y2(i)=(sig-1.d0)/p
        u(i)=(6.d0*((y(i+1)-y(i))/(x(i+    &
     &1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*  &
     &u(i-1))/p
11    continue
      if (ypn.gt..99d30) then
        qn=0.d0
        un=0.d0
      else
        qn=0.5d0
        un=(3.d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.d0)
      do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12    continue
      return
      end
!  (C) Copr. 1986-92 Numerical Recipes Software 2721[V3.
!-------------------------------------------------------

      subroutine splint(xa,ya,y2a,n,x,y)

      implicit none

      integer :: n
      double precision :: x,y,xa(n),y2a(n),ya(n)
      integer :: k,khi,klo
      double precision :: a,b,h

      klo=1
      khi=n
1     if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
          khi=k
        else
          klo=k
        endif
      goto 1
      endif
      h=xa(khi)-xa(klo)
      if (h.eq.0.) pause 'bad xa input in splint'
      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h
      y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**  &
     .2)/6.d0
      return
      END
!  (C) Copr. 1986-92 Numerical Recipes Software 2721[V3.
