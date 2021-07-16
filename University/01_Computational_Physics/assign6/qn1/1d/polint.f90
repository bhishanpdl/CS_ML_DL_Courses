      subroutine polint (xa,ya,n,x,y,dy)

!       given arrays xa and ya, each of length n, and given value of x,
!       this routine returns a value y, and an error estimate dy.
!       If P(x) is the polynomial of degree N-1 sich that 
!       P(xa_i)=ya_i, i=1,...,n, then the returned value is y=P(x).
!

      integer :: n,NMAX
      double precision :: dy,x,y,xa(n),ya(n)
!       largest anticipated value of n
      parameter (NMAX=10)

      integer :: i,m,ns
      double precision :: den,dif,dift,ho,hp,w,c(NMAX),d(NMAX)

      ns=1
      dif=abs(x-xa(1))
!        find the index ns of the closest table entry
      do  i=1,n
        dift=abs(x-xa(i))
        if (dift.lt.dif) then
          ns=i
          dif=dift
        endif
        c(i)=ya(i)
        d(i)=ya(i)
      end do 

      y=ya(ns)
      ns=ns-1
      do 13 m=1,n-1
        do 12 i=1,n-m
          ho=xa(i)-x
          hp=xa(i+m)-x
          w=c(i+1)-d(i)
          den=ho-hp
!     error if two points in xa are (within roundoff error) identical
          if(den.eq.0.)pause 'failure in polint'
          den=w/den
          d(i)=hp*den
          c(i)=ho*den
12      continue

        if (2*ns.lt.n-m)then
          dy=c(ns+1)
        else
          dy=d(ns)
          ns=ns-1
        endif
        y=y+dy
13    continue

      return
      END
!  (C) Copr. 1986-92 Numerical Recipes Software 2721[V3.
