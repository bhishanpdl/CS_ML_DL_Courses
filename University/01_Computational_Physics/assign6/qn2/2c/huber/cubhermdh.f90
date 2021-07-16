      subroutine cubhermdh (xold,n,xnew,m,spl,index)

      implicit double precision (a-h,o-z)
!
!     This subroutine prepares the interpolation of a function
!     given at the n grid points xold to the m points xnew.
!     The interpolating functions are cubic hermitian splines.
!     The first derivatives at the grid points are taken from a
!     parabola through the actual grid point and its two neighbours.
!     For the end points the parabola is taken through the two
!     right or left neighbours, resp.

!     In the calling routine you still have to take the sum i=1,4
!     ynew(j)=sum_i spl(j,i)*yold(index(j,i))

!     by Dirk Hueber, 08.02.1996
!     Appendix B in
!     Few-Body Systems 22, 107 (1997)

!     parameter (nmax=100,mmax=100)
      dimension xold(n),xnew(m),spl(m,4),index(m,4)
      logical enough

      enough=n.ge.3

!     if (n.gt.nmax) stop'nmax to small in cubherm'
!     if (m.gt.mmax) stop'mmax to small in cubherm'

!     evaluation of the indices

      do 10 j=1,m

!     If xnew(j) is less than xold(1), we extrapolate using the first
!     two grid points xold(1) and xold(2).
 
       index(j,2)=1
10    continue
      do 11 i=1,n
       do 11 j=1,m
        if (xnew(j).gt.xold(i)) index(j,2)=i
11    continue
      do 12 j=1,m
       index(j,2)=min(index(j,2),n-1)
       index(j,1)=index(j,2)-1
       index(j,3)=index(j,2)+1
       index(j,4)=index(j,2)+2

!     Indices 1 and 4 are used only for the ewertcation of the derivatives.
!     The following settings provide the derivatives at the borders of the
!     grid xold.

       if (index(j,1).eq.0) index(j,1)=3
       if (index(j,4).eq.n+1) index(j,4)=n-2
12    continue
      do 20 j=1,m

!     We don't extrapolate to the right!

       if (xnew(j).le.xold(n).and.enough) then
        i0=index(j,1)
        i1=index(j,2)
        i2=index(j,3)
        i3=index(j,4)
        x0=xold(i0)
        x1=xold(i1)
        x2=xold(i2)
        x3=xold(i3)

!      Factors for the derivatives

        d10=x1-x0
        d21=x2-x1
        d32=x3-x2
        d20=x2-x0
        d31=x3-x1
        dfak13=(d21/d10-d10/d21)/d20
        dfak14=-d32/(d21*d31)
        dfak23=d10/(d21*d20)
        dfak24=(d32/d21-d21/d32)/d31
        dfak03=-d21/(d10*d20)
        dfak34=d21/(d32*d31)

!     the cubic hermitian splines

        xn=xnew(j)
        dn1=xn-x1
        d2n=x2-xn
        phidiv=1./(d21*d21*d21)
        phi1=d2n*d2n*phidiv*(d21+2.*dn1)
        phi2=dn1*dn1*phidiv*(d21+2.*d2n)
        phidiv=phidiv*d21*dn1*d2n
        phi3=phidiv*d2n
        phi4=-phidiv*dn1
 
!     combining everything to the final factors
 
        spl(j,2)=phi1+phi3*dfak13+phi4*dfak14
        spl(j,3)=phi2+phi3*dfak23+phi4*dfak24
        spl(j,1)=phi3*dfak03
        spl(j,4)=phi4*dfak34
       else
        spl(j,2)=0.
        spl(j,3)=0.
        spl(j,1)=0.
        spl(j,4)=0.
       endif
  20  continue
      end
