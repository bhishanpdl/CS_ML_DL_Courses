!Bhishan Poudel
!Dec 7, 2015

! clear; f95 pr2qn7.f90 -llapack && ./a.out ; rm *~ a.out

! Topic     : This program calcuates the binding energy of deuteron
! Dependence: This program uses lapack routine 'dgeev' and external subroutine 'gauleg'


program pr2qn7
implicit none

integer          :: i,j,ng,yes
double precision :: mu,hc,pi
double precision :: v,norm

parameter (ng=50)

double precision :: x(ng),w(ng)  ! points and weights from Gauss-Legendre
double precision :: p(ng),w1(ng) ! points and weights to do the integral
double precision :: h(ng,ng)     ! Hamiltonian Matrix

double precision :: WR(ng),WI(ng),VL(ng,ng),VR(ng,ng),WORK(4*ng)
integer          :: INFO
integer          :: kwrite1,kwrite2
data kwrite1/6/, kwrite2/7/

! constants
mu=938.9d0/2.d0  ! average mass is 938.9 Mev
hc=197.3269788d0 ! hbar*c is 197.327 Mev fm
pi=2.d0*acos(0.d0)

! guessing points and weights 
call gauleg (-1.d0,1.d0,x,w,ng)

! mapping as given in Hjorth-Jensen fall 2007 page 321
do i=1,ng
  p(i)=tan(pi/4.d0*(1.d0+x(i)))                  ! p  = tan((x+1)*pi/4)  
  w1(i)=pi/4.d0*w(i)/cos(pi/4.d0*(1.d0+x(i)))**2 ! w1 = pi_by_4 * w/ [cos^ {pi_by_4*(1+x)}]
end do


! create Hamiltonian matrix
do i=1,ng
  p(i) = p(i)
  do j=1,ng
    if (i.eq.j) then
      h(i,j)=hc**2*p(i)**2/(2.d0*mu) + p(i)**2*w1(i)*v(p(i),p(i))
    else
      h(i,j)=v(p(i),p(j))*w1(j)*p(j)**2
    end if
  end do
end do

!********************************************************************************

! soubroutine dgeev.f to evaluate eigenvalues and eigenvectors 
call DGEEV('N', 'V', ng, h, ng, WR, WI, VL, ng, VR, ng, WORK, 4*ng, INFO)

! position of the binding energy
do i=1,ng
  if (wr(i).lt.0.d0) then
      yes=i 
      exit
  end if
end do

!write(*,*) 'Binding energy is equal to ',wr(yes),'MeV'

! calculate the norm of the wave function
norm=0.d0
do i=1,ng
   norm=norm+w1(i)*vr(i,yes)**2
end do
norm=sqrt(norm)

! Write wavefunction  
open(kwrite2,file= 'wavefunctionNew.dat',status='unknown')
write(kwrite2,*)   '#number of gauss points = 50'
write(kwrite2,100) '#q','psi(q)'
100 format (2a20)

do i=1,ng
  p(i) = p(i)
  write(kwrite2,10000) p(i),-vr(i,yes)/norm
10000 format(2f20.10) 
end do

 close(kwrite2)

stop 'data is saved'
end program

!********************************************************************************
! integrating the potential vl(q,q') from -1 to 1 with gauleg subroutine
 double precision function v(qprime,q)
      
      implicit none
      integer          :: ng,nsxx,ngp,i
      double precision :: sxx,wxx,u,w,sum, Vr, LaR, muR, Va, muA, LaA
      double precision :: pi,qprime,q,x,v1
      double precision :: m,E,Eprime 
       parameter (ng=50)
       parameter (nsxx=2000000)
       parameter (ngp=1000000)
       dimension sxx(nsxx),wxx(nsxx)
       dimension u(ngp),w(ngp)
       
       parameter (m = 938.9d0)
       parameter (pi= 3.14159265359d0)

       parameter (Va = 900.3073d0)
       parameter (muA = 1.673d0)
       parameter (LaA = 7.6015d0)
       parameter (Vr = 1843.8384d0)
       parameter (muR = 3.10d0)
       parameter (LaR = 7.6015d0)

       !E = sqrt(m*m+q*q)
       !Eprime = sqrt(m*m + qprime *qprime)
       !factor1 = sqrt(m/E)

          v1(qprime,q,x)= 0.5d0/pi**2 * sqrt(m / sqrt(m*m+q*q)) * sqrt(m/ sqrt(m*m + qprime *qprime)) * ( &
                          &               Vr/(qprime**2+q**2-2.d0*qprime*q*x+muR**2) * &
                          &                        (((LaR**2-muR**2)/(qprime**2+q**2-2.d0*qprime*q*x+LaR**2))**2) - &
                          &               Va/(qprime**2+q**2-2.d0*qprime*q*x+muA**2)  * &
                          &                        ((LaA**2-muA**2)/(qprime**2+q**2-2.d0*qprime*q*x+LaA**2))**2)
  
 
       call gauleg (-1.d0,1.d0,u,w,ng)
 
       do i=1,ng
           sxx(i)=u(i)
           wxx(i)=w(i)
          end do

!     integrating
           sum=0.d0
           do i=1,ng
            sum=sum+2.d0*pi*v1(qprime,q,sxx(i))*wxx(i)
           end do
      v=sum
      return
      end function
!*******************************************************************************
!*******************************************************************************

      subroutine gauleg (x1,x2,x,w,n) 
!
!        calculate gauss points for gauss-legendre integration
!                 (numerical recipes)  
!
      implicit double precision (a-h,o-z) 
!
      parameter (eps=3.d-14) 
      dimension x(n),w(n)
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
      end 
      

