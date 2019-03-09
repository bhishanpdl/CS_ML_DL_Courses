!  clear; f95 pr2qn2.f90 && ./a.out ; rm *~ a.out

  program pr2qn2
  implicit none
  
  integer          :: kread,kwrite,i,j,n,kwrite1,k,ng,l  
  double precision :: P,com,act
  double precision :: pi
  double precision :: sxx(400),wxx(400),u(400),w(400),hc,c1,c2
  double precision :: x,sumleg
  double precision :: Va,MuA,LambdaA,Vr,MuR,LambdaR
  double precision :: q,qprime
  double precision :: term1,term2,term3 ! (q-q')^2 and other terms
  
  data kread/5/,kwrite/6/,kwrite1/7/
  
  parameter(pi=3.14159d0)
    
    ! constants given in question
    hc=197.3286d0       ! hbarC Mevfm-1
    Vr=1843.8384d0/hc   ! dimensionless 
    MuA=1.673d0*hc      ! Mev 
    LambdaA=7.6015d0*hc ! Mev 
    Va=1085.3073d0/hc   ! dimensionless
    MuR=3.10d0*hc       ! MeV
    LambdaR=7.6015d0*hc ! Mev
    
    !intialize the Values
    qprime = 2.5d0*hc     ! Mev
    q      = 0.01d0
    l      = 4
    
    !write output
    open (kwrite1, file=   "v4b.dat",status="unknown")
        write(kwrite1,*)   '# potential for l=',l,'and q''=', qprime/hc ,'fm^-1'
        write(kwrite1,*) 
        write(kwrite1,100) '#q', 'vl(q,q'')'        
   
        q=0.01d0
   
        do while(q.le.20.0d0)  
            ng = 10 !No of gauss points 
            call gauleg (-1.0d0,1.0d0,u,w,ng)
             
           do i=1,ng
               sxx(i)=u(i)
               wxx(i)=w(i)!weights
           end do

           !integrate
           sumleg=0.d0
      
           do  i=1,ng
               call Legendre_Polynomials(l,sxx(i),P)
               term1=(q*hc)**2+qprime**2-2*q*hc*qprime*sxx(i)
               term2=(Vr/(term1+MuR**2))*((LambdaR**2-MuR**2)/(term1+LambdaR**2))**2
               term3=(Va/(term1+MuA**2))*((LambdaA**2-MuA**2)/(term1+LambdaA**2))**2
               sumleg=sumleg +wxx(i)*P*(term2-term3) 
          end do
     
	     c1=sqrt(938.9d0/(sqrt(938.9**2+q*q*hc*hc)))
	     c2=sqrt(938.9d0/(sqrt(938.9**2+qprime**2)))
         com=(1/pi)*sumleg*c1*c2 !computed integral
   
         write(kwrite1,1000)q,com
         q=q+0.01d0  ! updating Values of q (this will have unit MeV)
    end do

  100  format (2a16)
  1000 format (2es16.6)
  close(kwrite1)
  stop'data is saved'
  end program 
  
!subroutine to calcMuAte legendre polynomials 
!-------------------------------------------------------------------------------     
subroutine  Legendre_Polynomials(k,x,P)
implicit none
    integer         ::n,k
    double precision::x,P,leg(k)
    
    leg(0)=1  ! P0(x) = 0
    leg(1)=x  ! P1(x) = x
   
    do n=1,k-1,1
        leg(n+1)= ((2*n+1)/(n+1))*x*leg(n)-(n/(n+1))*leg(n-1) ! recMuRsion reLambdaAtion
    end do
    P=leg(k)  ! legendre polynomial
  return
  end 
!-------------------------------------------------------------------------------
!subroutine to integrate using Gaus-Legendre integration method
!-------------------------------------------------------------------------------
subroutine gauleg (x1,x2,x,w,n) 
!
!        calcuLambdaAte gauss points for gauss-legendre integration
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
!  (C) Copr. 1986-92 Numerical Recipes Software 2721[V3.
!******************************************************************************
