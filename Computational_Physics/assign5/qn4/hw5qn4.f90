!Template: integ3.f90 was provided by Dr. Elster


! Author: Bhishan Poudel
! Date  : Sep 27,2015

! cmd: clear; f90 hw5qn4.f90 && ./a.out


      program hw5qn4
!
!        integrate sin(x)/x from xstart to xend with
!        - Simpson integration
!        - Gauss-legendre integration
!        - Filon's method   (Abramowitz/Stegun p. 890)
!
!
      implicit none

      integer :: nxmt,nxms,ng,nsxx,ngp
      integer :: kread, kwrite
      integer :: i,nxmx,nxmx1,nxmx2,nxfil,nxm,nxf,j,y
      double precision :: pi,sxx,wxx,u,w,x,xstart,xend
      double precision :: xival,dx,sum,sumg,d43,d23,dxf,sum1,sums,exact
      double precision :: alpha,beta,gamm,alphas,betas,gamms,f,fil1p,th,be1,ga1
      double precision :: s2n,s2n1,scn1,t2n,a2n,fil1,x2n,x2n1
      double precision :: xadd2n,xadd2n1,al1
      double precision :: eS,eG,eF ! relative errors


!    parameter (nxm=1000)
!    parameter (ng=96)
!    parameter (nxfil=900)
      
      parameter (nsxx=100000,ngp=100000)
      
      dimension sxx(nsxx),wxx(nsxx)
      dimension u(ngp),w(ngp)

      parameter (pi=3.141592654d0)
      parameter (kread=5,kwrite=6)
!
!
!        statement functions for filon integration
!
    
      alpha(x)=(x+0.5d0*sin(2.d0*x)-2.d0*sin(x)*sin(x)/x)/(x*x)
      beta(x)=2.d0/(x*x)*((1.d0+cos(x)**2)-sin(2.d0*x)/x)
      gamm(x)=4.d0/(x*x)*(sin(x)/x-cos(x))

!        for small values of x
      alphas(x)=2.d0/45.d0*x**3-2.d0/315.d0*x**5+7.d0/4725.d0*x**7
      betas(x)=2.d0/3.d0+2.d0/15.d0*x*x-4.d0/105.d0*x**4  &
     &  +2.d0/567.d0*x**6
      gamms(x)=4.d0/3.d0-2.d0*x*x/15.d0+x**4/210.d0-x**6/11340.d0
 
 
 
      xstart=1.0d0
      xend=1000.d0
      exact=0.62415d0 ! from Wolfram Alpha

      xival=xend-xstart
      open(kwrite, File="hw5qn4.dat", status="unknown")

      write (kwrite,10000) '#N',  'exact',  'Simpson','Gauss-Legendre','Filon',   'eS', 'eG',   'eF'  ! **** CHANGE
     
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     !       set up grid for Simpson integration
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	
	j=1
      do y=1,220
      nxm=4.0d0*j+1   ! now 4*1+=5
      nxmx=nxm/4.0d0  ! now 5/4=1 this changes 1,2,3,4,....
      nxmx=4*nxmx+1   ! now 4*1+1=5    
      dx=xival/float(nxmx-1)
       
      d43=4.0d0/3.d0
      d23=2.0d0/3.d0
 
      wxx(1)=dx/3.d0
      wxx(nxmx)=dx/3.d0
      
      sxx(1)=xstart
      sxx(nxmx)=xend
 
      nxmx1=nxmx-1
      nxmx2=nxmx-2
      
      
      
      ! alloting the values to sxx(i) and wxx(i)
      do i=2,nxmx1,2
      sxx(i)=float(i-1)*dx+xstart 
      wxx(i)=d43*dx
      end do
 
      do i=3,nxmx2,2
      sxx(i)=float(i-1)*dx +xstart
      wxx(i)=d23*dx
      end do
!
!       integrate
!
      sums=0.d0
      do i=1,nxmx
      sums=sums+(f(sxx(i)))*wxx(i)
      end do
 
 
!     write (kwrite,10001) nxm
!      write (kwrite,10000) sum
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!        gauss integration
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!        get gauss points and weights
!

      !print*, 'ng=', ng  ! ng = 2,4,6,8,..............
      ng=4.0d0*j+1.0d0   ! for j=1, ng=5 and nxmx is also 5
      call gauleg (xstart,xend,u,w,ng)
 
      do i=1,ng
      sxx(i)=u(i)
      wxx(i)=w(i)
      end do
!
!        integrate
!
      sumg=0.d0
      do i=1,ng 
      sumg = sumg+ f(sxx(i))*wxx(i)
      end do

 
!      write (kwrite,10002) ng
!      write (kwrite,10000) sum
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!        Filon integration
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!        prepare grid for Filon integration

      nxf=4*j+1   ! for j=1, nxf=5 and nxmx is also 5
      nxfil=nxf
      dxf=xival*0.5d0/float(nxfil)
      sum=0.d0

!        even and odd Filon sums

      s2n=0.d0
      s2n1=0.d0
      scn1=0.d0

!       first element of s2n

      if (xstart.lt.5.d-2) then
      t2n=1.d0-xstart*xstart/6.d0 +x**4/120.d0
      a2n=1.d0/x -0.5d0*x +x**3/24.d0
      else 
      t2n=sin(xstart)*fil1(xstart)
      a2n=cos(xstart)*fil1(xstart)
      endif

      s2n=t2n

      do i=1,nxf

        x2n=dxf*float(2*i)+xstart
        x2n1=dxf*float(2*i-1)+xstart


        if (x2n.lt.5.d-2) then
        xadd2n=1.d0-x2n*x2n/6.d0+x2n**4/120.d0
        else
        xadd2n=fil1(x2n)*sin(x2n)
        endif 

        if (x2n1.lt.5.d-2) then
        xadd2n1=1.d0-x2n1*x2n1/6.d0+x2n1**4/120.d0
        else
        xadd2n1=fil1(x2n1)*sin(x2n1)
        endif


        s2n=s2n+xadd2n
        s2n1=s2n1+xadd2n1
        scn1=scn1+fil1p(x2n1)*cos(x2n1)
      end do


!        complete s2n

      s2n=s2n-0.5d0*(fil1(xend)*sin(xend)+t2n) 


!        get Filon coefficients

      th=dxf

      if (th.lt.1.d-2) then
        al1=alphas(th)
        be1=betas(th)
        ga1=gamms(th)
      else
        al1=alpha(th)
        be1=beta(th) 
        ga1=gamm(th)
      endif


!        complete integration

      sum=sum+al1*(a2n-fil1(xend)*cos(xend))
      sum=sum+be1*s2n+ga1*s2n1

      sum1=sum+2.d0/45.d0*dxf**4*scn1

      sum1=sum1*dxf

 
 
  !    write (kwrite,10003) nxf
  !    write (kwrite,10000) sum1
 
  !     Writing the outputs:
  eS = abs((sums-exact)/exact)
  eG = abs((sumg-exact)/exact)
  eF = abs((sum1-exact)/exact)
  
        write(kwrite,20000) nxmx, exact, sums, sumg, sum1, eS,eG,eF
                           
      
        if(j.lt.150) then
        j=j+1
        else
        j=j+15
        endif
 
 end do
      10000 format(T1,A4,2x, 7(A12,3x))
      20000 format(T1,I4,2x, 6(F12.5,3x),F16.5)    ! for table
      
 close(kwrite)
      stop 'data saved in hw5qn4.dat'
      end
!**************************************************************
       double precision function f(r)
!         spherical besselfunction for l=0
       implicit none
       double precision r,result

       if(abs(r).lt.1.d-2) then
          result=1.d0-r*r/6.d0+r**4/1.2d2
      else
          result=sin(r)/r
      endif

       f = result
 
       return
       end
!*************************************************************************8888
       double precision function fil1(r)

       implicit none
       double precision r

       fil1=1.d0/r
       return
       end
!********************************************************************************88
       double precision function fil1p(r)

       implicit none
       double precision r

       fil1p=-6.d0/r**4
       return
       end 
       
!*****************************************************************************************************
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
