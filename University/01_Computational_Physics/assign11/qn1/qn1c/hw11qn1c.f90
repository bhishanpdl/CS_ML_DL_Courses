!cmd: clear; f95 hw11qn1c.f90 && ./a.out ; rm -f *.o *fort* *~

      program testdifsis
!
!       test diff equation
!
      implicit none

      external f1

      double precision x,y,h
      double precision xinitial, xfinal,exact(500),error(500),eps

      double precision xx(500)
      double precision yy(500)
      integer i, ix, n

      integer kread, kwrite
      data kread/5/, kwrite/6/, eps /1d-16/
      
      
     open(unit=kwrite,file='hw11qn1c.dat',status='unknown')

      n=1
      xinitial = 0.d0
      xfinal   = 2.d0
      h=0.05d0

      !initialize

      x=xinitial
      y=0.d0      

      ix=1
      
      !initialize error
      error(ix) = 0.d0

 100  continue
        exact(ix)=exp(2.d0*x)-1.d0
        
        if (exact(ix).gt.eps) then    !to avoid floating point exception (divide by zero)
        error(ix)=abs(exact(ix)-y)/abs(exact(ix))
        else
        error(ix)=0.d0
        end if
      call difsis (f1,y,x,n,h)

      ix=ix+1
      xx(ix)=x
      yy(ix)=y

      if (x.le.xfinal) go to 100 

      write (kwrite,*) '#final h:',h
      write (kwrite,*) '#      x            y               exact           error  '
        exact(ix)=exp(2.d0*x)-1.d0
        
        !avoid divide by zero
        if(abs(error(ix)).gt.eps) then
        error(ix)=abs(exact(ix)-y)/abs(exact(ix))
        else
        error(ix) = 0.d0
        end if
        
      xx(1)=xinitial
      yy(1)=0.d0    
 
      do i=1,ix
          write (kwrite,10001) xx(i),yy(i),exact(i),error(i)
      end do
10001 format(4d15.6)
 
      close(kwrite)
      stop 'data is saved in hw11qn1c.dat' 
      end program
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine f1 (xl,y,dy)
      implicit none

      double precision xl
      double precision y,dy

     ! dy=-y*xl
           dy=2.d0*y+2.d0

      return
      end subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine difsis(f,y,x,n,h)

!   ***   differential equation solver ***
!    *   n: number of equations,  f: f(x,y,y'),
!    *   eps: accuracy, h: step size, 
!    *   x: independent variable, y: dependent variable
!
      implicit double precision (a-h,o-z)

      parameter (eps=1.d-8)

      dimension y(22),ya(22),yl(22),ym(22),dy(22),dz(22),dt(22,7)
      dimension d(7),s(22),ep(4)

      logical konv,bo,kl,gr
      data ep/0.4d-1,0.16d-2,0.64d-4,0.256d-5/
      data kwrite/6/

      data jtimax/30/

      jti=0
      fy=1.d0
      eta = dabs (eps)
      if(eta.lt.1.d-11) eta=1.d-11
      do 100 i=1,n
  100 ya(i)=y(i)
      xl = x
      call f(xl,y,dz)
      if(dz(1).gt.1.d36) go to 30
   10 xn=x+h
      bo=.false.
      do 110 i=1,n
  110 s(i)=0.d0
      m=1
      jr=2
      js=3
      do 260 j=1,10
      if(.not.bo) go to 200
      d(2)=1.77777777778
      d(4)=7.11111111111
      d(6)=28.4444444444
      go to 201
  200 d(2)=2.25d0
      d(4)=9.d0
      d(6)=3.6d1
  201 if(j.le.7) go to 202
      l=7
      d(7)=6.4d1
      go to 203
  202 l=j
      d(l)=m*m
  203 konv=l.gt.3
      m=m+m
      g=h/dfloat(m)
      b=g+g
      m=m-1
      do 210 i=1,n
      yl(i)=ya(i)
  210 ym(i)=ya(i)+g*dz(i)
      do 220 k=1,m
      xl = x+dfloat(k)*g
      call f(xl,ym,dy)
      if(dy(1).gt.1.d36) go to 30
      do 220 i=1,n
      u=yl(i)+b*dy(i)
      yl(i)=ym(i)
      ym(i)=u
      u=dabs(u)
      if(u.gt.s(i)) s(i)=u
  220 continue
      xl = xn
      call f(xl,ym,dy)
      if(dy(1).gt.1.d36) go to 30
      kl=l.lt.2
      gr=l.gt.5
      fs=0.d0
      do 233 i=1,n
      v=dt(i,1)
      c=(ym(i)+yl(i)+g*dy(i))*0.5d0
      dt(i,1)=c
      ta=c
      if(kl) go to 233
      do 231 k=2,l
      b1=d(k)*v
      b=b1-c
      w=c-v
      u=v
      if(b.eq.0.d0) go to 230
      b=w/b
      u=c*b
      c=b1*b
  230 v=dt(i,k)
      dt(i,k)=u
  231 ta=u+ta
      if(.not.konv) go to 232
      if(dabs(y(i)-ta).gt.s(i)*eta) konv=.false.
  232 if(gr.or.s(i).eq.0.d0) go to 233
      fv= dabs(w)/s(i)
      if(fs.lt.fv) fs=fv
  233 y(i)=ta
      if(fs.eq.0.d0) go to 250
      fa=fy
      k=l-1
      fy=(ep(k)/fs)**(1./dfloat(l+k))
      if(l.eq.2) go to 240
      if(fy.lt.0.7*fa) go to 250
  240 if(fy.gt.0.7) go to 250
      h=h*fy
      jti=jti+1
      if(jti.gt.jtimax) go to 30
      go to 10
  250 if(konv) go to 20
      d(3)=4.d0
      d(5)=1.6d1
      bo=.not.bo
      m=jr
      jr=js
  260 js=m+m
      h=h*0.5d0
      go to 10
   20 x=xn
      h=h*fy
      return
   30 h=0.d0
      do 300 i=1,n
  300 y(i)=ya(i)

      write(kwrite,310) x,jti
  310 format('0',t10,'danger !!!',3x,'difsis stopped at x =',d10.3, &
     &    5x,'jti =',i2)
      x = x + 1000.d0
      return
      end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


