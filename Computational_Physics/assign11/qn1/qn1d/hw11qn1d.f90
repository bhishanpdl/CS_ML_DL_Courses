!cmd: clear; f95 hw11qn1d.f90 && ./a.out
!

program picard
implicit none

    double precision  :: y0(0:1000) , y1(0:1000)
    double precision  :: x,xi,xf,f
    double precision  :: exact,h,sum,error
    integer           :: nstep,niter,i,it,n
    
    !choosing xi,xf,and steps
    data xi,xf,nstep/0.d0,2.d0,100/
    
    !function to use
    f(x,n)=2.d0*y0(n) + 2.d0
    
    !exact solution
    exact(x) = exp(2.d0*x)-1.d0
    
    !step size 
    h=(xf-xi)/float(nstep)
    
    !write output in files
    open(unit=4, file='picard4.dat', status='unknown')
    open(unit=8, file='picard8.dat', status='unknown')
    open(unit=12,file='picard12.dat',status='unknown')
    open(unit=16,file='picard16.dat',status='unknown')
    
         
         do niter=4,16,4
         write(niter,200) '#x','answer','exact','error'
            do 10 i=0,nstep
               x=xi+h*float(i)
               y0(i)=0.d0
            10 continue

            do 50 it=1,niter
               sum=y0(0)

            do 20 i=1,nstep
               x=xi+h*float(i)
               sum=sum+(h/2.d0)*(f(x,i) + f(x-h,i-1) )
               y1(i)=sum
            20 continue

            do 30 i=0,nstep
               y0(i)=y1(i)
               if(i.eq.0)y0(i)=0.d0
            30 continue
            50 continue
        
            do 60 i=0,nstep,int(float(nstep)/20.d0)
               x=xi+h*float(i)
               error=abs(y0(i)-exact(x))
               write(niter,100) x,y0(i),exact(x),error  !write output in niter
            60 continue
            
            !-ve values
            do 70 i=1,nstep
               x=xi-h*float(i)
               y0(i)=0.d0
            70 continue

            do 80 it=1,niter
               sum=y0(0)

            do 90 i=1,nstep
               x=xi-h*float(i)
               sum=sum-(h/2.d0)*(f(x,i) + f(x-h,i-1) )
               y1(i)=sum
            90 continue

            do 110 i=0,nstep
               y0(i)=y1(i)
               if(i.eq.0)y0(i)=0.d0
           110 continue
            80 continue

            do 120 i=0,nstep,int(float(nstep)/20.d0)
               x=xi-h*float(i)
               error=abs(y0(i)-exact(x))
               write(niter,100) x,y0(i),exact(x),error
            120 continue
       end do
 
100 format(4f16.6)
200 format(4a16)
        stop'Data is saved'
        end
