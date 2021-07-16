!cmd : clear; gfortran -Wall bisection.f90 && ./a.out > bisection2.dat
!cmd : clear; f90 bisection.f90 && ./a.out > bisection1.dat    for 0.1 and 1.0
!cmd : clear; f90 bisection.f90 && ./a.out > bisection2.dat    for 5.1 and 10.0


!***********************************************
!*        Main Program to calculate root       *
!***********************************************
PROGRAM Bisection

real*8   tolerance,x,x0,x1,F  ! rea*8  and double precision are same things
integer step

!! initial range (x0,x1) to find the root within it
!x0 = 0.1d0   !! for first root ****CHANGE THIS
!x1 = 1.0d0   
x0 = 5.1d0   !! for second root
x1 = 10.0d0
   
tolerance  = 1e-7  ! no. of depends on tolerance


  call Bisect( tolerance,step,x,x0,x1)    ! Call bisection routine  
  stop

end

!***********************************************
!*        Function F(x)                        *
!***********************************************
real*8 Function F(x) 
  real*8 x
  F = x*x - 7*x - LOG(x)
end


!***********************************************
!*        Bisection method subroutine          *
!***********************************************
Subroutine Bisect( tolerance,step,x,x0,x1)
  integer step            ! output variables
  real*8  tolerance,x0,x1 ! input variables
  real*8 x,F              ! output variables
  real*8 y0,yy            ! local variables
  integer kread, kwrite
  data kread/5/, kwrite/6/
  
    step=0
    write (kwrite,*) '#Search Root via Bisection Method : '
    write(kwrite,45) '#ieration', 'root(x)', 'f(x)'
    45 format (T1,A12, T20, A8, T35, A8)
    
    
    !! starting the loop
    50 y0=F(x0)         !x0 and x1 are input values
    x=(x0+x1)*0.5d0     ! multiplicaton is faster than division
    yy=F(x) 
    step=step+1
    
    write(kwrite,49) step,x,F(x)
    if (dabs(yy*y0).eq.0) return
    if ((yy*y0)<0.d0) x1=x
    if ((yy*y0)>0.d0) x0=x
    if (dabs(x1-x0)> tolerance) goto 50
  
  return
  49 format(I4,T14,F12.5,T35, E12.5)
end


! End of file Bisect.f90
