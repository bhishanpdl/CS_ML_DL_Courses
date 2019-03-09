!Topic     : Assign1 Question1 plot2
!Programmer: Bhishan Poudel
!cmd       : clear; gfortran -o qn1p2 qn1p2.f95 && ./qn1p2 |& tee qn1p2.dat

!note      : sinx = x - x^3/6 + x^5/120 - .....
!            cosx = 1 - x^2/2 + x^4/24 + .....

program qn1
implicit none

    real :: x = 0.0
    real :: y
              
            
    do while ( x < 10)
        if(x<0.1) then
            y = x/3 - x*x*x/30  ! using Taylor expansion near x = 0
            write(*,*) x,y
        else
            y = sin(x)/(x*x) - cos(x)/x
            write(*,*) x,y
        end if
    x = x + 0.001
    end do
    
  
end program qn1
