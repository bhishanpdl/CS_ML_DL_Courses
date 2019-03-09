!Topic     : Assign1 Question1 plot2
!Programmer: Bhishan Poudel
!cmd       : clear; f90 qn1p2.f90 && ./a.out

!note      : sinx = x - x^3/6 + x^5/120 - .....
!            cosx = 1 - x^2/2 + x^4/24 + .....

program qn1
implicit none

    real :: x = 0.0
    real :: y
    real :: error         
    real :: step    ! step for do loop
    character(len=20) :: filename ! file to write data
    
    error = 0.01
    step  = 0.001
    filename = 'qn1p2.dat'
              
     
    open(unit=9,file=filename,status='replace')        
    do while ( x < 10)
        if(x<0.1) then
            y = x/3 - x*x*x/30  ! using Taylor expansion near x = 0
            write(9,100) x,y
        else
            y = sin(x)/(x*x) - cos(x)/x
            write(9,100) x,y
        end if
        x = x + 0.001
    end do
    100 format (1x,2(f10.5))  ! formatting two floats
    close(9)
    print*, 'The data is stored in the file ', filename
    
  
end program qn1
