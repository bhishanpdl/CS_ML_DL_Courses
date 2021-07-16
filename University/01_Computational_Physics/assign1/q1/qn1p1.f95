!Topic     : Assign1 Question1 plot 1
!Programmer: Bhishan Poudel
!cmd       : clear; gfortran -o qn1p1 qn1p1.f95 && ./qn1p1 |& tee qn1p1.dat
!            we can see output as well as save it in a file

!note      : sinx = x - x^3/6 + x^5/120 - .....
!            cosx = 1 - x^2/2 + x^4/24 + .....

program qn1p1
implicit none

    real :: x = 0.d0
    real :: y
    integer:: n = 0 ! to find the no. of data
              
            
    do while ( x < 10)
        if(x<0.001) then             ! using Taylor expansion near x = 0
            y = 1 - x*x/2.0
            write(*,*) x,y
        else
            y = sin(x)/x
            write(*,*) x,y
        end if
    x = x + 0.0001
    n = n+1
    end do
    
  ! to create the data output, we give following command in linux
  ! clear; gfortran -o qn1a qn1a.f95 && ./qn1a |& tee qn1a.dat
    
  
end program qn1p1
