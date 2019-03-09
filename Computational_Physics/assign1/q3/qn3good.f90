!Topic     : Assign1 Q3
!Programmer: Bhishan Poudel
!cmd       : clear; f90 qn3good.f90 && time ./a.out  !sunstudio compiler doesnot compile
!cmd       : clear; gfortran -o qn3good -Wall qn3good.f90 && time ./qn3good  !gfortran compiler works fine
!                            W means Warning
!exp -x    : 1 -x + x^2/2! -x^3/3! + ...
!BADWAY is          using term = term*(-x)/factorial
!GOODWAY is         using term = term*(-x)/float(n)


program expngood
implicit none

    !defining variables
    real             :: Err = 1e-8  !pre-defined error estimate	
    integer          :: n           !counter for diff values of x
    double precision :: x           !x=0.1,1.0,10.0,100.0and 1000.0
    double precision :: sum
    double precision :: term
    double precision :: ratio
    integer          :: kwrite      ! standard value for write function
    data kwrite/6/
    
    !using time function
    double precision :: time_begin,time_end
    call cpu_time(time_begin)
      
       
    !writing the Table header   
    write(kwrite,*) '   ________________________________________________' 
    write(kwrite,*) '      x       imax          sum           ratio    '
    write(kwrite,*) '   ________________________________________________' 
    
    !use do while loop for different values of x    
    x = 0.01d0           ! initialize the value of x to be changed later
    do while (x < 1000 ) ! loop to get x = 0.1,1.0,10.0,100.0 and 100.0                   
        x = x*10                  
       
        term = 1.d0   !first term = 1
        sum = 1.d0    !sum upto first term
      
        n = 0
      
        !example 1st iteration: x =1, we will calculate exp(-x) within the error            
        do while ((abs(term/sum) > err))
            n = n + 1
            term = term*(-x)/float(n) 
            sum = sum + term
        end do
        
        ratio = abs(sum - exp(-x))/sum ! given in the question     
        write(kwrite,100) x,term,sum,ratio
        100 format (3x,F6.1,3(3x,E11.4)) ! formatting the output
        ! 3x = 3space, f6.1 = float of width 6, and 3 other scientific values
    
    end do ! end of loop for eg x = 0.1/1.0/etc
    write(kwrite,*) '   ________________________________________________'
    
    call cpu_time(time_end)
    print*,
    print*, '   The cpu_time = ', time_end - time_begin, 'seconds'
end program expngood
