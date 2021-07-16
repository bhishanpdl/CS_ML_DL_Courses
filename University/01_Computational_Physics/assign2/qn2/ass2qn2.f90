! Author:  Bhishan Poudel
! Date  ;  Sep 3, 2015
! Topic : Assessment 2, qn 2 (problem 2.9 of Morten Hjorth-Jensen ch.2 page 39)
! cmd   : clear; gfortran -Wall ass2qn2.f90 && ./a.out
! cmd   : clear; f90 ass2qn2.f90 && ./a.out


! quadratic eq: http://www.math.com/students/calculators/source/quadratic.htm
! problem     : roots of x^2 + 4x -1 = 0  ans: 0.2360679774997898 and -4.23606797749979
!           x    = 1/ (4 + x)  gives first  +ve root x1=  0.23607
!           x    = -4 + 1/x    gives second -ve root x2= -4.23607
! note: 4th iteration gives correct answer upto 5 digits after decimal

program ass2qn2
    implicit none
    double precision , dimension(20) :: x,y    ! 1d arrays with 20 elements
    integer :: i                               ! counter for iteration
    integer,parameter :: kread = 5, kwrite = 6 ! standard values
    
    x(1) = 0 ! initialize first value ( caveat: CANNOT BE -4 )
    y(1) = 1 ! initialize first value ( caveat: CANNOT BE 0  )
    
    
    open(unit=kwrite,file='ass2qn2.dat',status='replace')
    write(kwrite,*) '#---------------------------------'
    write(kwrite,*) '#counter','  ', 'first root', '   ', 'second root' 
    write(kwrite,*) '#---------------------------------'                              

    do i=2,20
        x(i) = 1/(4+x(i-1))
        y(i) = -4 + 1/y(i-1)
        
        write(kwrite,100) i, x(i),y(i)
    end do
    100 format (T2, I4, T10, F8.5, T24, F8.5)
    write(kwrite,*) '#---------------------------------'
    close(kwrite)

end program ass2qn2 

! x^2 + 4x = 1
! x(x+4)   = 1
! x        = 1/ (4 +x)  gives positive root ( DO NOT	 choose x = -4 as initial value)

! x+4      = 1/x        gives negative root
! x        = -4 + 1/x  ( DO NOT	 choose x = 0 as initial value) 


