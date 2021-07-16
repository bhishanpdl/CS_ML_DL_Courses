!Topic     : Assign2 Question1
!Programmer: Bhishan Poudel
!cmd       : clear; gfortran ass2qn1Double.f90 && ./a.out
!cmd       : clear; f90 ass2qn1Double.f90 && ./a.out

!note      : sinx = x - x^3/6 + x^5/120 - .....
!            cosx = 1 - x^2/2 + x^4/24 + .....
!Problem   : Problem 2.3 from Morten Hjorth-Jensens ch.2
!            Computation of x - sin(x) ( bc -l then 99.9 - s(99.9)
!            x = 99.9 y = x - sin(x) = 100.48992416131740726185


program ass2qn1
    implicit none
    double precision    :: x,y
    character(len = 50) :: filename
    integer,parameter   :: kwrite = 6
    
    
    !!!Variable for sine series
      integer                                   :: j
      integer, parameter                        :: maxterm = 5 ! Taylor terms
      double precision                          :: mysum
      double precision                          :: term
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    filename = 'ass2qn1Doube.dat'
    open(unit = kwrite, file = filename, status = 'replace')
    write(kwrite,*) '           #x','                     ', '    x-sin(x)'
    
    do x = -20,20,0.1  ! we can check x = -2,2,0.1
    
    
        ! Using Taylor expansion for |x|< 1.9    
        if(abs(x)<1.9) then  ! eg. x = -20 it is ignored, but 1.8 is affected
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!routine to find the sum of the series
        term = x    
        mysum = x 
     
        j = 1  
        do while ( j < maxterm + 1)   
        
          term = term * (-1*   x * x) / (2 * j * (2 * j + 1))    
          mysum = mysum + term    
          j = j+1
        end do  
        y = x- mysum
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        else
        y = x - sin(x)
        
        end if
        write(kwrite,100) x,y
    end do
    100 format(f20.6,T30,f20.6)
    close(unit = kwrite)

end program ass2qn1
!note:
! m  =         1        2         3 
! x - sinx = -x^3/3! + x^5/5!  - x^7/7! + ..... maxTaylor terms
!                    (-x^3/3!) * -x*x/5*4
!                     5 = 2n+1 and 4 = 2n
!                     mth term = (m-1)term *    -x*x / (2m * (2m+1) )

! table looks like this
! x(i)   y(i)
! -20.0  y = x -sin(x)
! -19.9
! -1.9
! ========================
! -1.8  Taylor expansion x = 1.8 &  y = 1.8^3/3! + 1.8^5/5! - ....
! -1.7
! +1.8
!=========================
! +1.9
! +20.0 etc
!            To open calculator bc -l and sine funtion is s(value), then quit
!            Computation of x - sin(x)
!            x = 2    y = x - sin(x) = 1.09070257317431830461
!            x = 1    y = 0.158529 
!            x = 1.8  y = x^3/3! - x^5/5!     + x^7/7! = 0.8266 and 
!                     3! = 6       5! = 120   7!= 5040
!                     1.8^3        1.8^5      1.8^7
!                     5.832        18.89568   61.2220032
!                     0.972        -0.157464  +0.0121472 = .8266 also, 1.8 - s(1.8) 


