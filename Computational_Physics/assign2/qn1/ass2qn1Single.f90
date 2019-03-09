!Topic     : Assign2 Question1
!Programmer: Bhishan Poudel
!cmd       : clear; gfortran ass2qn1Single.f90 && ./a.out
!cmd       : clear; f90 ass2qn1Single.f90 && ./a.out

!note      : sinx = x - x^3/6 + x^5/120 - .....
!            cosx = 1 - x^2/2 + x^4/24 + .....
!Problem   : Problem 2.3 from Morten Hjorth-Jensens ch.2
!            Computation of x - sin(x) ( bc -l then 99.9 - s(99.9)
!            x = 99.9 y = x - sin(x) = 100.48992416131740726185


program ass2qn1
    implicit none
    integer,parameter   :: kwrite = 6
    real                :: x,y
    character(len = 50) :: filename
    
    
    !!!Variable for sine series
      integer              :: j
      integer, parameter   :: maxterm = 5 ! Taylor terms
      real                 :: mysum
      real                 :: term
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    filename = 'ass2qn1Single.dat'
    open(unit = kwrite, file = filename, status = 'replace')
    write(kwrite,*) '           #x','                     ', '    x-sin(x)'
    
    do x = -20,20,0.1  ! we can check x = -2,2,0.1
        y = x - sin(x)
        write(kwrite,100) x,y
    end do
    100 format(f20.6,T30,f20.6)
    close(unit = kwrite)

end program ass2qn1

