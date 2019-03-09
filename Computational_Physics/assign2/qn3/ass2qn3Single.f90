! Author:  Bhishan Poudel
! Date  ;  Sep 3, 2015
! Topic : Assessment 2, qn 3 (topic  3.8 of Landau 2E ch.3 page 38)
! cmd   : clear; gfortran -Wall ass2qn3Single.f90 && ./a.out
! cmd   : clear; f90 ass2qn3Single.f90 && ./a.out

! Bessel functions via recursion
!=================================
! fn-1(x) +  fn+1(x) =  (2n + 1/x) fn(x)   Arfken7E page. 702 or, Landau2E page 35
! In these equations f(n) may represent j(n), y(n), h1(n) , or h2(n)
! jl+1(x) =  (2l + 1/x) jl(x) - jl-1(x) put l = 1 for j2
! jl-1(x) =  (2l + 1/x) jl(x) - jl+1(x) put l = 3 for j2

! j0(x) = sin(x) / x  (Stegun page 457)
! j0(x) = sin(x)/x**2 - cos(x)/x
! j2(x) = (2*1+1/x) j1(x) - j0(x) l =1  upward recursion we get j2 from j1 and j0
! j2(x) = (2*3+1/x) j3(x) - j4(x) l = 3 dnward recursion we get j2 from j3 and j4
 
! j0(x) = 1.0000, 0.99833, 0.84147, -5.4402e-2  (when x = 0,0.1,1,10)
! j1(x) = 0.0000, 0.03330, 0.30116, +7.8467e-2  (when x = 0,0.1,1,10) 

program BesselRecursion
    implicit none
    integer,parameter :: kwrite = 6  ! standard value
    real              :: jup(0:25)   ! j0 to j25 upward (use single precision to see effect earlier)
    real              :: jdown(0:50) ! j0 to j25 downward
    real              :: rd(0:25)    ! relative difference betwn jup and jdown
    real              :: rdup(0:25)  ! relative difference betwn jup and jtrue
    real              :: rddn(0:25)  ! relative difference betwn jdn and jtrue
    integer           :: l           ! do loop counter
    real              :: x           ! x = .1 or 1.0 or 10.0 we can choose
    double precision  :: myscale     ! for normalizing downward recursion
    double precision  :: jtrue(0:8)  ! j0(1.8) to j8(1.8) values
    
    jtrue(0) =	0.99833
    jtrue(1) =	0.03330				
    jtrue(2) = 	0.00067		
    jtrue(3) =	9.5185e-6		
    jtrue(4) =	1.0577e-7
    jtrue(5) =	9.6163e-10
    jtrue(6) = 	7.3975e-12		
    jtrue(7) =  4.9319e-14
    jtrue(8) =	2.9012e-16

    
    x = 0.1d0
  
    open (unit=kwrite, file ="bessel.dat",status = 'replace')

    write (kwrite,*) "#----------------------------------------------------------------"
    write (kwrite,*) "#x     l         jup            jdown         relative difference "
    write (kwrite,*) "#----------------------------------------------------------------"

    !************* Upward Recursion **************
    !***** Taking care of dividing by zero
    write(kwrite,*) 'jup values'
    if (x == 0) then 
        jup(0) = 1.d0
        jup(1) = 0.d0
    else
        jup(0) = sin(x)/x
        jup(1) = sin(x)/x**2 - cos(x)/x
    end if 
    !***** end of taking care
    write(kwrite,*) x,0,jup(0)
    write(kwrite,*) x,1,jup(1)
        
    do l = 1,19 !eg. l = 1 (when we reach l = 20, we get floating point exception)
        
        jup(l+1) = ( (2.d0*real(l)+ 1.d0)/x ) * jup(l)  - jup(l-1)
        write(kwrite,*) x,l+1,jup(l+1)
        
    end do 
!    
    
    !************* Downward Recursion Unnormalized **************
    ! initialize some values(we will normalize later)
    write(kwrite,*) 'Unnormalized jdown values'
    jdown(18) = 10.d0 ! this value doesnot change ( if it is >= 19, we get floating point exception)
    jdown(17) = 9.d0 ! this value doesnot change (Note: take diff values)

        do l = 17, 1,-1 !eg. l=49
            jdown(l-1) = ((2.d0*real(l)+1.d0)/x)*jdown(l)  - jdown(l+1) ! we get l = 48
            write(kwrite,*)x,l-1,jdown(l-1)
          
        end do
        
    !************* Downward Recursion Normalized **************
    ! here we get some value of j0(x) = 1.985426E+38
    ! for x = 0.1, we know that j0(x) = 0.99833
    ! scale = jdown(0)/0.99833 ********************************
    write(kwrite,*) 'Normalized jdown values'
    myscale = 0.99833/jdown(0)
    
        do l = 16, 0,-1 !eg. l=16
            jdown(l) = jdown(l) * myscale
            write(kwrite,*)x,l,jdown(l)
          
        end do
        
    !!*************** Relative difference for upward*******************
    write(kwrite,*) 'relative differences'
    rdup(0) = 0 ! intializing value
        
        do l = 0,8 
        
            rdup(l) = abs(jup(l)-jtrue(l))/(abs(jtrue(l)))
            rddn(l) = abs(jdown(l)-jtrue(l))/(abs(jtrue(l)))
            write (kwrite,*) x, l, rdup(l), rddn(l) 
        end do 


    !!*************** Relative difference *******************
    rd(0) = 0 ! intializing value
        
    do l = 0,8 
        rd(l) = abs(jup(l)-jdown(l))/(abs(jup(l))+abs(jdown(l)))
        write (kwrite,*) l, rd(l) 
    end do 

           
    close(kwrite) 

end program BesselRecursion



