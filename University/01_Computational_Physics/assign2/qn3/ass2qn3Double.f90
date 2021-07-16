! Author:  Bhishan Poudel
! Date  ;  Sep 3, 2015
! Topic : Assessment 2, qn 3 (topic  3.8 of Landau 2E ch.3 page 38)
! cmd   : clear; gfortran -Wall ass2qn3Double.f90 && ./a.out
! cmd   : clear; f90 ass2qn3Double.f90 && ./a.out

! Bessel functions via recursion
!=================================
! fn-1(x) +  fn+1(x) =  (2n + 1/x) fn(x)   Arfken7E page. 702 or, Landau2E page 35
! In these equations f(n) may represent j(n), y(n), h1(n) , or h2(n)
! jl+1(x) =  (2l + 1/x) jl(x) - jl-1(x) put l = 1 for j2    UPWARD
! jl-1(x) =  (2l + 1/x) jl(x) - jl+1(x) put l = 3 for j2    DOWNWARD

! j0(x) = sin(x) / x  (Stegun page 457)
! j0(x) = sin(x)/x**2 - cos(x)/x
! j2(x) = (2*1+1/x) j1(x) - j0(x) l =1  upward recursion we get j2 from j1 and j0
! j2(x) = (2*3+1/x) j3(x) - j4(x) l = 3 dnward recursion we get j2 from j3 and j4
 
! j0(x) = 1.0000, 0.99833, 0.84147, -5.4402e-2  (when x = 0,0.1,1,10)
! j1(x) = 0.0000, 0.03330, 0.30116, +7.8467e-2  (when x = 0,0.1,1,10) 

program BesselRecursion
    implicit none
    integer,parameter :: kwrite = 6  ! standard value
    double precision  :: jup(0:25)   ! j0 to j25 upward (use single precision to see effect earlier)
    double precision  :: jdown(0:50) ! j0 to j25 downward
    double precision  :: rd(0:25)    ! relative difference betwn jup and jdown
    double precision  :: rdup(0:25)  ! relative difference betwn jup and jtrue
    double precision  :: rddn(0:25)  ! relative difference betwn jdn and jtrue
    integer           :: l           ! do loop counter
    double precision  :: x           ! x = .1 or 1.0 or 10.0 we can choose
    double precision  :: myscale     ! for normalizing downward recursion
    double precision  :: jtrue(0:8)  ! j0(1.8) to j8(1.8) values
    
    
    !***** for x = 0.1
    jtrue(0) =	0.99833
    jtrue(1) =	0.03330				
    jtrue(2) = 	0.00067		
    jtrue(3) =	9.5185e-6		
    jtrue(4) =	1.0577e-7
    jtrue(5) =	9.6163e-10
    jtrue(6) = 	7.3975e-12		
    jtrue(7) =  4.9319e-14
    jtrue(8) =	2.9012e-16
    !!!!!!!!!!!!!!!!!!!!!!! NOTE: if x changes All RELATIVE VALUES ARE CHANGED

    
    x = 0.1d0    !!WARNING: WHEN X CHANGES, MYSCALE AND JDOWN(0) CHANGES  ! ******* CHANGE *****
    !x = 1.d0      !! look at line 49,55, 101, 103 & 137
    !x = 10.d0

    !************* Upward Recursion **************
    
    open (unit=1, file ="jup_A.dat",status = 'replace')  !*********** CHANGES *********
    write(1,100)
    !***** Taking care of dividing by zero
    if (x == 0) then 
        jup(0) = 1.d0
        jup(1) = 0.d0
    else
        jup(0) = sin(x)/x
        jup(1) = sin(x)/x**2 - cos(x)/x
    end if 
    !***** end of taking care
    write(1,110) x,0,jup(0)
    write(1,110) x,1,jup(1)
        
    do l = 1,24 !eg. l = 1 
        
        jup(l+1) = ( (2.d0*real(l)+ 1.d0)/x ) * jup(l)  - jup(l-1) ! we get j2
        write(1,110) x,l+1,jup(l+1)
        100  format(T4, '#x', T12, 'l', T18, 'jup value')
        110 format(T2, F5.2, T9, I4, T14, E16.6)
        
    end do 
    close(1)
    
    !************* Downward Recursion Unnormalized **************
    ! initialize some values(we will normalize later)
    open(unit = 2, file = 'jdown_unnormalized_A.dat', status='replace')
    write(2,120)
    jdown(51) = 2.d0 ! this value doesnot change ( if it is >= 19, we get floating point exception)
    jdown(50) = 1.d0 ! this value doesnot change (Note: take diff values)

    do l = 50, 1,-1 !eg. l=50
        jdown(l-1) = ((2.d0*real(l)+1.d0)/x)*jdown(l)  - jdown(l+1) ! we get l-1 = 49
        write(2,130)x,l-1,jdown(l-1)
          
    end do
    120  format(T4, '#x', T12, 'l', T18, 'jdown unnormalized value')
    130 format(T2, F5.2, T9, I4, T14, E16.6)    
    close(2)
    
       
    !************* Downward Recursion Normalized **************
    ! here we get some value of j0(x) = 1.985426E+38
    ! for x = 0.1, we know that j0(x) = 0.99833
    ! scale = jdown(0)/0.99833 ********************************
    !print*, jdown(0)
    open(unit = 3, file = 'jdown_A.dat', status='replace')     ! ******** CHANGE *****
    write(3,140)
    myscale = 0.99833/jdown(0)     !!WARNING: THIS VALUE CHANGES WHEN X CHANGES   ! ****** CHANGE ****
    !myscale = 0.84147/jdown(0)     !!WARNING: THIS VALUE CHANGES WHEN X CHANGES
    !myscale = -5.4402e-2/jdown(0)  !!WARNING: THIS VALUE CHANGES WHEN X CHANGES
    
    
        do l = 0, 25
            jdown(l) = jdown(l) * myscale
            write(3,150)x,l,jdown(l)
          
        end do
       
        140  format(T4, '#x', T12, 'l', T18, 'jdown_normalized_value')
        150 format(T2, F5.2, T9, I4, T14, E16.6)    
    close(3)
    
    
    !!*************** Relative difference w.r.t true values *******************
    open(unit = 4, file = 'rd_wrt_true_A.dat', status='replace')
    write(4,160)
    
    rdup(0) = 0 ! intializing value
        
        do l = 0,8 
        
            rdup(l) = abs(jup(l)-jtrue(l))/(abs(jtrue(l)))
            rddn(l) = abs(jdown(l)-jtrue(l))/(abs(jtrue(l)))
            write (4,170) x, l, jtrue(l), jup(l), jdown(l), rdup(l), rddn(l) 
        end do 
        160  format(T4, '#x', T12, 'l', T18, 'jtrue(x)', T34, 'jup(x)', T54, 'jdown(x)', T74, 'rd_for_up', T94,'rd_for_dn')
        170 format(T2, F5.2, T9, I4, T14, E16.6,T30, E16.6,T50, E16.6,T70, E16.6,T90, E16.6)    
    close(4)

    !!*************** Relative difference *******************
    
    open (unit=5, file ="relative_diff_A.dat",status = 'replace')  ! **** CHANGE *********
    write (5,*) "#----------------------------------------------------------------"
    write (5,*) "#   x     l       jup           jdown               relative difference "
    rd(0) = 0 ! intializing value
        
    do l = 0,25 
        rd(l) = abs(jup(l)-jdown(l))/(abs(jup(l))+abs(jdown(l)))
        write (5,180)x, l,jup(l),jdown(l), rd(l) 
    end do 
    write (5,*) "#----------------------------------------------------------------"
    180 format(T2, F5.2, T9, I4, T14, E16.6,T30, E16.6, T50, E16.6)       
    close(5) 

end program BesselRecursion



