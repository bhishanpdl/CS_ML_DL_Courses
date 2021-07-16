! cmd:  clear; gfortran -Wall hw4qn3b.f90 && ./a.out
! cmd:  clear; f90 hw4qn3b.f90 && ./a.out

!        calculate the population growth


      program hw4qn3b

      implicit none

      double precision :: n0, dt, a,b
      double precision :: n(1000), t(1000), ntrue(1000), error(1000) 

      integer :: j,tmax

      integer :: kwrite

      data kwrite/6/

!        initialize


      n0   = 1000 ! initial population
      a    = 10.d0 ! growth rate
      b    = 3.d0  ! death rate
 
      dt = 0.0001
      tmax = 1000

      t(1) =0.d0
      n(1)=n0
      ntrue(1)=n0
     
!        propagate solution in time

      do j=2,tmax
        t(j) = t(j-1) + dt      ! increasing time
        
        ! N(t+dt) = N(t)   + (aN-bN^2) dt
        n(j)      = n(j-1) + (a* n(j-1) - b * n(j-1)* n(j-1)) * dt
        
        !ntrue =        N0 a exp(at)
        !            ----------------------------
        !             a-bN0 + bN0 exp (at) 
        ntrue(j) = ntrue(1)*a*exp(a*t(j)) / ( a - b*n(1) + b*n(1) * exp(a*t(j)))
        
        error(j) = abs(n(j) - ntrue(j))/abs(ntrue(j))
      end do 

!        write output

      open (kwrite, File = 'hw4qn3b.dat', Status = 'Unknown')

      write (kwrite,*) 
      write (kwrite,10000) '#t(days)','N(t)','   N(t) True','  error' 

      do j=1,tmax
      write (kwrite,20000) t(j),n(j),ntrue(j),error(j)
      end do     
      
!      end do

10000 format(4A14) 
20000 format(4E14.6)
      close(kwrite)

      stop 
      
      end program hw4qn3b
