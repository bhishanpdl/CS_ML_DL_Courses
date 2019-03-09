!bp: clear; f90 integration.f90 && ./a.out < grid.dat

!bp: clear; f90 integration.f90 && ./a.out < grid.dat > integration.dat
      program integrate


      sum=0.d0
      do i=1,19
      read(5,*) x,y,w
      
      sum = sum + y*w

      end do
      
      write(6,10000) 'integral value = ' ,sum
      10000 format (A20,F10.2)
      
      end
