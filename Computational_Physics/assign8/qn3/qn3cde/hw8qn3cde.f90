!Bhishan Poudel
!Oct 22,2015

!bp: clear; f90 hw8qn3cde.f90 && ./a.out

!note: plot of r vs. sqrt(n) looks like Landau2E page 147(161)

      Program walk
      implicit none

      integer          :: np,ntrial
      parameter (np=1000,ntrial=1000)

      integer          :: i, it, j, max, iflag
      double precision :: drand, root2, theta, x, y, r(np), dx, dy
      double precision :: xi(np), yi(np)

      integer          :: kread, kwrite, kpunch

      data kread/5/, kwrite/6/, kpunch/20/


!        set parameters (# of steps)

      max   = np
      iflag = 30
      
!     open files
      open (kwrite, FILE='hw8qn3c3.dat',  Status='Unknown')
      !open (kpunch, FILE='hw8qn3ab.dat', Status='Unknown')

!         plant seed

      r(1)=drand(iflag)  
      iflag=0 

!         clear array
      do  j=1, max
      r(j) = 0.d0
      end do

!         average over ntrials
      do  j = 1, ntrial
          x = 0.d0
          y = 0.d0

!         take max steps
        do  i = 1, max
            dx = (drand(iflag)-0.5d0)      ! choosing random numbers
            dy = (drand(iflag)-0.5d0)
            x = x + dx/sqrt(dx*dx + dy*dy) ! normalizing dx and dy
            y = y + dy/sqrt(dx*dx + dy*dy)
            r(i) = r(i)+ sqrt(x*x + y*y)
            xi(i)=x
            yi(i)=y
        end do
       end do

      

!        output data for plot of r vs. sqrt(N) and the actual walk

       do i = 1, max
          write (kwrite,10000) sqrt(dble(i)), r(i)*0.001d0
          !write (kpunch,10000) xi(i)*0.001, yi(i)*0.001  ! divide by ntrial to get average
       end do
       
       10000 format(2f10.3)  ! three significant figures for sqrt(n)
       close(kwrite)
       close(kpunch)

       stop 'data saved in output file'
       end program
