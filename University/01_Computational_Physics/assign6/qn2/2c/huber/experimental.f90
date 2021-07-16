!bp: clear; f90 experimental.f90 && ./a.out

! The equation is f(E)=(68658.34)/( (E-75)^2 + 822.26 )
! The derivation of this equation is given in writeup

! usage: hw6 qn2b


program experimental
 
       double precision  :: f,E
       integer,parameter :: kwrite = 6
        
       f = 0.d0
       
       open(unit=kwrite,file='experimental05.dat',status='unknown')
       do E=0,200,5
     
           f = (68658.34d0)/( (E-75)**2 + 822.26)
      
           write(6,10000) E,f
              
       end do
       close(kwrite)
       10000 format(1x,f7.2,3x,f5.2)
       stop 'data saved in output file'
end program
