
!bp: clear; gfortran -w pr1qn2pot.f90 && ./a.out
!bp: clear; gfortran pr1qn2pot.f90 && ./a.out


program pr1
implicit none
external V
double precision  :: V
double precision  :: r
integer           :: n
integer,parameter :: kwrite=6
character(len=50) :: outputfile


outputfile='pr1qn2pot.dat'  ! *** CHANGE
open(unit=kwrite,file=outputfile,status='unknown')

     do r = 0.001d0, 0.14, 0.001d0
     
       write(kwrite,10000) r, V(r)
     
     end do
    
     10000 format(f10.3,e20.6)      !(to check correctness)
     
     close(kwrite)
     stop 'data is saved in outputfile'
     

end program
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


double precision function V(r)
implicit none
  
  double precision,intent(in) :: r
  double precision            :: x,r_inv
  double precision,parameter  :: tolerance=1e-6
  
  r_inv= 1.d0/r
  x  = -7.559d0 * r
  
  
  V= 28.80d0  * r_inv * exp(x)
  
  ! Taylor exansion for small x
  ! e^x = 1 + x + x^2/2 + x^3/3 + ...
  
  if(r.lt.tolerance) then
  
  V = 28.8d0 * r_inv * (1+ x+ x*x/2.d0 + x*x*x/3.d0)
  
  end if 
  
  
  
  return 
end function



