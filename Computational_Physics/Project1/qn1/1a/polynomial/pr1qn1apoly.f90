
!bp: clear; gfortran -w pr1qn1apoly.f90 && ./a.out
!bp: clear; gfortran pr1qn1apoly.f90 && ./a.out


program Hermite

integer           :: n
integer,parameter :: kwrite=6,p=16
real(kind=p)      :: hnx,x
character(len=50) :: outputfile


outputfile='n12.dat'  ! *** CHANGE
open(unit=kwrite,file=outputfile,status='unknown')

     !do 10 x= -4._p,4._p,0.01_p  ! for n=1,2,3 with x=-4 to 4
     do 10 x= 0._p,10._p,1._p     ! for checking correctness for n=5,12 and x=3,12 with Abramowitz
     
     
     n = 12      ! *** CHANGE

     call H(n,x,hnx)
     write(kwrite,10000) n,x, hnx
     
     10 end do
    
     !10000 format(i4,f10.2,f20.10)       ! 10 significant figures
     10000 format(i4,f10.1 ,es20.10)      !(to check correctness)
     
     close(kwrite)
     stop 'data is saved in outputfile'
     

end program
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!







!******************************************************
!* Hermite polynomial coefficients
!* Hn(x) = a0 +a1 x + a2 x^2 + a3 x^3 + .... + an x^n                
!* n     = order of polynomial
!* x     = x value
!* hnx   = Hn(x) value
!******************************************************

Subroutine H(n,x,hnx)
  
  integer           :: i,j,n
  integer,parameter :: p=16                ! quadruple precision
  real(kind=p)      :: A(0:100)
  real(kind=p)      :: B(0:100,0:100)
  real(kind=p)      :: x,hnx
  real(kind=p),parameter:: tolerance=1e-16
  
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! avoid erro at x = 0
if(abs(x)<tolerance) then
x = 0._p
end if

!if we do not do this we will get this error:
!! example: when n=1, x = 0._p, then H(x) = -7.1108944665E-32
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
  
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! for H(n) when n>1
  ! initializing values
  B(0,0)=1._p                ! H0(x) = 1
  B(1,0)=0._p ; B(1,1)=2._p  ! H1(x) = 0 + 2x  
  
  hnx =0._p  
  !Return if order is less than two ( second last line)
  if (n>1) then
    do i = 2, n
      B(i,0)=-2._p*(i-1)*B(i-2,0)
      do j = 1, i
      
        !Recursion relation
        ! H(n+1) = 2xH(n) - 2n H(n-1)
        ! H(n)=    2x     H(n-1)        - 2       n      H(n-2)
        B(i,j)=    2._p * B(i-1,j-1)    - 2._p * (i-1) * B(i-2,j)
      end do
    end do
    do i = 0, n
      A(i)=B(n,i)
      hnx= hnx + A(i)* x**i
    end do
  end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! for H0(x)    
  if(n.eq.0) then
  hnx = 1._p
  end if
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! for H1(x)
  
  if(n.eq.1) then
  hnx = 2._p * x
  end if
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  
  return 
end



