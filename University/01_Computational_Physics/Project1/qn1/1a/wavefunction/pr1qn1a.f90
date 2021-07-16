!bp: clear; gfortran -Wall pr1qn1a.f90 && ./a.out

PROGRAM Hermite

external Psi
real(kind=16):: Psi
integer,parameter:: p=16,kwrite=6
real(kind=16)::  x
integer      ::  n
double precision:: y
character(len=50):: outputfile

 outputfile='n1.dat'  !*** CHANGE this and value of n below 

 open(unit=kwrite,file=outputfile,status='unknown')
 
 do y=-4.d0,4.d0,0.01d0
 x=y

 write(kwrite,10000) n,x,Psi(x)
 
 end do 
 
 
 10000 format(i4,f10.2,e14.6)
 close(kwrite)
 print*, 'data is saved in  ', outputfile


 end

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
real(kind=16) function Psi(x)


       implicit none
       integer,parameter:: p=16
       real(kind=p)::  A(0:100)
       real(kind=p)::  B(0:100,0:100)
       integer     ::  n,i,factorial
       real(kind=16),parameter :: pi=3.141592654
       
       real(kind=16) :: x,hnx
       
       
     ! initializine A(k) values
     do i=1,100
     A(i) = 0._p
     end do

     n = 1   !*** CHANGE  
     call Hermite_Coeff(n,A,B)
     
       
     hnx=0._p
     do i=0,n
     hnx= hnx+A(i)* (x**i)
     end do
     
     
     
     Psi = exp(-x*x*0.5_p)*hnx/ sqrt (   (2._p**n) *factorial(n) *sqrt(pi)   )
      
 
       return
       end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       
INTEGER FUNCTION factorial(n) 
IMPLICIT NONE 
 
INTEGER, INTENT(IN) :: n 
INTEGER :: i, Ans 

Ans = 1 
DO i = 1, n 
Ans = Ans * i 
END DO 

factorial = Ans 
END FUNCTION factorial
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Subroutine Hermite_Coeff(n,A,B)
  
  integer           :: i,j,n
  integer,parameter :: p=16
  real(kind=p)      :: A(0:10), B(0:10,0:10)


  B(0,0)=1._p ; B(1,0)=0._p ; B(1,1)=2._p


  if (n>1) then
    do i = 2, n
      B(i,0)=-2._p*(i-1)*B(i-2,0)
      do j = 1, i
      
        !recursion relation
        B(i,j)=2._p*B(i-1,j-1)  -  2._p*(i-1)*B(i-2,j)
      end do
    end do
    do i = 0, n
      A(i)=B(n,i)
    end do
  end if
  
  if(n.eq.0) then
  A(0) = 1._p
  end if
  
  if(n.eq.1) then
  A(0)= 0._p
  A(1)= 2._p
  end if
  
  
  return
end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




