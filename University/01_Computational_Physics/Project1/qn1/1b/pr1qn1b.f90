!bp: clear; gfortran -Wall pr1qn1b.f90 && ./a.out

!bp: time f90 pr1qn1b.f90 && ./a.out

PROGRAM Hermite

 external Psi
 integer,parameter:: p=16,kwrite=6
 real(kind=16)    ::  x,Psi
 integer          ::  n,i
 character(len=50):: filename
     
       
       
       filename='n30.dat'  !*** change this filename and value of n in below subroutine
       open(unit=kwrite,file=filename,status='unknown')
       
       
      n=30
      x = -10._p 
      do 10 i=1,201 
 
      write(kwrite,10000) x,Psi(n,x)
 
      x=x+0.1_p

 
     10 end do

 10000 format( f10.2,4x, d20.10)
 close(kwrite)

stop 'data saved in outputfile'
end program

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

real(kind=16) function Psi(n,x)


       implicit none
       external factorial
       double precision        :: factorial
       integer,parameter       :: p=16
       
       integer,intent(in)      :: n ! degree of Hermite Polynomial
       real(kind=16),intent(in):: x  ! argument of Hn(x)
       
       real(kind=p)            :: A(0:100)
       real(kind=p)            :: B(0:100,0:100)
       integer                 :: i
       real(kind=16),parameter :: pi=3.141592654       
       real(kind=16)           :: hnx
       
       
     ! initializine A(i) values
     do i=1,100
     A(i) = 0._p
     end do
 
     call Hermite_Coeff(n,A,B)
       
     hnx=0._p
     do i=0,n
     hnx= hnx+A(i)* (x**i)
     end do     
     
     Psi = exp(-0.5_p * x *x )* hnx/ (sqrt ( (2._p**n) *factorial(n) *sqrt(pi)   ))

 
     return
     end function Psi
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       
double precision function factorial(n) 
implicit none 
 
integer, intent(in) :: n 
integer :: i
double precision:: Ans 

Ans = 1 
do i = 1, n 
Ans = Ans * i 
end do 

factorial = Ans 
end function factorial
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Subroutine Hermite_Coeff(n,A,B)
  
  integer,parameter :: p=16  ! for quadruple precision
  integer,intent(in):: n
  real(kind=p)      :: A(0:100) ! A(i) are coeffients of Hermite Polynomial
  real(kind=p)      :: B(0:100,0:100)
  integer           :: i,j


  B(0,0)=1._p ! value of Hn(x) when n=0, x=0 i.e. H0(x=0) = 1
   
  B(1,0)=0._p ! H1(x) = 0 + 2x
  B(1,1)=2._p


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
end subroutine Hermite_Coeff
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
