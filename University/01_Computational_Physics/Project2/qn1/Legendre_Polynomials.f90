! Bhishan Poudel
! Dec 5, 2015 Sat

! clear; f95 Legendre_Polynomials.f90 && ./a.out ; rm *~ a.out

! This program calculates and print legendre polynomials of order 0 to 8.
! It uses the function pl(x,n) to find the polynomials.

 program Legendre
 implicit none

 double precision x, xmin, xmax, dx
 double precision pl, plg
 integer n,kwrite

 data kwrite/6/

 xmin = -1.d0         ! left point
 dx   =  0.001d0        ! step
 xmax =  1.d0         ! right point

 open (unit=kwrite,file="Legendre_Polynomials.dat",status='replace')
! write(kwrite,*) '# Legendre polynomials for n=',n
! write(kwrite,10000) '#x','Pn(x)'
 
 do n = 0,8               ! order of Pn(x)
 
 write(kwrite,*)
 write(kwrite,*)
 write(kwrite,*)     '# Legendre polynomials for n=',n
 write(kwrite,10000) '#x','Pn(x)'


 x = xmin
 
 

 do while (x < xmax+0.01d0)
     plg = pl(x,n)
     write(kwrite,10001) x, plg
     x = x + dx
 end do


10000 format(2A20)            
10001 format(2f20.7)

end do

 close(kwrite)
 stop 'Data is saved in Legendre_Polynomials.dat'
 end program

!!==============================================================================
     function pl(x,n)
!======================================
! calculates Legendre polynomials Pn(x)
! using the recurrence relation
! if n > 100 the function retuns 0.0
!======================================
double precision pl
double precision x
double precision pln(0:n)
integer n, k

pln(0) = 1.d0
pln(1) = x

if (n <= 1) then
  pl = pln(n)
  else
  do k=1,n-1
    pln(k+1) = ((2.d0*k+1.d0)*x*pln(k) - real(k)*pln(k-1))/(real(k+1))
  end do
  pl = pln(n)
end if
return
end
!!==============================================================================
