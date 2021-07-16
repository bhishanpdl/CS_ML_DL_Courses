!Bhishan Poudel
!Oct 19, 2015

!bp: clear; f90 hw8qn1abc.f90 && ./a.out


! Purpose:
!   This program generates random numbers using the linear congruent method  
!   given equation: ri ≡ (a ri−1 + c) mod m                                                            



program random_numbers

	implicit none
	integer, parameter      :: b = 1000, kread=5,kwrite1=6,kwrite2=8
	integer                 :: r0, a, c, i, m, j
    integer, dimension(b+1) :: r
	 				
!   supplying values to parameters
	a=57;c=1;m=256;r0=10

!   printing random numbers     
	open(kwrite1, file = 'hw8qn1a.dat', status = 'unknown')		
	  write(kwrite1,*)'#         i           r(i) '
	  r(1) = r0
	  do i = 1,m
	     r(i+1) = mod((a*r(i)+c),m)	!bp: r(i) ≡ [a*r(i−1) + c] mod M 
	     write(kwrite1,10000)i, r(i)
	  end do

!   printing r(i) and r(i+1) for the plot
	open(kwrite2, file = 'hw8qn1c.dat', status = 'unknown')
	  do j = 1,(m+1)/2
	    write(kwrite2,10000) r(2*j-1), r(2*j)
	  end do

      10000 format(2i12)		
        
	close(kwrite1)
	close(kwrite2)
	
stop 'data saved in output file'
end program
