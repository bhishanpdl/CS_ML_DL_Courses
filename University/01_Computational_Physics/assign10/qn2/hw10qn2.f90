!cmd: clear; f95 hw10qn2.f90 && ./a.out; rm -f a.out *~															 

			
	program
	Implicit none
	
	integer ::i, j, max, element, prop(100),iflag,seed
	real*8  ::change, drand, energy, newE, oldE, out, path(100)
	
	
	max	 = 250000
	open(9, FILE='hw10qn2a.dat', Status='Unknown')
	open(10, FILE='hw10qn2b.dat', Status='Unknown')
	
    ! plant seed
    iflag=1
    seed=drand(iflag)
    iflag=0
    			 
    ! initial path and initial probability 
	do 10	j=1,100
		path(j)=0.0
		prop(j)=0 
    10 end do
		
    ! find energy of initial path    
	oldE = energy(path, 100)
    do 20	i=1,max

        ! pick one drandom element
		element = drand(iflag)*100+1
		!write(9,*) element
		
        ! change it by an drandom value -0.9..0.9
			change	= ((drand(iflag)-0.5)*2)
			path(element)=path(element)+change
			
			
        ! find the new energy
		newE=energy(path, 100)
		
        ! reject change if new energy is greater and the Boltzmann factor
        ! is less than another drandom number 
					if ((newE>oldE) .AND. (exp(-newE+oldE)<drand(iflag))) then
						 path(element)=path(element)-change
					endif
					
					
        ! add up probabilities
		do 30	j=1,100
			element=path(j)*10+50						 
			prop(element)=prop(element)+1
	    30 end do
		oldE = newE
     20 end do
 
 
    ! write output data to file
	do 40	j=1,100
		out=prop(j)
		write(9,10000) j-50, sqrt(out/max)		
    40 end do
	
	
	! write output data to file
	do	j=1,100
		
		write(10,10000) j, path(j)
	end do
	10000 format(i4,f14.6)
	
	close(9)
	close(10)
	stop 'data saved in hw10qn2a.dat and hw10qn2b.dat'
	end program



!
! function calculates energy of the system
	function energy(array, max)
	implicit none
	
	integer ::i, max
	real*8  ::energy, array(max)
	
	energy=0
	do	i=1,(max-1)
		 energy=energy + (array(i+1)-array(i))**2 + array(i)**2
	end do
	return
	end
		
