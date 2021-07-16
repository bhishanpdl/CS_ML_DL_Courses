!bp: clear; f95 hw9qn2c.f90 && ./a.out; rm -f fort.8 *~

!time ./a.out 
!real	0m0.045s
!user	0m0.040s
!sys	0m0.008s


      Program THREE_D
      

!        Example of three-dimensional Monte Carlo integration 
!        to find the total charge in a cubic box with exponential distribution
!        rho(x,y,z)=1/(1+x^2) * y*exp(-y^2) * exp(-z)/sqrt(z)

      implicit double precision(a-h,o-z)

      parameter (eps=2.0e-30,iter_max=15,pi=3.1415926)
      parameter (ap=0.0d0,b=10.0,const = pi/2.0d0)

      dimension iseed(10),xi(3)

      data (iseed(j),j=1,9)/234579,90911,1237,343,39,59,10349,511,101/
      
      rho(xa,ya,za)= pi/4.d0 * (1-exp(-1.d0))**2 * ya * exp(-ya*ya+ya) /sqrt(za)
      
      open(unit=kwrite,file='hw9qn2c.dat',status='unknown')
      write(kwrite,*) '#iteration','   1/n', '           result','      n'


!        Set EPS as the maximum error allowed.
!        Set ITER_MAX as the maximum number of iterations.
!        Define the integrand as a function $\rho(x,y,z)$.

      

!        box normalization

100   a=1.0d0

2     format (A)

      if (a.le.0.0) then
            stop 'Normal end of THREE_D'
       endif

      cnst=a**3
       n=1

!        Calculate the integral for $N$ random points:
!        zero the iteration counter
       iter=0
300    sum_n=0.0

!        initialize sobel sequence
       call sobseq(-1,xi,iseed(iter+1))

       do 390 i=1,n

!        Generate three random numbers with uniform distribution and use
!        them as the values of $x$, $y$, and $z$.

       call sobseq(3,xi,iseed(iter+1))

       x=xi(1)
       y=xi(2)
       z=xi(3)

!        introduce weights

       xp=  tan(pi*x/4.d0) 
       yp= -1.d0* log(1.d0 - (1-exp(-1.d0))*y) 
       zp= -1.d0* log(1.d0 - (1-exp(-1.d0))*z) 
       

!        Calculate the value of the integrand at the point $(x,y,z)$
!        and add the value to the sum.

       sum_n=sum_n+(rho(xp,yp,zp))
390    continue

!        Convert the sum to the value of the integral using (2-42) 

       rslt=(cnst*sum_n/float(n))
       write(kwrite,10000)iter,1.d0/dble(n),rslt,n
10000  format (i10,4x,e10.3,f10.2,i8)

!        Compare the two values of the integral obtained.

       if (iter.lt.iter_max) then

!        If the difference is larger than $\epsilon$ then
!        take the average of the two values.

       rslt_old=(rslt_old+rslt)/2.0

!        Double the value of $N$
      n=n+n

!        Increase the iteration counter by 1 

      iter=iter+1

!        Repeat the calculation for another N points
      go to 300
      endif
      stop 'data is saved'
      end program
      
!============================================================      
      subroutine sobseq(n,x,iseed)
!
!         quasi-random seqence  (num.rec. p 302)
!
!         when n is negative, routine internally initializes 
!         a set of maxbit direction numbers for each of maxdim 
!         different sobol' sequences. 
!         when n is positive (but < maxdim), returns as the vector
!         x(1,...n) the next values from n of these sequences. 
!         n must not be changed betweeen initializations.
!
!  **** NOW CHANGED TO RE-INITIALIZE WITHIN PROGRAM

      implicit double precision(a-h,o-z)
 
      parameter (maxbit=30, maxdim=12)
 
      integer ip(maxdim), mdeg(maxdim),ix(maxdim)
      integer iv(maxbit*maxdim), iu(maxdim,maxbit)
      dimension x(*)
      save ip,mdeg,ix,iv,in,fac
      equivalence (iv,iu)
!        initialize
      if (n.lt.0)  then 

          call sobseq_init(ip,mdeg,ix,iv,in,fac,iu,iseed)
      else
!
!        calculate next vector in sequence
!
      im=in
 
      do  j=1,maxbit
!        find rightmost zero bit
      if (iand(im,1).eq.0) go to 1
      im=im/2
      end do 
 
      pause 'maxbit too small in subroutine sobseq'
 
    1 im=(j-1)*maxdim
 
      do 16 k=1,min(n,maxdim)
      ix(k)=ieor(ix(k),iv(im+k))
      x(k)=ix(k)*fac
   16 continue
 
!        increment counter
      in=in+1
 
      endif
 
      return 
      end 

!============================================================

      subroutine sobseq_init(ip,mdeg,ix,iv,in,fac,iu,iseed)

      implicit double precision(a-h,o-z)

      parameter (maxbit=30, maxdim=12)
 
      integer ip(maxdim), mdeg(maxdim),ix(maxdim)
      integer iv(maxbit*maxdim), iu(maxdim,maxbit)
      
      integer ip2(maxdim), mdeg2(maxdim),ix2(maxdim)
      integer iv1(maxdim,maxdim)

!        for 1D and 2D addressing
      save ip2,mdeg2,ix2
 
      data ip2/0,1,1,2,1,4,2,4,7,11,13,14/
      data mdeg2/1,2,3,3,4,4,5,5,5,5,5,5/, ix2/12*0/

      do 150 k=1,maxdim
          ip(k)=ip2(k)
          mdeg(k)=mdeg2(k)
          ix(k)=ix2(k)
        do 160 j=1,maxdim
             iv1(k,j)=-99
 160    continue
 150   continue

        call ivset(iv1,mdeg,maxdim,iseed)
 
        write(8,*)' After call to ivset'
 
          do i = 1,12
          write(8,101) ( iv1(i,j),j=1,12)
          end do
 
        call ivcalc(iv1,mdeg,ip,maxdim)
 
        write(8,*)' After call to ivcalc'
 
          do i = 1,maxdim
          write(8,101) ( iv1(i,j),j=1,12)
 101      format(12(i5,','))
            do j = 1,maxdim
            iv(12*(j-1)+i) = iv1(i,j)
            iu(i,j)=iv1(i,j)
            end do
          end do
 
          do i = maxdim**2+1,maxbit*maxdim
          iv(i) = 0
          end do

      do 14 k=1,maxdim
!
!        normalize stored values 
!
        do j = 1,mdeg(k) 
        iu(k,j)=iu(k,j)*2**(maxbit-j)
        end do
!        use recurrence to get other values
      do 13 j=mdeg(k)+1,maxbit
      ipp=ip(k)
      i=iu(k,j-mdeg(k))
      i=ieor(i,i/2**mdeg(k))
 
      do 12 l=mdeg(k)-1,1,-1
      if (iand(ipp,1).ne.0) i=ieor(i,iu(k,j-l))
      ipp=ipp/2
   12 continue
 
      iu(k,j)=i
   13 continue
   14 continue
 
      fac=1.d0/2.d0**maxbit
      in=0
      end
!
!***********************************************************************
!                                                                      *
!***********************************************************************
!
      subroutine ivset(iv,mdeg,maxdim,iseed)
 
      implicit none
 
      integer maxdim,i1,i,j,j1,iseed,nvec
      integer iv(maxdim,maxdim),mdeg(maxdim)
      double precision rancid(25000),yran
 
      i1 = 0
      nvec = 35
      write(8,*)' iseed = ',iseed
      call xvrand(iseed,rancid,0,nvec)
 
        do i = 1,12
          do j = 1,mdeg(i)
            if ( j .EQ. 1 ) then
            iv(i,j) = 1
            else
            i1 = i1 + 1
            j1 = 2**(j-1)
            yran = j1*rancid(i1)
            iv(i,j) = 2*int(yran)+1
            endif
          end do
        end do
        return
        end
!
!***********************************************************************
!                                                                      *
!***********************************************************************
!
      subroutine ivcalc(iv,mdeg,ip,maxdim)

      implicit none
 
      integer i,j,k,mm(20),maxdim
 
      integer ip(maxdim), mdeg(maxdim),iv(maxdim,maxdim)
      integer i1,j1,ideg,ipol,a(12),ibin
!
!
!***********************************************************************
!     The vector iv can be changed in the following manner:            *
!        The elements, iv(i,j) that are nonzero are arbitrary and can  *
!        be changed to any odd integer that is less than 2**j          *
!***********************************************************************
!
!
        do 10 i1 = 1,12
        ideg = mdeg(i1)
        ipol = ip(i1)
          do j1 = 1,ideg
          mm(j1) = iv(i1,j1)
          a(j1) = 0
          end do
        a(ideg) = 1
        ipol = ibin(ipol,1)
          if ( ideg .GE. 2 ) then
          a(1) = ipol / 10**(ideg-2)
            if ( ideg .GE. 3 ) then
              do j1 = 2,ideg-1
              ipol = ipol - a(j1-1)*10**(ideg-j1)
              a(j1) = ipol / 10**(ideg-j1-1)
              end do
            endif
          endif
 
          do i = ideg+1,12
          k = mm(i-ideg)
            do j = ideg,1,-1
              if ( a(j) .NE. 0 ) then
              k = a(j) * xor(2**j*mm(i-j),k)
              endif
            end do
          mm(i) = k
          iv(i1,i) = k
          end do
 
 10     continue
 
        return
        end
!
!***********************************************************************
!   Generates the Gray code for an integer, n, if is > 0               *
!   If is < 0, then return the inverse Gray code of n, output `gray'   *
!***********************************************************************
!
        function igray(n,is)
 
        implicit none
 
        integer igray,is,n,idiv,ish
 
          if ( is .GE. 0 ) then
          igray = ieor(n,n/2)
          else
          ish = -1
          igray = n
 1          continue
            idiv = ishft(igray,ish)
            igray = ieor(igray,idiv)
              if ( idiv .LE. 1 .OR. ish .EQ. -16 ) RETURN
            ish = ish + ish
            GOTO 1
          endif
        return
        end
!
!***********************************************************************
!   Generates the binary code for an integer, n, if is > 0             *
!   If is < 0, then return nothing                                     *
!***********************************************************************
!
        function ibin (n,is)
 
        implicit none
 
        integer is,n,ish,n1,i1,ideg,ibin1,i,ibin
 
          if ( is .GE. 0 ) then
            if ( n .LE. 0 ) then
            ibin = 0
            return
            endif
          ideg = LOG(real(n)) / LOG(2.0) + 1
          ish = n/2**ideg
          ibin1 = ish*(10**ideg)
          n1 = n
          i1 = ish
            do i = ideg-1,0,-1
            n1 = n1 - i1*2**(i+1)
            i1 = n1 / 2**i
            ibin1 = ibin1 + i1*10**i
            end do
          endif
        ibin = ibin1
        return
        end
!
!***********************************************************************
!                                                                      *
!***********************************************************************
!
      subroutine xvrand(seed,rancid,isdum,nran)
 
      implicit double precision (a-h,o-z)

      integer  seed,prime,rvec
      save izz
 
      parameter  (maxnop=199,maxbuf=1000,lshufl=5,lvec=25000)
      parameter  (nddd=lshufl*lvec)
      parameter  (ic=(10**4-27),ia=25819,ib=22263,im=(2**15-19))
      parameter  (xm=(1.0d0*im)**(-1))
 
      dimension  prime(maxnop),iwork(nddd)
      dimension  rvec(lvec,2),rancid(nran)
 
      data izz   /0/
 
      if(izz.eq.0) then
      call  tabprm(maxbuf,mprim,prime,iwork,maxnop,maxbuf)
      call  tabcid(seed,rvec,iwork,lvec,lshufl,iseed1,iseed2,nddd, &
     &              mprim,prime)

      izz=99
      end if
!
!
!***********************************************************************
!                                                                      *
!         Portable vectorized random number generator (G. Arbanas)     *
!                                                                      *
!***********************************************************************
!
!
      do 1 j=1,nran
       ir=rvec(j,1)
       is=rvec(j,2)
       rvec(j,2)=ir
       rvec(j,1)=mod(ia*ir+ib*is+ic,im)
       rancid(j)=xm*rvec(j,1)+0.5d0*xm
    1 continue
      return
      end

       subroutine  tabcid(iseed,rvec,iwork,nran,nshfl,iseed1,iseed2,   &
     &                    nddd,mprim,prime)

       implicit double precision (a-h,o-z)
       integer  prime,rvec

       parameter  (ic=(10**4-27),ia=25819,ib=22263,im=(2**15-19))
       parameter  (xm=(1.0d0*im)**(-1))

       dimension  rvec(nran,2),iwork(nddd),prime(mprim)
!                                                                      *
!                                                                      *
!***********************************************************************
!                                                                      *
!         Give linear congruence a preliminary warmup                  *
!         after  mapping initial `iseed' on to two succesive primes    *
!                                                                      *
!***********************************************************************
!                                                                      *
!                                                                      *
       iss=6+mod(iseed,mprim-5)
       is=prime(iss)
       iss=6+mod(iseed+1,mprim-5)
       ir=prime(iss)
       iseed1=is
       iseed2=ir
       do 1 j=1,197
         iss=is
         is=ir
         ir=mod(ia*ir+ib*iss+ic,im)
    1  continue

!
!***********************************************************************
!                                                                      *
!         Set up long table in sequence                                *
!                                                                      *
!***********************************************************************
!

       long=nshfl*nran
       do 3 k=1,long
         iss=is
         is=ir
         ir=mod(ia*ir+ib*iss+ic,im)
         iwork(k)=ir
    3  continue

!
!***********************************************************************
!                                                                      *
!         Generate the shuffled table                                  *
!                                                                      *
!***********************************************************************
!

      do 5 j=1,nran
      iss=is
      is=ir
      ir=mod(ia*ir+ib*iss+ic,im)
      k=mod(ir,long)+1
      rvec(j,2)=iwork(k)
      iss=is
      is=ir
      ir=mod(ia*ir+ib*iss+ic,im)
      k=mod(ir,long)+1
      rvec(j,1)=iwork(k)
    5 continue

      return
      end

      subroutine tabprm(nmaxi,mprim,nprim,ivec,maxnop,maxbuf)

      implicit double precision (a-h,o-z)

!
!***********************************************************************
!                                                                      *
!         Generates a table of prime numbers .le.nmaxi,                *
!         nprim(m), m=1,...,mprim                                      *
!                                                                      *
!***********************************************************************
!
!
       dimension  nprim(maxnop),ivec(maxbuf)

       nbuf=((nmaxi-1)/maxbuf)+1
       mprim=0
       do 101 ibuf=1,nbuf
         nmin=maxbuf*(ibuf-1)+1
         nmax=nmin+maxbuf-1
         xmax=nmax+1
         mdiv=dsqrt(xmax)
         do 103 n=1,maxbuf
       ivec(n)=n+nmin-1
  103     continue
      do 102 idiv=2,mdiv
      irem=mod(nmin,idiv)
      if(ibuf.eq.1) nmin1=nmin+2*idiv-irem
      if(ibuf.gt.1) then
      nmin1=nmin+idiv-irem
      if(irem.eq.0) nmin1=nmin
      end if
      do 104 n=nmin1,nmax,idiv
      ivec(n-nmin+1)=0
  104         continue
  102     continue
      do 105 n=1,maxbuf
      if(ivec(n).ne.0) then
       mprim=mprim+1
       nprim(mprim)=ivec(n)
      end if
      if(mprim.eq.maxnop) return
  105  continue
  101  continue
       return
       end

