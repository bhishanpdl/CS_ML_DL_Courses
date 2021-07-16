! name:    dminv
!
! purpose:
!
! invert a matrix
!
! usage:   call dminv (a,n,d,l,m)
!
! parameters:
!
! a:       input matrix, destroyed in computation and replaced by
!          resultant inverse.
!          double precision required.
!
! n:       order of matrix a
!
! d:       resultant determinant
!          double precision required.
!
! l:       work vector of length n
!
! m:       work vector of length n
!
! remarks: matrix a must be a general matrix
!
! method:
!
! the standard gauss-jordan method is used. the determinant
! is also calculated. a determinant of zero indicates that
! the matrix is singular.
!
! programs required:
!          none
!
!**********************************************************************
      subroutine dminv (a,n,d,l,m)
      double precision a,d,biga,hold

      dimension a(1),l(1),m(1)


!        search for largest element
!
      d=1.0
      nk=-n
      do 80 k=1,n
      nk=nk+n
      l(k)=k
      m(k)=k
      kk=nk+k
      biga=a(kk)
      do 20 j=k,n
      iz=n*(j-1)
      do 20 i=k,n
      ij=iz+i
   10 if (abs(biga)-abs(a(ij)))  15,20,20
   15 biga=a(ij)
      l(k)=i
      m(k)=j
   20 continue
!
!        interchange rows
!
      j=l(k)
      if(j-k) 35,35,25
   25 ki=k-n
      do 30 i=1,n
      ki=ki+n
      hold=-a(ki)
      ji=ki-k+j
      a(ki)=a(ji)
   30 a(ji) =hold
!
!        interchange columns
!
   35 i=m(k)
      if(i-k) 45,45,38
   38 jp=n*(i-1)
      do 40 j=1,n
      jk=nk+j
      ji=jp+j
      hold=-a(jk)
      a(jk)=a(ji)
   40 a(ji) =hold
!
!        divide column by minus pivot (value of pivot element is
!        contained in biga)
!
   45 if(biga) 48,46,48
   46 d=0.0
      return
   48 do 55 i=1,n
      if(i-k) 50,55,50
   50 ik=nk+i
      a(ik)=a(ik)/(-biga)
   55 continue
!
!        reduce matrix
!
      do 65 i=1,n
      ik=nk+i
      hold=a(ik)
      ij=i-n
      do 65 j=1,n
      ij=ij+n
      if(i-k) 60,65,60
   60 if(j-k) 62,65,62
   62 kj=ij-i+k
      a(ij)=hold*a(kj)+a(ij)
   65 continue
!
!        divide row by pivot
!
      kj=k-n
      do 75 j=1,n
      kj=kj+n
      if(j-k) 70,75,70
   70 a(kj)=a(kj)/biga
   75 continue
!
!        product of pivots
!
      d=d*biga
!
!        replace pivot by reciprocal
!
      a(kk)=1.0/biga
   80 continue
!
!        final row and column interchange
!
      k=n
  100 k=(k-1)
      if(k) 150,150,105
  105 i=l(k)
      if(i-k) 120,120,108
  108 jq=n*(k-1)
      jr=n*(i-1)
      do 110 j=1,n
      jk=jq+j
      hold=a(jk)
      ji=jr+j
      a(jk)=-a(ji)
  110 a(ji) =hold
  120 j=m(k)
      if(j-k) 100,100,125
  125 ki=k-n
      do 130 i=1,n
      ki=ki+n
      hold=a(ki)
      ji=ki-k+j
      a(ki)=-a(ji)
  130 a(ji) =hold
      go to 100
  150 return
      end
