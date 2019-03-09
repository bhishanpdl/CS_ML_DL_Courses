!===============================================================================      
!===============================================================================
!  This subroutine prints the matrix eigenvalues and eigenvectors
!  examples: call print_matrix('Real Eigenvalues', 1,nn,wr,1)
!  examples: call print_matrix('Left eigenvectors(columns)', nn,nn,vl,nn)

    subroutine print_matrix(desc,m,n,a,lda)
    
    character*(*),intent(in) :: desc ! description
    integer,intent(in) :: m,n,lda
    double precision,intent(inout):: a(lda,*)
    integer :: i,j
    
    write(*,*)
    write(*,*) desc
    
    do i=1,m
      write(*,10000) (a(i,j), j=1,n )
    end do 
    10000 format(5(:,2x,f6.3))  ! format works upto n=5 matrix
    return
    end subroutine
!===============================================================================      

