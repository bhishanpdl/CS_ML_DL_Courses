!Topic     : Assign1 Q2
!Programmer: Bhishan Poudel
!cmd       : clear; f90 qn2WithoutJumps.f90 && ./a.out 

program qn2WithoutJumps
implicit none

      integer nphi, i
      integer kwrite !kread = 5 and kwrite = 6 are standard values

      double precision pi
      double precision x,y,r,phi,arctan,arctan2
      double complex   z,lnz,sqrtz

      data kwrite/6/ ! initializing value kwrite = 6
    
      !!variables for extra precision
      integer xx
      double precision countmin,countmax,step,yy,jj,error,diff
    
      !calculate and display value of pi

      pi = acos(0.d0)*2.d0

      write (kwrite,*) 'pi = ' ,pi ! 3.1415926535897931

      r=1.d0
      
      write (kwrite,*) 'complex numbers z = r * e ^ (i phi)'
      write (kwrite,*) 'sqrt(z) = sqrt(r) * cos (phi/2) + img (sqrt(r) * sin (phi/2))'
      write (kwrite,*) 'ln(z) = ln(r) + img (phi)'
      write (kwrite,*) 'Table  for r=1'
      write (*,*)
      
      WRITE (*,80)
      80 FORMAT (T1,'Counter',T10,'phi',T22,'x',T32,'y',T40,'sq(z)',T51,'img', T61,'lnz',T71,'img',T81,'atan',T90,'atan2')
      
      write(*,*)

      do i=-16,16,1
      
          if(mod(i,2) ==0) then              
              countmin = real(i) - 0.1 ! -16 -0.1 = -16.01 
              countmax = real(i) + 0.1 ! -16 +0.1 = -15.99
              step     = 0.01
              yy        = 0.0
              error    = 0.001
            
              do jj = countmin,countmax,step ! eg. jj = -16.01
              
                  yy = jj + step ! -16.01 + 0.01 = -16.02
                              
                  diff = yy - real(i) ! we will exclude exact multiples
              
                  phi=yy*0.25d0*pi ! -16.02*0.25*pi

                  x=r*cos(phi)
                  y=r*sin(phi)

                  z=dcmplx(x,y)
                  lnz=log(z)
                  arctan=atan(y/x)
                  arctan2=atan2(y,x)
                  sqrtz = sqrt(z)
          
                   
                  if(abs(diff)>error) then  ! excluding multiples
                      !write (*,100) yy
                      write (kwrite,100) yy,phi,x,y,sqrtz,lnz,arctan,arctan2
                  endif
                
              end do 
      
      
          else
        
              phi=float(i)*0.25d0*pi

        x=r*cos(phi)
        y=r*sin(phi)

        z=dcmplx(x,y)
        lnz=log(z)
        arctan=atan(y/x)
        arctan2=atan2(y,x)
        sqrtz = sqrt(z)

              write (kwrite,110) i,phi,x,y,sqrtz,lnz,arctan,arctan2
          end if

      end do
      
      100 format(10(2x,f8.2))     ! formatting for 10 float points
      110 format(i10,9(2x,f8.2))  ! formatting for 1 integer and 9 float points 
     
      open(unit = 9, file='qn2WithoutJumps.dat', status='replace') !replacing old file if exitsts
      do i=-16,16,1
      
          if(mod(i,2) ==0) then              
              countmin = real(i) - 0.1 ! -16 -0.1 = -16.01 
              countmax = real(i) + 0.1 ! -16 +0.1 = -15.99
              step     = 0.01
              yy        = 0.0
              error    = 0.001
            
              do jj = countmin,countmax,step ! eg. jj = -16.01
              
                  yy = jj + step ! -16.01 + 0.01 = -16.02
                              
                  diff = yy - real(i) ! we will exclude exact multiples
              
                  phi=yy*0.25d0*pi ! -16.02*0.25*pi

                  x=r*cos(phi)
                  y=r*sin(phi)

                  z=dcmplx(x,y)
                  lnz=log(z)
                  arctan=atan(y/x)
                  arctan2=atan2(y,x)
                  sqrtz = sqrt(z)
          
                   
                  if(abs(diff)>error) then  ! excluding multiples
                      !write (*,100) yy
                      write (9,200) yy,phi,x,y,sqrtz,lnz,arctan,arctan2
                  endif
                
              end do 
      
      
          else
        
              phi=float(i)*0.25d0*pi

        x=r*cos(phi)
        y=r*sin(phi)

        z=dcmplx(x,y)
        lnz=log(z)
        arctan=atan(y/x)
        arctan2=atan2(y,x)
        sqrtz = sqrt(z)

              write (9,210) i,phi,x,y,sqrtz,lnz,arctan,arctan2
          end if

      end do
      
      200 format(10(2x,f8.2))     ! formatting for 10 float points
      210 format(i10,9(2x,f8.2))  ! formatting for 1 integer and 9 float points
      
      close(unit =9)
   
end program qn2WithoutJumps

