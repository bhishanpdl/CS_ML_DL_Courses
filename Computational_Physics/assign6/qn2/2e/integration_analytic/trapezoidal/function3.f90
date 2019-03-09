
  double precision function f3(E)
    implicit none
    double precision, intent(in) :: E
    f3 = (68658.34d0)/( (E-75)**2 + 822.26)
    
    
    
  end function f3

