
  subroutine trapezoid_rule(f, a, b, n, integral)
    double precision :: f
    double precision, intent(in) ::  a, b
    integer, intent(in) :: n
    double precision, intent(out) ::  integral
    integer :: k
    double precision :: s
    s = 0
    do k=1, n-1
      s = s + f(a + (b-a)*k/n)
    end do  
    integral = (b-a) * (f(a)/2 + f(b)/2 + s) / n
  end subroutine trapezoid_rule  



