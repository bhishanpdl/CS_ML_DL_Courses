# gnuplot hw9qn2b.gp && xdg-open hw9qn2b.eps

f = 'hw9qn2b.dat'
set terminal postscript enhanced eps color
set output 'hw9qn2b.eps'

set title 'Plot of 1/sqrt(n) vs. integral'
set xrange [*:*]
set yrange [*:*]

set xlabel "1/sqrt(n)"
set ylabel "integral"

plot f u 1:2 w l t "label"

