f = 'hw9qn2a.dat'
set terminal postscript color
set output 'plot2a.eps'

set title 'Plot of 1/sqrt(n) vs. integral'
set xrange [*:*]
set yrange [*:*]

set xlabel "1/sqrt(n)"
set ylabel "integral"

plot f u 1:2 w l t "label",0.37 w l t 'y = 0.37'

