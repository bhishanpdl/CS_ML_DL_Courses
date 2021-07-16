# gnuplot hw9qn2c.gp && xdg-open hw9qn2c.eps; rm -f *~

f = 'hw9qn2c.dat'
set terminal postscript enhanced eps color
set output 'hw9qn2c.eps'

set title 'Plot of 1/n vs. integral'
set xrange [*:*]
set yrange [*:*]

set xlabel "1/n"
set ylabel "integral"

plot f u 1:2 w l t "label"

