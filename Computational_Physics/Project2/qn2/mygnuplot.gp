# clear; gnuplot mygnuplot.gp; rm *~

set terminal postscript eps color enhanced   # size 10,5 font ",20"
set output 'potentialA.eps'

# set the position of the key
#set key top left
set key at graph 0.90, graph 0.55

set key autotitle columnheader
set key font ",24"
													
set title  "Potentials v_l(q,q') \n {/*0.7for q'=0.5 fm^{-1}}" font ",24"
set xlabel "q (Mev)" font ",24"
set ylabel "v0(q,q') (1/Mev^2)" font ",24"

# additional labels inside the graph
#set label "q' = 2.5 fm^{-1}" left at graph 0.15, graph 0.95  # graph 0,0 is left bottom 
# ranges
set autoscale
set xrange [0:5]
#set yrange [-1:1]

# tickmarks
set xtic auto
set ytic auto



plot 'v0a.dat' using 1:2 with lines title 'v0',\
     'v2a.dat' using 1:2 with lines title 'v2',\
     'v4a.dat' using 1:2 with lines title 'v4'

################### Notes ######################################################
# greek letters
#set ylabel "d^2{/Symbol s}/dp/d{/Symbol W} (mb/(MeV/c)/str)" 
#set xlabel "K^+ Momentum (GeV/c)"


