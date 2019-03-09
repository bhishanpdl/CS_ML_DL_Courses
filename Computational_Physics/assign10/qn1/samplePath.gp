# Bhishan Poudel
# Nov 15, 2015

# cmd: gnuplot samplePath.gp

# x and y label for all plots
set xlabel "bin"
set ylabel "sample value"


#for plot1
set terminal png enhanced
set output 'samplePath1.png'
set title "plot of samplePath"
#plot "<(sed -n '1,20p' hw10qn1.dat)" title "plot1"
plot "<(sed -n '1,20p' hw10qn1.dat)" u 1:2 w lp t "plot1"

# plot2
set terminal png enhanced
set output 'samplePath2.png'
#plot "<(sed -n '23,42p' hw10qn1.dat)" title "plot2"
plot "<(sed -n '23,42p' hw10qn1.dat)" u 1:2 w lp t "plot2"

# for plot3
set terminal png enhanced
set output 'samplePath3.png'
#plot "<(sed -n '45,64p' hw10qn1.dat)" title "plot3"
plot "<(sed -n '45,64p' hw10qn1.dat)" u 1:2 w lp t "plot3"

# plot4
set terminal png enhanced
set output 'samplePath4.png'
#plot "<(sed -n '67,86p' hw10qn1.dat)" title "plot4"
plot "<(sed -n '67,86p' hw10qn1.dat)" u 1:2 w lp t "plot4"


