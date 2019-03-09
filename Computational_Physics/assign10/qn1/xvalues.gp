#Bhishan Poudel
# Nov 15, 2015

# cmd: gnuplot xvalues.gp


#for plot1
set terminal png enhanced
set output 'xvalues1.png'
set title "plot of xvalues"
plot "<(sed -n '1,1000p' xvalues.dat)" title "plot1"

# plot2
set terminal png enhanced
set output 'xvalues2.png'
plot "<(sed -n '1003,2002p' xvalues.dat)" title "plot2"

# for plot3
set terminal png enhanced
set output 'xvalues3.png'
plot "<(sed -n '2005,3004p' xvalues.dat)" title "plot3"

# plot4
set terminal png enhanced
set output 'xvalues4.png'
plot "<(sed -n '3007,4006p' xvalues.dat)" title "plot4"


