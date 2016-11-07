set terminal postscript eps color

#set output "inputplot2d.eps"
#plot "inputplot2ddata.txt" using 1:2

set output "valueplot2d.eps"
plot "valueplot2ddata.txt" using 1:2 with lines title "Network Output", \
     "inputplot2ddata.txt" using 1:2 title "Training Samples"

#set yrange [-1:11]

set output "costplot2d.eps"
plot "costplot2ddata.txt" using 1:2 with lines title "Network Output", \
     "inputcostplot2ddata.txt" using 1:2 title "Training Samples"
