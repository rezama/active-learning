set terminal postscript eps color

set yrange [0:1.2]

set xlabel "Sample"
set ylabel "Error, Normalized against Random Sampling"

set output "compare-error-norm.eps"
plot "random-error-norm.txt" using 1:2 with lines title "Random Sampling", \
     "active-error-norm.txt" using 1:2 with lines title "Active Sampling", \
     "actcost-error-norm.txt" using 1:2 with lines title "Cost-Efficient Active Sampling"

set xlabel "Sample"
set ylabel "Average Sample Cost, Normalized against Random Sampling"

set output "compare-cost-norm.eps"
plot "random-cost-norm.txt" using 1:2 with lines title "Random Sampling", \
     "active-cost-norm.txt" using 1:2 with lines title "Active Sampling", \
     "actcost-cost-norm.txt" using 1:2 with lines title "Cost-Efficient Active Sampling"

set xlabel "Sample"
set ylabel "Error * Average Sample Cost, Normalized against Random Sampling"

set output "compare-errortimescost-norm.eps"
plot "random-errortimescost-norm.txt" using 1:2 with lines title "Random Sampling", \
     "active-errortimescost-norm.txt" using 1:2 with lines title "Active Sampling", \
     "actcost-errortimescost-norm.txt" using 1:2 with lines title "Cost-Efficient Active Sampling"
     
