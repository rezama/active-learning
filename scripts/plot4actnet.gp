set terminal postscript eps color

set xlabel "Sample"
set ylabel "Error"

set output "actnet-error.eps"
plot "actnet-error.txt" using 1:2 with lines title "Cost-Efficient Active Sampling"

set xlabel "Sample"
set ylabel "Average Sample Cost"

set output "actnet-cost.eps"
plot "actnet-cost.txt" using 1:2 with lines title "Cost-Efficient Active Sampling"

set xlabel "Sample"
set ylabel "Error * Average Sample Cost"

set output "actnet-errortimescost.eps"
plot "actnet-errortimescost.txt" using 1:2 with lines title "Cost-Efficient Active Sampling"

set xlabel "Sample"
set ylabel "Error"

set output "compare-error.eps"
plot "random-error.txt" using 1:2 with lines title "Random Sampling", \
     "active-error.txt" using 1:2 with lines title "Active Sampling", \
     "actcost-error.txt" using 1:2 with lines title "Cost-Efficient Active Sampling", \
     "actnet-error.txt" using 1:2 with lines title "Learned Active Sampling"

set xlabel "Sample"
set ylabel "Average Sample Cost"

set output "compare-cost.eps"
plot "random-cost.txt" using 1:2 with lines title "Random Sampling", \
     "active-cost.txt" using 1:2 with lines title "Active Sampling", \
     "actcost-cost.txt" using 1:2 with lines title "Cost-Efficient Active Sampling", \
     "actnet-cost.txt" using 1:2 with lines title "Learned Active Sampling"

set xlabel "Sample"
set ylabel "Error * Average Sample Cost"

set output "compare-errortimescost.eps"
plot "random-errortimescost.txt" using 1:2 with lines title "Random Sampling", \
     "active-errortimescost.txt" using 1:2 with lines title "Active Sampling", \
     "actcost-errortimescost.txt" using 1:2 with lines title "Cost-Efficient Active Sampling", \
     "actnet-errortimescost.txt" using 1:2 with lines title "Learned Active Sampling"

set xlabel "X"
set ylabel "Y"
     
set output "actnet-points-0.eps"
plot "actnet-points-0.txt" using 1:2 title "Learned Active Sampled Points"

set dgrid3d 40, 40, 10
set pm3d

set xlabel "X"
set ylabel "Y"
set zlabel "Value"

set output "actnet-learnedvalue-0.eps"
splot "actnet-learnedvalue-0.txt" using 1:2:3 with lines title "Learned Active Sampling Learned Value Function"


