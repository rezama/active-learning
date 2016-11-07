set terminal postscript eps color

set xlabel "Sample"
set ylabel "Error"

set output "random-error.eps"
plot "random-error.txt" using 1:2 with lines title "Random Sampling"

set output "active-error.eps"
plot "active-error.txt" using 1:2 with lines title "Active Sampling"

set xlabel "Sample"
set ylabel "Average Sample Cost"

set output "random-cost.eps"
plot "random-cost.txt" using 1:2 with lines title "Random Sampling"

set output "active-cost.eps"
plot "active-cost.txt" using 1:2 with lines title "Active Sampling"

set xlabel "Sample"
set ylabel "Error * Average Sample Cost"

set output "random-errortimescost.eps"
plot "random-errortimescost.txt" using 1:2 with lines title "Random Sampling"

set output "active-errortimescost.eps"
plot "active-errortimescost.txt" using 1:2 with lines title "Active Sampling"

set xlabel "Sample"
set ylabel "Error"

set output "compare-error.eps"
plot "random-error.txt" using 1:2 with lines title "Random Sampling", \
     "active-error.txt" using 1:2 with lines title "Active Sampling"

set xlabel "Sample"
set ylabel "Average Sample Cost"

set output "compare-cost.eps"
plot "random-cost.txt" using 1:2 with lines title "Random Sampling", \
     "active-cost.txt" using 1:2 with lines title "Active Sampling"

set xlabel "Sample"
set ylabel "Error * Average Sample Cost"

set output "compare-errortimescost.eps"
plot "random-errortimescost.txt" using 1:2 with lines title "Random Sampling", \
     "active-errortimescost.txt" using 1:2 with lines title "Active Sampling"
set xlabel "X"
set ylabel "Y"
     
set pointsize 5

set output "random-points-0.eps"
plot "random-points-0.txt" using 1:2 title "Random Sampled Points"

set output "active-points-0.eps"
plot "active-points-0.txt" using 1:2 title "Active Sampled Points"

set dgrid3d 31, 31, 10
set pm3d

set xlabel "X"
set ylabel "Y"
set zlabel "Value"

set output "funcvalue.eps"
splot "funcvalue.txt" using 1:2:3 with lines title "Value Function"

set output "random-learnedvalue-0.eps"
splot "random-learnedvalue-0.txt" using 1:2:3 with lines title "Random Sampling Learned Value Function"
      
set output "active-learnedvalue-0.eps"
splot "active-learnedvalue-0.txt" using 1:2:3 with lines title "Active Sampling Learned Value Function"
      
set zlabel "Cost"
      
set output "funccost.eps"
splot "funccost.txt" using 1:2:3 with lines title "Cost Function"


