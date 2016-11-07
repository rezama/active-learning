# to be run inside:
# active/data/somefolder/

export PYTHONPATH=~/workspace/active/lib

python ../../src/normalize.py
gnuplot ../../scripts/plot3basic.gp
gnuplot ../../scripts/plot3basicnorm.gp

