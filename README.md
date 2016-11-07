# Requirements

You need to have the pybrain library in PYTHONPATH to run this script:

```shell
$ export PYTHONPATH=~/active/pybrain:$PYTHONPATH
```

# Usage

The program can run four types of experiments:

```shell
$ python active.py random  # Random sampling
$ python active.py active  # Active sampling
$ python active.py actcost # Cost-efficient active sampling
$ python active.py actnet  # Neural-network-based active sampling
```

Running the experiments generates data files in the plot3d folder.

The python script is configured to run eight trials in parallel.  I suggest you
run the script on an 8-core machine like lust.cs.utexas.edu. 

After running all four experiments you can generate plots by running plot.sh
inside the plot3d folder:

```shell
$ cd plot3d
$ bash ../plot.sh
```

The plots for the last experiment are generated by a different script:

```shell
$ cd plot3d
$ gnuplot ../plot3dnet.gp
```

For running simulator experiments, the python variable TASK should be set to:

```python
TASK = TASK_SIM
```

Also, since the condor\_run script should be launched using the full path to the
script, this path must be modified in the python file (active.py).  It is 
currently set to:

```shell
/u/reza/villasim/agents/nao-agent/optimization/condor-run.sh
```