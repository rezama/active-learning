export PYTHONPATH=~/workspace/active/src/pybrain/
for i in `ls -d */ | grep -v old`
do
  echo $i
  cd $i
  rm *eps
  python ../../postprocess.py
  gnuplot ../plot3d.gp
  gnuplot ../normplot3d.gp
  cd ..
done

