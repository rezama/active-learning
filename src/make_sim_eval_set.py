
import sys
import os

from fnapprox import EVAL_SAMPLES_AXIS, X_MIN, X_MAX, Y_MIN
from simulator import SIM_VAR_NAME1, SIM_VAR_NAME2
import simulator

def generate_param_files():
    unit_x = (X_MAX - X_MIN) / (EVAL_SAMPLES_AXIS - 1)
    unit_y = (X_MAX - X_MIN) / (EVAL_SAMPLES_AXIS - 1)
    index = 0
    for i in range(EVAL_SAMPLES_AXIS):
        for j in range(EVAL_SAMPLES_AXIS):
            x = X_MIN + i * unit_x
            y = Y_MIN + j * unit_y
            filename = '../simdata/results/params_1_i_%d.txt' % index
            f_params = open(filename, 'w')
            f_params.write('%s\t%f\n' % (SIM_VAR_NAME1, x))
            f_params.write('%s\t%f\n' % (SIM_VAR_NAME2, y))
            f_params.close()
            index += 1
            
    f_paramswritten = open('../simdata/results/paramswritten_1.txt', 'w')
    f_paramswritten.close()
    
    print 'You can now submit the condor job:'
    print 'ruby condor_run.rb ~/active/simdata 1 %d <trials>' % index

def process_value_files():
    f_evalset = open('../simdata/evalset.txt', 'w')
    filenames = os.listdir('../simdata/results/')
    for filename in filenames:
        if filename.startswith('value_'):
            param_filename = filename.replace('value_', 'params_')
#            param_filename = param_filename.replace('_r_0', '')
#            param_filename = param_filename.replace('_r_1', '')
#            param_filename = param_filename.replace('_r_2', '')
#            param_filename = param_filename.replace('_r_3', '')
#            param_filename = param_filename.replace('_r_4', '')
            f_run = open('../simdata/results/' + filename, 'r')
            f_params = open('../simdata/results/' + param_filename, 'r')
            x = f_params.readline().split()[1]
            y = f_params.readline().split()[1]
            dist = f_run.readline().rstrip()
            angle = f_run.readline().rstrip()
            
            if simulator.IGNORE_NEGATIVE and (float(dist) < 0):
                dist = '0.0' 
            
            f_evalset.write('%s %s %s %s\n' % (x, y, dist, angle))
            
            f_run.close()
            f_params.close()
            
    f_evalset.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Please supply job parameter (generate/process)'
    else:
        mode = sys.argv[1]
        if (mode == 'generate'):
            generate_param_files()
        elif (mode == 'process'):
            process_value_files()
        else:
            print 'Invalid job: %s' % mode
