'''
Created on Nov 21, 2011

@author: reza
'''

import fnapprox

def normalize(file_suffix):
    TXT_SUFFIX = '.txt'
    NORM_SUFFIX = '-norm' + TXT_SUFFIX
    f1_name = '%s-%s' % (fnapprox.EXP_RANDOM, file_suffix)
    f2_name = '%s-%s' % (fnapprox.EXP_ACTIVE, file_suffix)
    f3_name = '%s-%s' % (fnapprox.EXP_ACTCOST, file_suffix)
    f1 = open(f1_name + TXT_SUFFIX, 'r')
    f2 = open(f2_name + TXT_SUFFIX, 'r')
    f3 = open(f3_name + TXT_SUFFIX, 'r')
    f1_norm = open(f1_name + NORM_SUFFIX, 'w')
    f2_norm = open(f2_name + NORM_SUFFIX, 'w')
    f3_norm = open(f3_name + NORM_SUFFIX, 'w')
    
    index = 0
    for f1line in f1:
        f2line = f2.readline()
        f3line = f3.readline()
        value1 = float(f1line.split(' ')[1])
        value2 = float(f2line.split(' ')[1])
        value3 = float(f3line.split(' ')[1])
        f1_norm.write('%d %f\n' % (index, 1.0))
        f2_norm.write('%d %f\n' % (index, value2 / value1))
        f3_norm.write('%d %f\n' % (index, value3 / value1))
        index += 1
    
    f1.close()
    f2.close()
    f3.close()
    f1_norm.close()
    f2_norm.close()
    f3_norm.close()
    
if __name__ == '__main__':
    normalize('cost')
    normalize('error')
    normalize('errortimescost')
    
    