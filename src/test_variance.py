'''
Created on Nov 21, 2011

@author: reza
'''
import random

NUM_TRIALS = 1000
NUM_POINTS = 5

def get_variance(low, high):
    sum_avg_var = 0.
    for trial in range(NUM_TRIALS): #@UnusedVariable
        values = []
        for i in range(NUM_POINTS): #@UnusedVariable
            num = random.uniform(low, high)
            values.append(num)
        mean = sum(values) / NUM_POINTS
        variances = [(e - mean) ** 2 for e in values]
        avg_variance = sum(variances) / NUM_POINTS
        sum_avg_var += avg_variance
    return sum_avg_var / NUM_TRIALS

if __name__ == '__main__':
    print get_variance(1, 10)
    print get_variance(-1, 1)