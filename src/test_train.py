'''
Created on Nov 14, 2011

@author: reza
'''

import math
import random

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from fnapprox import DomainFnApprox
from scipy import dot

RANDOM_TRAINING_SAMPLES = 1000
EVAL_SAMPLES_AXIS = 100
PLOT_SAMPLES_AXIS = 100
MAX_EPOCHS = 50
CONT_EPOCHS = 10
MOMENTUM = 0.
LEARNING_RATE = 0.01
VALIDATION_PROPORTION = 0.2

X_MIN = -7
X_MAX = 7
Y_MIN = -7
Y_MAX = 7

def fn(x):
    return math.sin(2 * x) + math.atan(x)

FN = fn

def cost_fn(x):
    (f, i) = math.modf(x) #@UnusedVariable
    if (abs(f) < 0.2) or (abs(f) > 0.8):
        return 10
    else:
        return 1

COST_FN = cost_fn

def base_experiment():
    (eval_dataset, eval_costset) = DomainFnApprox.make_evaluation_datasets()

    random_train_dataset = SupervisedDataSet(2, 1)
    random_train_costset = SupervisedDataSet(2, 1)
    for i in range(RANDOM_TRAINING_SAMPLES):
        x = random.uniform(X_MIN, X_MAX)
        y = random.uniform(Y_MIN, Y_MAX)
        z = FN(x, y)
        z_cost = COST_FN(x, y)
        random_train_dataset.addSample([x, y], [z])
        random_train_costset.addSample([x, y], [z_cost])
    
    value_network = buildNetwork(2, 80, 20, 1, hiddenclass = SigmoidLayer, bias = True)
    value_trainer = BackpropTrainer(value_network,
                                    learningrate = LEARNING_RATE, 
                                    momentum = MOMENTUM, verbose = True)
    
    print 'Value Network Topology:'
    print value_network
    
    cost_network = buildNetwork(2, 80, 20, 1, hiddenclass = SigmoidLayer, bias = True)
    cost_trainer = BackpropTrainer(cost_network,
                                   learningrate = LEARNING_RATE,
                                   momentum = MOMENTUM, verbose = True)
    
#    test_derivatives(value_network, [1, 1])
#    test_derivatives(cost_network, [1, 1])
    
    
    print 'Value MSE before: %.4f' % value_trainer.testOnData(eval_dataset)
    value_trainer.trainUntilConvergence(random_train_dataset, continueEpochs = 6,
                                        maxEpochs = MAX_EPOCHS)
#    value_trainer.trainOnDataset(random_train_dataset, 1000)
    print 'Value MSE after: %.4f' % value_trainer.testOnData(eval_dataset)
       
    print 'Cost MSE before: %.4f' % cost_trainer.testOnData(eval_costset)
    cost_trainer.trainUntilConvergence(random_train_costset, continueEpochs = 6,
                                       maxEpochs = MAX_EPOCHS)
#    cost_trainer.trainOnDataset(random_train_costset, 1000)
    print 'Cost MSE after: %.4f' % cost_trainer.testOnData(eval_costset)

#    test_derivatives(value_network, [1, 1])
#    test_derivatives(cost_network, [1, 1])
    
    f_value = open('../data/learnedvalue.txt', 'w')
    f_cost = open('../data/learnedcost.txt', 'w')
    unit = (X_MAX - X_MIN) / (EVAL_SAMPLES_AXIS - 1)
    for i in range(EVAL_SAMPLES_AXIS):
        for j in range(EVAL_SAMPLES_AXIS):
            x = X_MIN + i * unit
            y = Y_MIN + j * unit
            z = value_network.activate([x, y])
            z_cost = cost_network.activate([x, y])
            f_value.write('%f %f %f\n' % (x, y, z[0]))
            f_cost.write('%f %f %f\n' % (x, y, z_cost[0]))
    f_value.close()
    f_cost.close()

def calc_chained_derivs(module, seq, use_error = False):
    """This function is taken from the pybrain source code"""
    module.reset()
    for sample in seq:
        module.activate(sample[0])
    error = 0
    ponderation = 0.
    for offset, sample in reversed(list(enumerate(seq))):
        # need to make a distinction here between datasets containing
        # importance, and others
        target = sample[1]
        if use_error:
            outerr = target - module.outputbuffer[offset]
        else:
            outerr = module.outputbuffer[offset]
        if len(sample) > 2:
            importance = sample[2]
            error += 0.5 * dot(importance, outerr ** 2)
            ponderation += sum(importance)
            module.backActivate(outerr * importance)
        else:
            error += 0.5 * sum(outerr ** 2)
            ponderation += len(target)
            # FIXME: the next line keeps arac from producing NaNs. I don't
            # know why that is, but somehow the __str__ method of the 
            # ndarray class fixes something,
            str(outerr)
            module.backActivate(outerr)

    return error, ponderation

def test_derivatives(net, inp):
    print 'Computing derivatives:'
    derivatives = []
    point_x = inp[0]
    point_y = inp[1]
    xs = [point_x, point_y]
    zs = net.activate(xs)
    zstarget = [FN(point_x, point_y)]
    delta = 0.0001
    for i in range(len(net.params)):
        net.params[i] += delta
        newzs = net.activate(xs)
        d = (newzs[0] - zs[0]) / delta
        derivatives.append(d)
        net.params[i] -= delta

    print '[', 
    for i in range(len(derivatives)):
        print '%+4.2f,' % derivatives[i],
    print ']'

    dset = SupervisedDataSet(2, 1)
    dset.addSample(xs, zstarget)

    calc_chained_derivs(net, dset)
    print 'Bprop derivatives (value):'
    print '[', 
    for i in range(len(net.derivs)):
        print '%+4.2f,' % net.derivs[i],
    print ']'
    net.resetDerivatives()
    
    calc_chained_derivs(net, dset, use_error = True)
    print 'Bprop derivatives (error):'
    print '[', 
    for i in range(len(net.derivs)):
        print '%+4.2f,' % net.derivs[i],
    print ']'
    net.resetDerivatives()
    print

if __name__ == '__main__':
    f_input = open('../data2d/inputplot2ddata.txt', 'w')
    f_input_cost = open('../data2d/inputcostplot2ddata.txt', 'w')
    dataset = SupervisedDataSet(1, 1)
    costset = SupervisedDataSet(1, 1)
    for i in range(RANDOM_TRAINING_SAMPLES):
        x = random.uniform(-3, 3)
        z = fn(x)
        z_cost = cost_fn(x)
        dataset.addSample([x], [z])
        costset.addSample([x], [z_cost])
        f_input.write("%f %f\n" % (x, z))
        f_input_cost.write("%f %f\n" % (x, z_cost))
    f_input.close()
    f_input_cost.close()
    
    eval_dataset = SupervisedDataSet(1, 1)
    eval_costset = SupervisedDataSet(1, 1)
    for i in range(RANDOM_TRAINING_SAMPLES):
        x = random.uniform(-3, 3)
        z = fn(x)
        z_cost = cost_fn(x)
        eval_dataset.addSample([x], [z])
        eval_costset.addSample([x], [z_cost])

    value_network = buildNetwork(1, 20, 1, hiddenclass = SigmoidLayer, bias = True)
    value_trainer = BackpropTrainer(value_network, learningrate = 0.01, 
                              momentum = 0.90, verbose = True)
    
    print value_network
    
    cost_network = buildNetwork(1, 40, 20, 1, hiddenclass = SigmoidLayer, bias = True)
    cost_trainer = BackpropTrainer(cost_network, learningrate = 0.01,
                                   momentum = 0.00, verbose = True)
    
    print 'Value MSE before: %.4f' % value_trainer.testOnData(eval_dataset)
    value_trainer.trainUntilConvergence(dataset, continueEpochs = 6, maxEpochs = 500)
#    value_trainer.trainOnDataset(dataset, 1000)
    print 'Value MSE after: %.4f' % value_trainer.testOnData(eval_dataset)
    
    print 'Cost MSE before: %.4f' % cost_trainer.testOnData(eval_costset)
    cost_trainer.trainUntilConvergence(costset, continueEpochs = 6, maxEpochs = 500)
#    cost_trainer.trainOnDataset(costset, 1000)
    print 'Cost MSE after: %.4f' % cost_trainer.testOnData(eval_costset)
#    print cost_network.params

    f_value = open('../data2d/valueplot2ddata.txt', 'w')
    f_cost = open('../data2d/costplot2ddata.txt', 'w')
    plot_dataset = SupervisedDataSet(2, 1)
    plot_unit = 6.0 / PLOT_SAMPLES_AXIS
    for i in range(PLOT_SAMPLES_AXIS + 1):
        x = -3 + i * plot_unit
        z = value_network.activate([x])
        z_cost = cost_network.activate([x])
        f_value.write("%f %f\n" % (x, z))
        f_cost.write("%f %f\n" % (x, z_cost))
    f_value.close()
    f_cost.close()

    