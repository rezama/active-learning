'''
Created on Nov 21, 2011

@author: reza
'''
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer

if __name__ == '__main__':
    dataset = SupervisedDataSet(2, 1)
    dataset.addSample([0, 0], [0])
    dataset.addSample([0, 1], [1])
    dataset.addSample([1, 0], [1])
    dataset.addSample([1, 1], [0])
    
    network = buildNetwork(2, 4, 1)
    trainer = BackpropTrainer(network, learningrate = 0.01, momentum = 0.2,
                              verbose = False)
    
    print 'MSE before', trainer.testOnData(dataset)
    trainer.trainOnDataset(dataset, 1000)
    print 'MSE after', trainer.testOnData(dataset)
    
    z = network.activate([0, 0])
    print z
    
    print 'Final Weights: ', network.params
    
    