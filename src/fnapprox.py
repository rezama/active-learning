'''
Created on Nov 21, 2011

@author: reza
'''

import math
import random
import multiprocessing
import numpy
import sys

from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
#from pybrain.datasets.importance import ImportanceDataSet
from simulator import SimulatorWrapper

#EVAL_SAMPLES_AXIS = int(FIELD_RADIUS) * 2 * 5 + 1
#PLOT_SAMPLES_AXIS = int(FIELD_RADIUS) * 2 * 10 + 1

RANDOM_TRAINING_SAMPLES = 500

PLOT_SAMPLES_AXIS = 141
INIT_COST_SAMPLES_AXIS = 15

INIT_PERF_SAMPLES = 500

MAX_EPOCHS = 50
CONT_EPOCHS = 10
MOMENTUM_HIGH = 0.20
MOMENTUM_LOW = 0.
LEARNING_RATE = 0.01
VALIDATION_PROPORTION = 0.2

ACTIVE_ENSEMBLE_SIZE = 5
ACTIVE_TRY_RAND_POINTS = 1000

class Domain(object):

    NUM_CORES = 8
    NUM_ITERS = 40

    DOM_FNAPPROX = 'fnapprox'
    DOM_SIM = 'sim'
    
    @classmethod
    def is_valid_mode(cls, mode):
        return (mode == cls.DOM_FNAPPROX) or (mode == cls.DOM_SIM)
    
    def __init__(self, mode, inputdim, outputdim, input_range):
        self.mode = mode
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.input_range = input_range

    def generate_grid_points(self, points_per_axis):
        units = [None] * self.inputdim
        for dim in range(self.inputdim):
            span = self.input_range[dim][1] - self.input_range[dim][0]
            units[dim] = float(span) / points_per_axis  

        # FIXME: works for two dimensions only
        points = [] 
        for i in range(points_per_axis):
            for j in range(points_per_axis):
                x = self.input_range[0][0] + i * units[0]
                y = self.input_range[1][0] + j * units[1]
                points.append([x, y])
        return points

    def generate_random_points(self, num_points):
        points = []
        for i in range(num_points): #@UnusedVariable
            point = []
            for dim in range(self.inputdim):
                span_min = self.input_range[dim][0]
                span_max = self.input_range[dim][1]
                val = random.uniform(span_min, span_max)
                point.append(val)
            points.append(point)
        return points

    @classmethod
    def new(cls, mode):
        domain = None
        if mode == cls.DOM_FNAPPROX:
            domain = DomainFnApprox(mode)
        elif mode == cls.DOM_SIM:
            domain = DomainSim(mode)
        return domain

    @classmethod
    def clamp(cls, n, minn, maxn):
        return max(min(maxn, n), minn)

class DomainFnApprox(Domain):

#    USE_MULTIPROCESSING = True
#    NUM_TRIALS = 8
    USE_MULTIPROCESSING = False
    NUM_TRIALS = 1

    INPUTDIM = 2
    OUTPUTDIM = 1
    
    COST_LOW = 1
    COST_HIGH = 10

    EVAL_SAMPLES_AXIS = 71
    
    FIELD_HIGH_COST_MARGIN = 3.5
    #FIELD_HIGH_COST_CIRCLE1 = ( 0.,  4.5)
    #FIELD_HIGH_COST_CIRCLE2 = (-3., -3.)
    #FIELD_HIGH_COST_CIRCLE3 = ( 3., -3.)
    #FIELD_HIGH_COST_RADIUS = 2.5
    FIELD_HIGH_COST_CIRCLE1 = ( 0.,   2.)
    FIELD_HIGH_COST_CIRCLE2 = (-1.5, -1.5)
    FIELD_HIGH_COST_CIRCLE3 = ( 1.5, -1.5)
    FIELD_HIGH_COST_RADIUS = 1.0
    
    FIELD_RADIUS = 7.

    INPUT_RANGE = [None] * INPUTDIM
    INPUT_RANGE[0] = [-FIELD_RADIUS, FIELD_RADIUS]
    INPUT_RANGE[1] = [-FIELD_RADIUS, FIELD_RADIUS]
    
    def __init__(self, mode):
        super(DomainFnApprox, self).__init__(mode, self.INPUTDIM, self.OUTPUTDIM,
                                           self.INPUT_RANGE)

    def make_evaluation_datasets(self):
        eval_dataset = SupervisedDataSet(self.inputdim, self.outputdim)
        eval_costset = SupervisedDataSet(self.inputdim, self.outputdim)
        f_input = open('../data/funcvalue.txt', 'w')
        f_input_cost = open('../data/funccost.txt', 'w')
        points = self.generate_grid_points(PLOT_SAMPLES_AXIS)
        for point in points:
            z = self.fn_base(point)
            z_cost = self.cost_fn(point)
            point_str = str(point).strip('[]').replace(',', '')
            f_input.write('%s %f\n' % (point_str, z[0]))
            f_input_cost.write('%s %f\n' % (point_str, z_cost))
        f_input.close()
        f_input_cost.close()

        points = self.generate_grid_points(self.EVAL_SAMPLES_AXIS)
        for point in points:
            z = self.fn_base(point)
            z_cost = self.cost_fn(point)
            eval_dataset.addSample(point, z)
            eval_costset.addSample(point, [z_cost])
            
        return (eval_dataset, eval_costset)

    def fn_base_hills(self, inp):
        x = inp[0]
        y = inp[1]
        if x == 0:
            x = .0001
        if y == 0:
            y = .0001
        return [1 - 1.0 / (1.0 + x ** (-4)) - 1.0 / (1.0 + y ** (-4))]
    
    def fn_base_geom(self, inp):
        x = inp[0]
        y = inp[1]
        return [math.atan(x * y) + math.sin(x * y)]
    
    fn_base = fn_base_hills
    
    def fn_noise_zero(self, inp):
        return 0.0
    
    def fn_noise_constant(self, inp):
        return random.gauss(0.0, 3)
    
    def fn_noise_proportional(self, inp):
        base_value = self.fn_base(inp)[0] + 1 # ranges from 0 to 2
        return random.gauss(0.0, float(base_value) / 5)
    
    fn_noise = fn_noise_constant
    
    def fn(self, inp, num_samples):
        vals = []
        for i in range(num_samples): #@UnusedVariable
            val = self.fn_base(inp)[0] + self.fn_noise(inp)
            vals.append(val)
        avg = float(sum(vals)) / num_samples
        var = sum([(e - avg) ** 2 for e in vals]) / num_samples
        return ([avg], [var])
    
    def cost_fn_circles(self, inp):
        x = inp[0]
        y = inp[1]
        p1x = self.FIELD_HIGH_COST_CIRCLE1[0]; p1y = self.FIELD_HIGH_COST_CIRCLE1[1]
        p2x = self.FIELD_HIGH_COST_CIRCLE2[0]; p2y = self.FIELD_HIGH_COST_CIRCLE2[1]
        p3x = self.FIELD_HIGH_COST_CIRCLE3[0]; p3y = self.FIELD_HIGH_COST_CIRCLE3[1]
        radius = self.FIELD_HIGH_COST_RADIUS
        d1 = math.sqrt((x - p1x) ** 2 + (y - p1y) ** 2)
        d2 = math.sqrt((x - p2x) ** 2 + (y - p2y) ** 2)
        d3 = math.sqrt((x - p3x) ** 2 + (y - p3y) ** 2)
        if (d1 <= radius) or (d2 <= radius) or (d3 <= radius):
            return self.COST_HIGH
        else:
            return self.COST_LOW 
    
    def cost_fn_diag(self, inp):
        x = inp[0]
        y = inp[1]
        (frac, integ) = math.modf((x + y) / 7.0) #@UnusedVariable
        if frac < 0:
            frac += 1.0
        if (0 <= frac < .5):
            return self.COST_LOW
        else:
            return self.COST_HIGH
    
    def cost_fn_sides(self, inp):
        x = inp[0]
        if not (-self.FIELD_HIGH_COST_MARGIN <= x <= self.FIELD_HIGH_COST_MARGIN):
            return self.COST_HIGH
        else:
            return self.COST_LOW
        
    cost_fn = cost_fn_sides
    
class DomainSim(Domain):

    USE_MULTIPROCESSING = False
    NUM_TRIALS = 1

    INPUTDIM = 2
    OUTPUTDIM = 1

    COST_LOW = 1
    COST_HIGH = 10
    
    INPUT_RANGE = [None] * INPUTDIM
    INPUT_RANGE[0] = [-.3, -.1]
    INPUT_RANGE[1] = [-.1,  .1]

    EVAL_SAMPLES_AXIS = 31

    def __init__(self, mode):
        super(DomainSim, self).__init__(mode, self.INPUTDIM, self.OUTPUTDIM,
                                        self.INPUT_RANGE)
    
    def make_evaluation_datasets(self):
        eval_dataset = SupervisedDataSet(self.inputdim, self.outputdim)
        eval_costset = SupervisedDataSet(self.inputdim, self.outputdim)

        f_sim = open('simdata/evalset.txt')
        
        f_input = open('../data/funcvalue.txt', 'w')
        f_input_cost = open('../data/funccost.txt', 'w')
        for line in f_sim:
            line_segs = line.split()
            x = line_segs[0]
            y = line_segs[1]
            dist = float(line_segs[2])
            angle = line_segs[3]
            
            if dist < 0:
                cost = self.COST_HIGH
            else:
                cost = self.COST_LOW
                
            eval_dataset.addSample([x, y], [dist, angle])
            eval_costset.addSample([x, y], [cost])                        

            f_input.write('%s %s %f\n' % (x, y, dist))
            f_input_cost.write('%s %s %f\n' % (x, y, cost))
            
        f_input.close()
        f_input_cost.close()

        return (eval_dataset, eval_costset)

    def fn_sim(self, inp, num_samples):
        return SimulatorWrapper.run_simulator(inp, num_samples)
    
    fn_base = fn_sim
    fn = fn_sim
    
    def cost_fn_sim(self, inp):
        z = SimulatorWrapper.run_simulator(inp)
        return self.COST_LOW
        # FIXME:
        if z[0] < 0:
            return self.COST_HIGH
        else:
            return self.COST_LOW
    
    cost_fn = cost_fn_sim
    
#    def var_fn_sim(self, inp):
#        z = SimulatorWrapper.run_simulator(inp)
#        return z[2]
#    
#    var_fn = var_fn_sim
    
class Ensemble(object):
    
    def __init__(self, size, num_inp, num_hid1, num_hid2, num_out):
        self.size = size
        self.num_inp = num_inp
        self.num_hid1 = num_hid1
        self.num_hid2 = num_hid2
        self.num_out = num_out
        self.networks = []
        self.trainers = []
        self.starting_weights = []
        # initialize neural networks
        self.init()
    
    def init(self):
        self.networks = []
        self.trainers = []
        self.starting_weights = []
        for i in range(self.size): #@UnusedVariable
            if self.num_hid2 == 0:
                network = buildNetwork(self.num_inp, self.num_hid1, self.num_out,
                                       hiddenclass = SigmoidLayer, bias = True)
            else:
                network = buildNetwork(self.num_inp, self.num_hid1, self.num_hid2, self.num_out,
                                       hiddenclass = SigmoidLayer, bias = True)
            starting_weights = network.params.copy()
            trainer = BackpropTrainer(network,
                                      learningrate = LEARNING_RATE, 
                                      momentum = MOMENTUM_LOW, verbose = False)
            self.networks.append(network)
            self.trainers.append(trainer)
            self.starting_weights.append(starting_weights)
    
    def activate(self, inp):
        sumout = None
        for i in range(self.size):
            out = self.networks[i].activate(inp)
            if sumout is None:
                sumout = out
            else:
                sumout = [sum(pair) for pair in zip(sumout, out)]
        avgout = [e / float(self.size) for e in sumout]
        return avgout
    
    def getAmbiguity(self, inp):
        netout = self.activate(inp)
        sumvar = None
        for i in range(self.size):
            out = self.networks[i].activate(inp)
            var = [(a - b) ** 2 for (a, b) in zip(out, netout)]
            if sumvar is None:
                sumvar = var
            else:
                sumvar = [sum(pair) for pair in zip(sumvar, var)]
        avgvar = [e / float(self.size) for e in sumvar]
        return avgvar
    
    def train(self, ds, reinit = False):
        if reinit:
            self.init()
            
        for i in range(self.size):
            self.networks[i].params[:] = self.starting_weights[i]
            if len(ds) >= 5:
                self.trainers[i].trainUntilConvergence(ds,
                        continueEpochs = CONT_EPOCHS, maxEpochs = MAX_EPOCHS, 
                        validationProportion = VALIDATION_PROPORTION)            
            else:
                self.trainers[i].trainOnDataset(ds, MAX_EPOCHS)      

    def testOnData(self, ds):
        error = 0.0
        for i in range(self.size):
            error += self.trainers[i].testOnData(ds)
        return error / float(self.size)
    
    def save_starting_weights(self):
        for i in range(self.size): #@UnusedVariable
            self.starting_weights[i] = self.networks[i].params.copy()

class Experiment(object):

    EXP_RANDOM = 'random'
    EXP_ACTIVE = 'active'
    EXP_ACTCOST = 'actcost'
    EXP_ACTNET = 'actnet'
    EXP_ACTVAR = 'actvar'
    
    NUM_HIDDEN1 = 40
    NUM_HIDDEN2 = 20
    
    @classmethod
    def is_valid_mode(cls, mode):
        return (mode == cls.EXP_RANDOM) or (mode == cls.EXP_ACTIVE) or \
                (mode == cls.EXP_ACTCOST) or (mode == cls.EXP_ACTNET) or \
                (mode == cls.EXP_ACTVAR)
    
    def __init__(self, domain, mode, iters, ensemble_size, trial_number):
        self.domain = domain
        self.mode = mode
        self.iters = iters
        self.ensemble_size = ensemble_size
        self.trial_number = trial_number
        self.iteration = 0
        
        seed = abs(hash(self))
        numpy.random.seed(seed)
        random.seed(seed)
        seed = abs(hash(random.random()))
        numpy.random.seed(seed)
        random.seed(seed)
        print 'Seeding %d' % seed
        self.ensemble = Ensemble(self.ensemble_size, domain.inputdim, 
                                 self.NUM_HIDDEN1, self.NUM_HIDDEN2,
                                 domain.outputdim)
        (self.eval_dataset, self.eval_costset) = self.domain.make_evaluation_datasets()
        # used in run()
        self.train_dataset = SupervisedDataSet(domain.inputdim, domain.outputdim)
        self.current_error = 0.0
        self.current_avg_cost = 0.0
        self.current_error_times_avg_cost = 0.0
    
    @classmethod
    def new(cls, domain, mode, iters, trial_number):
        experiment = None
        if mode == cls.EXP_RANDOM:
            experiment = RandomExperiment(domain, iters, trial_number)
        elif mode == cls.EXP_ACTIVE:
            experiment = ActiveExperiment(domain, iters, trial_number)
        elif mode == cls.EXP_ACTCOST:
            experiment = ActCostExperiment(domain, iters, trial_number)
        elif mode == cls.EXP_ACTNET:
            experiment = ActNetExperiment(domain, iters, trial_number)
        elif mode == cls.EXP_ACTVAR:
            experiment = ActVarExperiment(domain, iters, trial_number)

        # saving parameter values to file
        if trial_number == 0:
            f_vars = open('../data/a-all-params.txt', 'w')
            for obj in [domain, experiment]:
                all_keys = dir(obj)
                for key in sorted(all_keys):
#                    if key == string.upper(key):
                    if not key.startswith('_'):
                        val = getattr(obj, key)
                        print '%s: %s' % (key, val)
                        f_vars.write('%s: %s\n' % (key, val))
            f_vars.close()
        return experiment
        
    def run(self):
    #    value_network = buildNetwork(2, 20, 1, hiddenclass = SigmoidLayer,
    #                                 bias = True)
    #    network_staring_weights = value_network.params.copy() 
    #    value_trainer = BackpropTrainer(value_network,
    #                                    learningrate = LEARNING_RATE, 
    #                                    momentum = MOMENTUM_LOW,
    #                                    verbose = False)
    
        points_filename = '../data/%s-points-%d.txt' % (self.mode, self.trial_number)
        f_points = open(points_filename, 'w')
        errors = []
        error_times_avg_costs = []
        avg_costs = []
        sum_cost = 0.
        for self.iteration in range(self.iters):
            print 'Iteration %d' % self.iteration
    #        current_error = value_trainer.testOnData(eval_dataset)
    #        print 'Error before: %.4f' % current_error 
    #        value_network.params[:] = network_staring_weights
    #        value_network = copy.deepcopy(value_network_base)
    
            print 'Requesting new training point...'
            point = self.get_training_point()
            
            z = self.domain.fn(point)
            z_cost = self.domain.cost_fn(point)
            sum_cost += z_cost
            point_str = str(point).strip('[]').replace(',', '')
            f_points.write('%s %f\n' % (point_str, z[0]))
            print 'Accepting %s with cost: %d' % (point, z_cost)
            self.train_dataset.addSample(point, z)
    
    #        value_trainer = BackpropTrainer(value_network,
    #                                        learningrate = LEARNING_RATE, 
    #                                        momentum = MOMENTUM_LOW,
    #                                        verbose = False)
    #        print 'training set length: ', len(self.train_dataset)
#            print 'Training ensemble with updated dataset...'

#            self.ensemble.train(self.train_dataset, reinit = True)
            self.ensemble.train(self.train_dataset)
            
            self.current_avg_cost = sum_cost / (self.iteration + 1)
#            print 'Testing ensemble on evaluation dataset...'
            self.current_error = self.ensemble.testOnData(self.eval_dataset)
            self.current_error_times_avg_cost = self.current_error * self.current_avg_cost
            errors.append(self.current_error)
            error_times_avg_costs.append(self.current_error_times_avg_cost)
            avg_costs.append(self.current_avg_cost)
            print 'Error: %.4f' % self.current_error
            print 'Error times cost: %.4f' % self.current_error_times_avg_cost
            print '---'
        
        f_points.close()
        
        # plot learned value function
        value_filename = '../data/%s-learnedvalue-%d.txt' % (self.mode, self.trial_number)
        self.write_plot_data(value_filename, self.ensemble)
        return (errors, error_times_avg_costs, avg_costs)

    def write_plot_data(self, filename, plot_ensemble):
        if self.trial_number == 0:
            print 'Writing plot data in %s for trial number: %d' % (filename, self.trial_number)
            points = self.domain.generate_grid_points(PLOT_SAMPLES_AXIS)
            f = open(filename, 'w')
            for point in points:
                z = plot_ensemble.activate(point)
                point_str = str(point).strip('[]').replace(',', '')
                f.write('%s %f\n' % (point_str, z[0]))
            f.close()

class RandomExperiment(Experiment):
    
    def __init__(self, domain, iters, trial_number):
        super(RandomExperiment, self).__init__(domain, Experiment.EXP_RANDOM, 
                iters, 1, trial_number)
        
    def get_training_point(self):
        points = self.domain.generate_random_points(1)
        return points[0]        

class ActiveExperiment(Experiment):
    
    def __init__(self, domain, iters, trial_number):
        super(ActiveExperiment, self).__init__(domain, Experiment.EXP_ACTIVE,
                iters, ACTIVE_ENSEMBLE_SIZE, trial_number)

    def get_training_point(self):
        best_point = None
        best_var = 0.
        points = self.domain.generate_random_points(ACTIVE_TRY_RAND_POINTS)
        for point in points:
            var = self.ensemble.getAmbiguity(point)
            if sum(var) > best_var:
                best_var = sum(var)
                best_point = point
        print 'Selecting %s, var: %.2f' % (best_point, best_var)
        return best_point

class ActCostExperiment(Experiment):
    
    def __init__(self, domain, iters, trial_number):
        super(ActCostExperiment, self).__init__(domain, Experiment.EXP_ACTCOST,
                iters, ACTIVE_ENSEMBLE_SIZE, trial_number)
        self.cost_ensemble = Ensemble(self.ensemble_size, domain.inputdim, 
                                      self.NUM_HIDDEN1, self.NUM_HIDDEN2, 1)
        self.train_costset = SupervisedDataSet(domain.inputdim, 1)
        
        # train cost network to reset costs
        points = self.domain.generate_grid_points(INIT_COST_SAMPLES_AXIS)
        init_costset = SupervisedDataSet(domain.inputdim, 1)
        for point in points:
            z_cost = self.domain.COST_LOW            
            init_costset.addSample(point, [z_cost])
        print 'Initializing Cost Ensemble...'
        self.cost_ensemble.train(init_costset)
        self.cost_ensemble.save_starting_weights()

    def compute_avgs(self):
#        print 'Computing averages...'
        sum_value_var = 0.
        sum_cost_var = 0.
        sum_cost = 0.
#        unit = (X_MAX - X_MIN) / (EVAL_SAMPLES_AXIS - 1)
#        for i in range(EVAL_SAMPLES_AXIS):
#            for j in range(EVAL_SAMPLES_AXIS):
#                x = X_MIN + i * unit
#                y = Y_MIN + j * unit
        points = self.domain.generate_random_points(ACTIVE_TRY_RAND_POINTS)
        for point in points:
            value_var = sum(self.ensemble.getAmbiguity(point))
            cost_var = sum(self.cost_ensemble.getAmbiguity(point))
            cost = sum(self.cost_ensemble.activate(point))
            sum_value_var += value_var
            sum_cost_var += cost_var
            sum_cost += cost
        avg_value_var = sum_value_var / float(ACTIVE_TRY_RAND_POINTS)
        avg_cost_var = sum_cost_var / float(ACTIVE_TRY_RAND_POINTS)
        avg_cost = sum_cost / float(ACTIVE_TRY_RAND_POINTS)
        print 'Avg value var: %.2f, Avg cost var: %.2f, Avg cost: %.2f' % \
                (avg_value_var, avg_cost_var, avg_cost)
        return (avg_value_var, avg_cost_var, avg_cost)

    def get_training_point(self):
        (avg_value_var, avg_cost_var, avg_cost) = self.compute_avgs() #@UnusedVariable
        best_point = None
        best_metric = -1.
        points = self.domain.generate_random_points(ACTIVE_TRY_RAND_POINTS)
        for point in points:
            value_var = sum(self.ensemble.getAmbiguity(point)) # / avg_value_var #/ 0.2823
#            cost = clamp(sum(self.cost_ensemble.activate(point)), COST_LOW, COST_HIGH)
            cost = sum(self.cost_ensemble.activate(point))
#            cost_var = sum(self.cost_ensemble.getAmbiguity([x, y])) #/ avg_cost_var #/ 5.4719
#            cost_adjusted = cost / (cost_var + 1.0)
            cost_adjusted = cost # * cost / math.sqrt(cost_var + 1.0)
            new_metric = ((value_var / avg_value_var) ** 2) / cost_adjusted
            if new_metric > best_metric:
                best_metric = new_metric
                best_point = point
        
        value_var = sum(self.ensemble.getAmbiguity(best_point)) #/ avg_value_var #/ 0.2823
        cost = sum(self.cost_ensemble.activate(best_point))
        z_cost = self.domain.cost_fn(best_point)
        cost_var = sum(self.cost_ensemble.getAmbiguity(best_point)) #/ avg_cost_var #/ 5.4719
#        cost_adjusted = cost / (cost_var + 1.0)
        cost_adjusted = cost # * cost / math.sqrt(cost_var + 1.0)
        new_metric = value_var / cost_adjusted
        print 'Selecting %s, var: %.2f, avg: %.2f,' % \
                (best_point, value_var, avg_value_var)
        print '... cost: %.2f (%d), avg: %.2f, cost var: %.2f, metric: %.2f' % \
                (cost, z_cost, avg_cost, cost_var, new_metric)

        # train the cost network before returning selected point     
        self.train_costset.addSample(best_point, [z_cost])
        self.cost_ensemble.train(self.train_costset)
        # if it is the last iteration, write_plot_data learned cost
        if self.iteration == self.iters - 1:
            cost_filename = '../data/%s-learnedcost-%d.txt' % (self.mode, self.trial_number)
            self.write_plot_data(cost_filename, self.cost_ensemble)

        return best_point

class ActNetExperiment(Experiment):
    
    def __init__(self, domain, iters, trial_number):
        super(ActNetExperiment, self).__init__(domain, Experiment.EXP_ACTNET,
                iters, ACTIVE_ENSEMBLE_SIZE, trial_number)
        # inputs: x, y, point's ambiguity, current average cost, current average value variance
        # output: ratio of next error * avg_cost to current
        self.cost_ensemble = Ensemble(self.ensemble_size, domain.inputdim, 
                                      self.NUM_HIDDEN1, self.NUM_HIDDEN2, 1)
        self.train_costset = SupervisedDataSet(domain.inputdim, 1)
        
        # train cost network to reset costs
        points = self.domain.generate_grid_points(INIT_COST_SAMPLES_AXIS)
        init_costset = SupervisedDataSet(2, 1)
        for point in points:
            z_cost = self.domain.COST_LOW            
            init_costset.addSample(point, [z_cost])

        print 'Initializing Cost Ensemble...'
        self.cost_ensemble.train(init_costset)
        self.cost_ensemble.save_starting_weights()


        self.perf_input_dim = 4
        self.perf_ensemble = Ensemble(self.ensemble_size, self.perf_input_dim,
                self.NUM_HIDDEN1, self.NUM_HIDDEN2, 1)
        self.train_inputs = []
        self.train_outputs = []
        #self.train_perfset = ImportanceDataSet(self.perf_input_dim, 1)
        self.train_perfset = SupervisedDataSet(self.perf_input_dim, 1)
        
        self.last_avg_value_var = None
        self.last_x_y_value_var = None
        self.last_x_y_cost = None
        self.last_x_y_actual_cost = None
        self.last_x_y_cost_var = None
        self.last_error_times_avg_cost = None
        self.last_predicted_drop = -1
        
        # train perf network to reset predictions
        #init_perfset = SupervisedDataSet(self.perf_input_dim, 1)
        init_perfset = SupervisedDataSet(self.perf_input_dim, 1)
        for i in range(INIT_PERF_SAMPLES): #@UnusedVariable
            x_y_value_var = random.uniform(-.1, 5)
            avg_value_var = random.uniform(-.1, 5)
            cost_gutter = float(self.domain.COST_HIGH - self.domain.COST_LOW) / 10
            x_y_cost = random.uniform(self.domain.COST_LOW - cost_gutter, 
                                      self.domain.COST_HIGH + cost_gutter)
            x_y_cost_var = random.uniform(-.1, 20)
            inp = [x_y_value_var, avg_value_var, x_y_cost, x_y_cost_var]
            out = [2]
            init_perfset.addSample(inp, out)
        print 'Initializing Perf Ensemble...'
        self.perf_ensemble.train(init_perfset)
        self.perf_ensemble.save_starting_weights()

    def compute_avg_value_var(self):
#        print 'Computing average value variance...'
        sum_value_var = 0.
#        sum_cost_var = 0.
        points = self.domain.generate_random_points(ACTIVE_TRY_RAND_POINTS)
        for point in points:
            value_var = sum(self.ensemble.getAmbiguity(point))
#            cost_var = sum(self.perf_ensemble.getAmbiguity([x, y]))
            sum_value_var += value_var
#            sum_cost_var += cost_var
        avg_value_var = sum_value_var / float(ACTIVE_TRY_RAND_POINTS)
#        avg_cost_var = sum_cost_var / float(ACTIVE_TRY_RAND_POINTS)
        print 'Avg value var: %.2f' % avg_value_var
#        print 'Avg cost var: %.2f' % avg_cost_var
#        return (avg_value_var, avg_cost_var)
        return avg_value_var

    def get_training_point(self):
        if len(self.train_dataset) >= 10:
            # train the perf network based on previous selection's performance
            # before selecting new sample point
            error_times_avg_cost_drop = self.current_error_times_avg_cost / self.last_error_times_avg_cost
            print 'Result: value var: %.2f, avg: %.2f, cost: %.2f (%d), cost var: %.2f...' % \
                    (self.last_x_y_value_var, self.last_avg_value_var, self.last_x_y_cost, self.last_x_y_actual_cost, self.last_x_y_cost_var)
            print '=> Drop: %.2f (Predicted: %.2f)' % \
                    (error_times_avg_cost_drop, self.last_predicted_drop)
                    
            inp = [self.last_x_y_value_var, self.last_avg_value_var, 
                   self.last_x_y_cost, self.last_x_y_cost_var]
            out = [(1 - error_times_avg_cost_drop) * 2]
            self.train_inputs.append(inp)
            self.train_outputs.append(out)
            self.train_perfset.addSample(inp, out)
            
#            self.train_perfset = ImportanceDataSet(self.perf_input_dim, 1)
            """            
            imp = 1.0
            for i in reversed(range(len(self.train_inputs))):
                inp = self.train_inputs[i]
                out = self.train_outputs[i]
                self.train_perfset.addSample(inp, out, imp)
                print 'Adding sample %d with importance %.2f' % (i, imp)
                print '%s -> %s' % (self.train_inputs[i], self.train_outputs[i])
                imp *= 0.90

            imp = 1.0
            for i in range(len(self.train_inputs)):
                inp = self.train_inputs[i]
                out = self.train_outputs[i]
                self.train_perfset.addSample(inp, out, imp)
                print 'Adding sample %d with importance %.2f' % (i, imp)
                print '%s -> %s' % (self.train_inputs[i], self.train_outputs[i])
                imp *= 1.10
            """           
           
#            print 'Training...'
            self.perf_ensemble.train(self.train_perfset)
#            print 'Training done.'
            
        # select new point
        avg_value_var = self.compute_avg_value_var()
        best_point = None
        if len(self.train_dataset) >= 5:
            # use the perf net to predict drop
            best_metric = -1000
            points = self.domain.generate_random_points(ACTIVE_TRY_RAND_POINTS)
            for point in points:
                x_y_value_var = sum(self.ensemble.getAmbiguity(point))
                x_y_cost = sum(self.cost_ensemble.activate(point))
                x_y_cost_var = sum(self.cost_ensemble.getAmbiguity(point))
                inp = [x_y_value_var, avg_value_var, x_y_cost, x_y_cost_var]
                new_metric = sum(self.perf_ensemble.activate(inp))
                if new_metric > best_metric:
                    best_metric = new_metric
                    best_point = point
        else:
            # use vanilla active sampling
            best_var = 0.
            points = self.domain.generate_random_points(ACTIVE_TRY_RAND_POINTS)
            for point in points:
                var = self.ensemble.getAmbiguity(point)
                if sum(var) > best_var:
                    best_var = sum(var)
                    best_point = point
        
        # print metrics
        x_y_value_var = sum(self.ensemble.getAmbiguity(best_point))
        x_y_cost = sum(self.cost_ensemble.activate(best_point))
        x_y_actual_cost = self.domain.cost_fn(best_point)
        x_y_cost_var = sum(self.cost_ensemble.getAmbiguity(best_point))
        inp = [x_y_value_var, avg_value_var, x_y_cost, x_y_cost_var]
        predicted_drop = 1.0 - (sum(self.perf_ensemble.activate(inp)) / 2.0)
        self.last_predicted_drop = predicted_drop
        print 'Selecting %s, var: %.4f, avg: %.2f, cost: %.2f (%d), predicted drop: %.2f' % \
                (best_point, x_y_value_var, avg_value_var, x_y_cost,
                x_y_actual_cost, predicted_drop)
                
        # train costset before returning point
        z_cost = self.domain.cost_fn(best_point)
        self.train_costset.addSample(best_point, [z_cost])
        self.cost_ensemble.train(self.train_costset)
        
        # saving current values in last_* variables
        self.last_x_y_value_var = x_y_value_var
        self.last_avg_value_var = avg_value_var
        self.last_x_y_cost = x_y_cost
        self.last_x_y_actual_cost = x_y_actual_cost
        self.last_x_y_cost_var = x_y_cost_var
        self.last_error_times_avg_cost = self.current_error_times_avg_cost
        return best_point

class ActVarExperiment(Experiment):
    
    def __init__(self, domain, iters, trial_number):
        super(ActVarExperiment, self).__init__(domain, Experiment.EXP_ACTVAR,
                iters, ACTIVE_ENSEMBLE_SIZE, trial_number)
        self.var_ensemble = Ensemble(self.ensemble_size, domain.inputdim, self.NUM_HIDDEN1, self.NUM_HIDDEN2, 1)
        self.train_varset = SupervisedDataSet(domain.inputdim, 1)
        
        # train cost network to reset costs
        points = self.domain.generate_grid_points(INIT_COST_SAMPLES_AXIS)
        init_varset = SupervisedDataSet(domain.inputdim, 1)
        for point in points:
            z_var = 1.0
            init_varset.addSample(point, [z_var])
        print 'Initializing Variance Ensemble...'
#        self.var_ensemble.train(init_varset)
#        self.var_ensemble.save_starting_weights()

    def compute_avgs(self):
#        print 'Computing averages...'
        sum_value_var = 0.
        sum_value = 0.
        sum_var = 0.
#        unit = (X_MAX - X_MIN) / (EVAL_SAMPLES_AXIS - 1)
#        for i in range(EVAL_SAMPLES_AXIS):
#            for j in range(EVAL_SAMPLES_AXIS):
#                x = X_MIN + i * unit
#                y = Y_MIN + j * unit
        points = self.domain.generate_random_points(ACTIVE_TRY_RAND_POINTS)
        for point in points:
            value_var = sum(self.ensemble.getAmbiguity(point))
            value = sum(self.ensemble.activate(point))
            var = sum(self.var_ensemble.activate(point))
            sum_value_var += value_var
            sum_value += value
            sum_var += var
        avg_value_var = sum_value_var / float(ACTIVE_TRY_RAND_POINTS)
        avg_value = sum_value / float(ACTIVE_TRY_RAND_POINTS)
        avg_var = sum_var / float(ACTIVE_TRY_RAND_POINTS)
        print 'Avg value var: %.4f, Avg value: %.4f, Avg predicted var: %.4f' % \
                (avg_value_var, avg_value, avg_var)
        return (avg_value_var, avg_value, avg_var)

    def get_training_point(self):
        (avg_value_var, avg_value, avg_var) = self.compute_avgs()
        best_point = None
        best_metric = -1000.
        points = self.domain.generate_random_points(ACTIVE_TRY_RAND_POINTS)
        for point in points:
            value_var = sum(self.ensemble.getAmbiguity(point))
            value = sum(self.ensemble.activate(point))
            var = max(sum(self.var_ensemble.activate(point)), 0.0)
#            cost_var = sum(self.cost_ensemble.getAmbiguity([x, y])) #/ avg_cost_var #/ 5.4719
#            cost_adjusted = cost / (cost_var + 1.0)
            new_metric = (value_var * value_var) / (var + 1.0)
#            new_metric = (value_var/avg_value_var + value/avg_value + var/avg_var)
            if new_metric > best_metric:
                best_metric = new_metric
                best_point = point 
                
        value_var = sum(self.ensemble.getAmbiguity(best_point))
        value = sum(self.ensemble.activate(best_point))
        var = max(sum(self.var_ensemble.activate(best_point)), 0.0)
        new_metric = (value_var / avg_value_var + value / avg_value + var / avg_var)

        print 'Selecting %s, ensemble var: %.2f, value: %.2f, predicted var: %.2f, metric: %.2f' % \
                (best_point, value_var, value, var, new_metric)

        # train the var network before returning selected point     
        z_var = self.domain.var_fn(best_point)
        self.train_varset.addSample(best_point, [z_var])
        self.var_ensemble.train(self.train_varset)
        # if it is the last iteration, write_plot_data learned cost
        if self.iteration == self.iters - 1:
            var_filename = '../data/%s-learnedvar-%d.txt' % (self.mode, self.trial_number)
            self.write_plot_data(var_filename, self.var_ensemble)

        return best_point

def one_experiment_trial((domain, exp_mode, trial_number)):
    experiment = Experiment.new(domain, exp_mode, domain.NUM_ITERS, trial_number)
    return experiment.run()

class Trial(object):
    
    def __init__(self, exp_mode, domain_mode):
        self.exp_mode = exp_mode
        self.domain_mode = domain_mode
        
    def run(self):
        domain = Domain.new(self.domain_mode)
        num_trials = domain.NUM_TRIALS
        if domain.USE_MULTIPROCESSING:
            pool = multiprocessing.Pool(processes = domain.NUM_CORES)
            params = []
            for trial in range(num_trials):
                params.append((domain, self.exp_mode, trial))
            results = pool.map(one_experiment_trial, params)
        else:
            results = []
            for trial in range(num_trials):
                result = one_experiment_trial((domain, self.exp_mode, trial))
                results.append(result)
                
        sum_errors = None
        sum_errortimescosts = None
        sum_costs = None
        for trial in range(num_trials):
            trial_errors = results[trial][0]
            trial_errortimescosts = results[trial][1]
            trial_costs = results[trial][2]
            if sum_errors is None:
                sum_errors = trial_errors
            else:
                sum_errors = [sum(pair) for pair in zip(sum_errors, trial_errors)]
            if sum_errortimescosts is None:
                sum_errortimescosts = trial_errortimescosts
            else:
                sum_errortimescosts = [sum(pair) for pair in zip(sum_errortimescosts, trial_errortimescosts)]
            if sum_costs is None:
                sum_costs = trial_costs
            else:
                sum_costs = [sum(pair) for pair in zip(sum_costs, trial_costs)]
        
        avg_errors = [e / float(num_trials) for e in sum_errors]
        avg_errortimescosts = [e / float(num_trials) for e in sum_errortimescosts]
        avg_costs = [e / float(num_trials) for e in sum_costs]

        error_filename = '../data/%s-error.txt' % self.exp_mode
        errortimescost_filename = '../data/%s-errortimescost.txt' % self.exp_mode
        cost_filename = '../data/%s-cost.txt' % self.exp_mode
        f_error = open(error_filename, 'w')
        f_errortimescost = open(errortimescost_filename, 'w')
        f_cost = open(cost_filename, 'w')
        for iteration in range(len(avg_errors)):
            f_error.write('%d %f\n' % (iteration, avg_errors[iteration]))
            f_errortimescost.write('%d %f\n' % (iteration, avg_errortimescosts[iteration]))
            f_cost.write('%d %f\n' % (iteration, avg_costs[iteration]))
        f_error.close()
        f_errortimescost.close()
        f_cost.close()
        
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'Please supply the experiment and domain.'
    else:
        exp_mode = sys.argv[1]
        domain_mode = sys.argv[2]
        if Experiment.is_valid_mode(exp_mode) and Domain.is_valid_mode(domain_mode):
            # running experiment trials
            trial = Trial(exp_mode, domain_mode)
            trial.run()
        else:
            print 'Invalid parameters: %s %s' % (exp_mode, domain_mode)
        
#    experiment = ActiveExperiment(NUM_ITERS)
#    experiment.run()
#    base_experiment()

