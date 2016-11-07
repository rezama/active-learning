
import os
import random

IGNORE_NEGATIVE = True

SIM_VAR_NAME1 = 'kick_xOff'
SIM_VAR_NAME2 = 'kick_yOff'

# set up for running on cs network
SIM_INPUT = '~/active/simdata/simin.txt'
SIM_OUTPUT = '~/active/simdata/simout.txt'
SIM_EXEC = 'rm %s; ~/villasim/agents/nao-agent/optimization/condor-run.sh %s %s >/dev/null 2>&1'

class SimulatorWrapper(object):
    last_sim_x = None
    last_sim_y = None
    last_sim_out1 = None
    last_sim_out2 = None
    last_sim_var1 = None
    last_sim_var2 = None

    @classmethod
    def run_simulator(cls, inp, num_samples):
        x = inp[0]
        y = inp[1]
        if (x != cls.last_sim_x) or (y != cls.last_sim_y):
            cls.last_sim_x = x
            cls.last_sim_y = y
            random_suffix = random.randint(1000, 2000)
            sum_out1 = 0
            sum_out2 = 0
            simin_file = '%s-%d' % (SIM_INPUT, random_suffix)
            print 'Writing params to %s' % simin_file
            f = open(simin_file, 'w')
            f.write('%s\t%f\n' % (SIM_VAR_NAME1, x))
            f.write('%s\t%f\n' % (SIM_VAR_NAME2, y))
            f.close()
            num_failed = 0
            set_out1 = []
            set_out2 = []
            for i in range(num_samples):
                simout_file = '%s-%d-%d' % (SIM_OUTPUT, random_suffix, i)
    
                command = SIM_EXEC % (simout_file, simin_file, simout_file)
                print command
                os.system(command)
                
                print 'Reading values from %s' % simout_file
                try:
                    f = open(simout_file, 'r')
                    line1 = f.readline().rstrip()
                    out1 = float(line1)
                    line2 = f.readline().rstrip()
                    out2 = float(line2)
                    out2 = 0.0
                    f.close()
                    set_out1.append(out1)
                    set_out2.append(out2)
                    sum_out1 += out1
                    sum_out2 += out2
                except IOError as e:
                    print 'Oh dear.'
                    num_failed += 1
                
            avg_out1 = sum_out1 / (num_samples - num_failed)
            avg_out2 = sum_out2 / (num_samples - num_failed)

            var_out1 = sum([(e - avg_out1) ** 2 for e in set_out1]) / (num_samples - num_failed)
            var_out2 = sum([(e - avg_out2) ** 2 for e in set_out2]) / (num_samples - num_failed)
            
            if IGNORE_NEGATIVE and (avg_out1 < 0):
                avg_out1 = 0

            cls.last_sim_out1 = avg_out1
            cls.last_sim_out2 = avg_out2
            cls.last_sim_var1 = var_out1
            cls.last_sim_var2 = var_out2
            print 'Simulator out1:', set_out1
            print 'Simulator returning [%.2f, %.2f], var [%.4f, %.4f]' % \
                    (cls.last_sim_out1, cls.last_sim_out2, cls.last_sim_var1, cls.last_sim_var2)
            
#        return [cls.last_sim_out1, cls.last_sim_out2], [cls.last_sim_var1, cls.last_sim_var2]
        return [cls.last_sim_out1], [cls.last_sim_var1]
    
if __name__ == '__main__':
    pass