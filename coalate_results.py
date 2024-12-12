import sys
# setting path
sys.path.append('./MainAlgo')
sys.path.append('./Common')
sys.path.append('./Baseline4')
sys.path.append('./Baseline3')
sys.path.append('./Baseline2')
sys.path.append('./Baseline1')

from MainAlgo.online_main import Online_Algo
from Baseline4.online_main import BL4_Online_Algo
from Baseline3.online_main import BL3_Online_Algo
from Baseline2.online_main import BL2_Online_Algo
from Baseline1.online_main import BL1_Online_Algo

from common_config import *
from online_const import *
from common_utils import *
from plots import *

class CoalatedResultsModel:
    def __init__(self) -> None:
        self.numUsers = common_NUM_USERS

        self.Rewards = []
        self.runningAvg = []

        self.Reward_online = []
        self.runningAvg_online = []
        # self.NSreward_online = []
        # self.REreward_online = []

        self.mean_Reward_online = None

        self.mean_REreward_online = None
        self.mean_REmigration_online = None
        self.mean_REdelay_online = None
        self.mean_REstorage_online = None
        self.mean_REcompDelay_online = None
        
        self.mean_NSreward_online = None
        self.mean_NSmigration_online = None
        self.mean_NSdelay_online = None
        self.mean_NSstorage_online = None
        self.mean_NScompDelay_online = None


        self.mean_Reward_online_over_runs = []

        self.mean_REreward_online_over_runs = []
        self.mean_REmigration_online_over_runs = []
        self.mean_REdelay_online_over_runs = []
        self.mean_REstorage_online_over_runs = []
        self.mean_REcompDelay_online_over_runs = []
        
        self.mean_NSreward_online_over_runs = []
        self.mean_NSmigration_online_over_runs = []
        self.mean_NSdelay_online_over_runs = []
        self.mean_NSstorage_online_over_runs = []
        self.mean_NScompDelay_online_over_runs = []
        
        self.trainer_obj_present = True

    def add_results(self, trainer, online_obj):
        if trainer:
            self.Rewards.append(trainer.Rewards)
            self.runningAvg.append(trainer.runningAvg)
        else:
            self.trainer_obj_present = False
            
        self.Reward_online.append(online_obj.Reward_online)
        self.runningAvg_online.append(online_obj.runningAvg_online)
        # self.min_ns_rew_len = self.min_ns_rew_len if len(online_obj.NSreward_online)
        # self.min_re_rew_len = self.min_re_rew_len if len(online_obj.REreward_online)
        # self.NSreward_online.append(online_obj.NSreward_online)
        # self.REreward_online.append(online_obj.REreward_online)

        self.mean_Reward_online_over_runs.append(online_obj.mean_Reward_online)
        self.mean_REreward_online_over_runs.append(online_obj.mean_REreward_online)
        self.mean_REmigration_online_over_runs.append(online_obj.mean_REmigration_online)
        self.mean_REdelay_online_over_runs.append(online_obj.mean_REdelay_online)
        self.mean_REstorage_online_over_runs.append(online_obj.mean_REstorage_online)
        self.mean_REcompDelay_online_over_runs.append(online_obj.mean_REcompDelay_online)
        self.mean_NSreward_online_over_runs.append(online_obj.mean_NSreward_online)
        self.mean_NSmigration_online_over_runs.append(online_obj.mean_NSmigration_online)
        self.mean_NSdelay_online_over_runs.append(online_obj.mean_NSdelay_online)
        self.mean_NSstorage_online_over_runs.append(online_obj.mean_NSstorage_online)
        self.mean_NScompDelay_online_over_runs.append(online_obj.mean_NScompDelay_online)

    def stack_n_mean(self, var, axis = 2):
        return np.mean(np.dstack(var),axis = axis)
    
    def generate_results(self):
        if self.trainer_obj_present:
            self.Rewards = self.stack_n_mean(self.Rewards)
            self.runningAvg = self.stack_n_mean(self.runningAvg)
            # self.NSreward_online = self.stack_n_mean(self.NSreward_online)
            # self.REreward_online = self.stack_n_mean(self.REreward_online)
        else:
            self.Rewards = None
            self.runningAvg = None
        
        self.Reward_online = self.stack_n_mean(self.Reward_online)
        self.runningAvg_online = self.stack_n_mean(self.runningAvg_online)
            
        self.mean_Reward_online = mean(self.mean_Reward_online_over_runs)
        
        self.mean_REreward_online = mean(self.mean_REreward_online_over_runs)
        self.mean_REmigration_online = mean(self.mean_REmigration_online_over_runs)
        self.mean_REdelay_online = mean(self.mean_REdelay_online_over_runs)
        self.mean_REstorage_online = mean(self.mean_REstorage_online_over_runs)
        self.mean_REcompDelay_online = mean(self.mean_REcompDelay_online_over_runs)
        
        self.mean_NSreward_online = mean(self.mean_NSreward_online_over_runs)
        self.mean_NSmigration_online = mean(self.mean_NSmigration_online_over_runs)
        self.mean_NSdelay_online = mean(self.mean_NSdelay_online_over_runs)
        self.mean_NSstorage_online = mean(self.mean_NSstorage_online_over_runs)
        self.mean_NScompDelay_online = mean(self.mean_NScompDelay_online_over_runs)

        self.stdev_Reward_online = stdev(self.mean_Reward_online_over_runs)
        
        self.stdev_REreward_online = stdev(self.mean_REreward_online_over_runs)
        self.stdev_REmigration_online = stdev(self.mean_REmigration_online_over_runs)
        self.stdev_REdelay_online = stdev(self.mean_REdelay_online_over_runs)
        self.stdev_REstorage_online = stdev(self.mean_REstorage_online_over_runs)
        self.stdev_REcompDelay_online = stdev(self.mean_REcompDelay_online_over_runs)
        
        self.stdev_NSreward_online = stdev(self.mean_NSreward_online_over_runs)
        self.stdev_NSmigration_online = stdev(self.mean_NSmigration_online_over_runs)
        self.stdev_NSdelay_online = stdev(self.mean_NSdelay_online_over_runs)
        self.stdev_NSstorage_online = stdev(self.mean_NSstorage_online_over_runs)
        self.stdev_NScompDelay_online = stdev(self.mean_NScompDelay_online_over_runs)

class Coalate_Results:
    def __init__(self):
        self.algos_names = ALGO_NAMES
        self.algo_names_to_objects = {}
    
    def load_data_from_dir(self, path, algo):
        trainer = load_object(path+"trainer_obj.pkl")
        online_obj = load_object(path+"online_obj.pkl")
        algo.add_results(trainer, online_obj)

    def coalate(self):
        for algo in self.algos_names:
            path = create_dir(dir_path = "./" + algo + "/diags/", generate_path = False)
            # the path which is scanned for the objects of various runs
            obj = CoalatedResultsModel()
            self.algo_names_to_objects[algo] = obj
            iteration_dirs = [os.path.join(path,filename) for filename in os.listdir(path) if os.path.isdir(os.path.join(path,filename)) and filename.isdigit()]
            for sub_dir in iteration_dirs:
                # open pkl files in these, load the objects, add to class variables
                print(sub_dir + "/")
                self.load_data_from_dir(sub_dir + "/", obj)
        
                # now we have all the trained and online algorithms with us,
                # find averages for them, and generate results
                
                # All the matrices will have the following dimensions:
                # 1. Users
                # 2. Time Steps
                # 3. Iterations -> new dimension added

            # Earlier, the plots were plotted iterating over the users
            # Same plots can be plotted after averaging over the 3rd dimension, which is the user's dimension
            obj.generate_results()
            if not obj.Rewards is None:
                gen_plot_rew(obj,dir=path)
            if not obj.runningAvg is None:
                gen_plot_runningAvg(obj,dir=path)

            initial_setup(None, dir = path)
            gen_online_plots(obj, dir = path)
            clean()

        # Now we have results of all the 4 algorithms, their mean, their stdevs
        # We need to generate the bar graphs now.
        dir = create_dir()
        generate_bar_graphs(self, dir)

if __name__ == '__main__':
    runner = Coalate_Results()
    runner.coalate()