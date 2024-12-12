import sys
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from LATENCY_MATRIX_CONFIG import *
from STORAGE_MATRIX_CONFIG import *
import os
NUM_USERS = 13
NUM_ACCESS_POINTS = 9
NUM_ACTIONS_PER_USER = (NUM_ACCESS_POINTS*(NUM_ACCESS_POINTS+1))
NO_BACKUP = 100
PR_LU_TRANS = np.array([
    [0.15,0.05,0.1,0.1,0.08,0.12,0.09,0.11,0.2],
    [0.1,0.08,0.12,0.09,0.11,0.03,0.17,0.15,0.15],
    [0.15,0.11,0.03,0.17,0.05,0.1,0.1,0.08,0.21],
    [0.15,0.05,0.1,0.1,0.11,0.03,0.17,0.08,0.21],
    [0.15,0.05,0.1,0.1,0.08,0.12,0.17,0.09,0.14],
    [0.12,0.09,0.15,0.05,0.1,0.1,0.08,0.11,0.2],
    [0.05,0.1,0.1,0.08,0.12,0.15,0.09,0.11,0.2],
    [0.09,0.11,0.03,0.15,0.05,0.1,0.1,0.08,0.29],
    [0.08,0.15,0.05,0.1,0.1,0.12,0.09,0.11,0.2]])
common_TRIALS = 15
common_TRIAL_LEN = 100#
totalTS=common_TRIAL_LEN*common_TRIALS
NUM_STATES_PER_USER = (NUM_ACCESS_POINTS*NUM_ACCESS_POINTS*2*(NUM_ACCESS_POINTS+1))
TOT_NUM_STATES = NUM_STATES_PER_USER^NUM_USERS
LATENCY_MATRIX = [[2.55316, 9.70525, 7.02418, 2.50611, 8.97695, 4.81203, 6.93397, 6.21317, 7.60077],
                  [9.70525, 2.55389, 6.25835, 11.8198, 7.18656, 3.92144, 6.99266, 6.2124, 6.71018],
                  [7.02418, 6.26684, 2.55398, 4.02147, 3.27882, 2.77404, 4.02353, 5.07656, 2.77417],
                  [2.50611, 11.8198, 4.02147, 2.55354, 9.28922, 7.44114, 9.19819, 4.21866, 11.8622],
                  [8.97695, 7.18656, 3.27882, 9.28922, 2.55301, 3.26512, 2.0394, 5.5403, 5.58373],
                  [4.75385, 3.92267, 2.77457, 7.44114, 3.26512, 2.55366, 1.58549, 3.0197, 2.78874],
                  [6.91204, 7.6288, 4.2441, 9.19819, 2.0394, 1.58474, 2.55359, 3.95576, 4.37348],
                  [6.21317, 6.2124, 5.31069, 4.21866, 6.59593, 3.06604, 3.95576, 2.55396, 5.85478],
                  [7.60077, 6.71018, 2.77417, 11.8622, 5.55827, 2.78874, 4.37348, 5.85478, 2.55313]]







AP_CAPACITY = 25
AP_CAPACITIES = np.ones((NUM_ACCESS_POINTS))*AP_CAPACITY
AP_USER_LOADS = np.zeros((NUM_ACCESS_POINTS))
USER_LOADS = np.ones((NUM_USERS))
WEAK_AP_LIST = {1,6,8}
common_FAILED_AP_SERV_LOC_REW = 0
LOCATION_BASED_FAILURE_TIME = 2
actualEPS = 0.01#0.15#
NEGATIVE_REWARD = -8000
LOCATION_BASED_FAILURE_ENABLED = True

def gen_per_user_states_list():
    rsList = set()
    back_up_options = list(range(NUM_ACCESS_POINTS))
    back_up_options.append(NO_BACKUP)
    combinations = [
    list(range(NUM_ACCESS_POINTS)),#[0,1, 2],
    list(range(NUM_ACCESS_POINTS)),#[0,1,2],
    [0,1], # Anomaly/Not
    back_up_options # Backup_loc: [0,1, 2]
    ]
    stateList0=np.zeros((NUM_STATES_PER_USER,4), dtype = int)
    a=0
    for element in itertools.product(*combinations):
        stateList0[a,:]=element[:]
        if stateList0[a,2]==1:
            rsList.add(a)
        a+=1
    return stateList0, rsList
PER_USER_STATE_LIST, RARE_STATES = gen_per_user_states_list()

def get_state_id(newUserLoc, newSvcLoc, anomaly, newBackupLoc):
    stateName=np.array([int(newUserLoc),int(newSvcLoc),int(anomaly),int(newBackupLoc)])
    new_state=np.where(np.all(PER_USER_STATE_LIST==stateName,axis=1))[0][0]
    return new_state

def gen_per_user_actions_list():
    back_up_options = list(range(NUM_ACCESS_POINTS))
    back_up_options.append(NO_BACKUP)
    actlists = [
        list(range( NUM_ACCESS_POINTS)),#[0,1, 2],
        back_up_options
    ]
    actList0=np.zeros((NUM_ACTIONS_PER_USER,2))
    a=0
    # b=0
    for element in itertools.product(*actlists):
        actList0[a,:]=element
        a=a+1
    # print(actList0)
    return actList0
PER_USER_ACTION_LIST = gen_per_user_actions_list()

class Combined_Online_Algo():
    def __init__(self, tot_users = NUM_USERS, users_to_care_about = NUM_USERS):
        self.numAP=NUM_ACCESS_POINTS
        self.numActions = NUM_ACTIONS_PER_USER
        self.noBackup= NO_BACKUP
        self.tot_users = tot_users
        self.users_to_care_about = users_to_care_about
        self.failed_APs = {}

        self.combined_data = []  # List to hold [latency_cost, backup_decision, location_index]
        self.backup_counts = [0] * NUM_ACCESS_POINTS
        self.pr_lu_trans_online = PR_LU_TRANS
        print(self.pr_lu_trans_online)
        print("--------------------------")
        
        self.stateList = PER_USER_STATE_LIST
        self.actList=PER_USER_ACTION_LIST

        self.Reward_online = np.zeros((totalTS, self.users_to_care_about))
        self.runningAvg_online = np.zeros((totalTS))
        self.Action=np.zeros((totalTS+1, self.users_to_care_about))

        self.REreward_online=[]
        self.REmigration_online=[]
        self.REdelay_online=[]
        self.REstorage_online=[]
        self.REcompDelay_online=[]

        self.NSreward_online=[]
        self.NSmigration_online=[]
        self.NSdelay_online=[]
        self.NSstorage_online=[]
        self.NScompDelay_online=[]

        MIGR_COST = np.zeros((self.numAP, self.numAP))
        STORAGE_COST_temp = np.zeros((self.numAP))
        for p in range(self.numAP):
          diff = random.uniform(-0.5, 0.5)
          STORAGE_COST_temp[p] = 5+diff
          for q in range(self.numAP):
            diff = random.uniform(-0.5, 0.5)
            MIGR_COST[p][q] = LATENCY_MATRIX[p][q]+diff


        self.storageCost_online=STORAGE_RANDOM_HIGH_RANGE_1
        self.storageCost_name = "STORAGE_RANDOM_HIGH_RANGE_1"
        # self.storageCost_online=STORAGE_COST_temp

        # self.commDelay_online=LATENCY_MATRIX#np.array([[2.55316,9.70525, 7.02418], [9.70525, 2.55389,6.25835], [7.02418,6.26684,2.55398]])
        self.commDelay_online=LATENCY_MATRIX_RANDOM_3
        self.commDelay_online_name = "LATENCY_MATRIX_RANDOM_3"


        self.migrationCost_online=MIGR_COST


        self.states = []
        for i in range(NUM_USERS):
          if i < self.users_to_care_about:
            self.states.append(random.randint(0,NUM_STATES_PER_USER-1))
          else:
            self.states.append(0)

        self.actions = []
        for i in range(self.users_to_care_about):
          # if i < self.users_to_care_about:
            self.actions.append(random.randint(0,NUM_ACTIONS_PER_USER-1))
          # else:
          #   self.actions.append(0)

        self.prev_backup_locs = []
        for i in range(NUM_USERS):
            self.prev_backup_locs.append(self.noBackup)

        self.Action[0]=self.actions #initialising
        self.ap_capacities = AP_CAPACITIES
        self.user_loads = USER_LOADS
        print("USERS to care about = " + str(self.users_to_care_about))
    
    def find_mean(self, arr):
        if len(arr) > 0:
            return np.mean(arr)
        else:
            return 0
    
    def gen_results(self):
        self.mean_Reward_online = self.find_mean(self.Reward_online)
    
        self.mean_REreward_online = self.find_mean(self.REreward_online)
        self.mean_REmigration_online = self.find_mean(self.REmigration_online)
        self.mean_REdelay_online = self.find_mean(self.REdelay_online)
        self.mean_REstorage_online = self.find_mean(self.REstorage_online)
        self.mean_REcompDelay_online = self.find_mean(self.REcompDelay_online)
        
        self.mean_NSreward_online = self.find_mean(self.NSreward_online)
        self.mean_NSmigration_online = self.find_mean(self.NSmigration_online)
        self.mean_NSdelay_online = self.find_mean(self.NSdelay_online)
        self.mean_NSstorage_online = self.find_mean(self.NSstorage_online)
        self.mean_NScompDelay_online = self.find_mean(self.NScompDelay_online)

    """
    Updates the failed location at every time step
    """
    def update_failed_locations(self):
        if not LOCATION_BASED_FAILURE_ENABLED:
            return
        for key in list(self.failed_APs):
            value = self.failed_APs[key]
            if value <= 1:
                self.failed_APs.pop(key, None)
            else:
                self.failed_APs[key] = value - 1
    """
    Adds a failed state to 
    """
    def insert_failed_location(self, ap):
        if ap in WEAK_AP_LIST and LOCATION_BASED_FAILURE_ENABLED:
            self.failed_APs[ap] = LOCATION_BASED_FAILURE_TIME



    def run(self, trained_obj):
        for t in range(0,totalTS):
            self.update_failed_locations()

            ap_user_load = np.zeros((self.numAP))
            random_nos_aps = np.random.rand(self.numAP)

            all_users_new_state = []
            back_up_locs = []
            all_rewards = []
            new_service_locs = []
            user_anomalies = []

            for user in range(self.users_to_care_about):
                user_reward = 0
                user_state = self.states[user]
                user_action = self.actions[user]
                prevUserLoc=int(self.stateList[user_state][0])
                prevSvcLoc=int(self.stateList[user_state][1])
                newSvcLoc=int(self.actList[user_action][0])
                backupLoc=int(self.actList[user_action][1]) # initialization
                user_prev_backup_loc = self.prev_backup_locs[user]
                
                storCost=0
                if backupLoc!=self.noBackup:
                    storCost=self.storageCost_online[backupLoc]
                
                backup_migration_cost = 0
                if user_prev_backup_loc != self.noBackup and backupLoc!=self.noBackup and user_prev_backup_loc != backupLoc:
                    backup_migration_cost = self.migrationCost_online[user_prev_backup_loc,backupLoc]
                    
                migration_cost=self.migrationCost_online[prevSvcLoc,newSvcLoc]

                newUserLoc=np.random.choice(np.arange(0,self.numAP), 1, replace=False, p=self.pr_lu_trans_online[prevUserLoc,:])[0]
                #eligibility[state,action,t]=eligibility[state,action,t]+1
                
                ap_user_load[newSvcLoc] += self.user_loads[user]
                new_service_locs.append(newSvcLoc)

                # Check if the new service location is already down due to some previous failure
                serv_loc_down = False
                if LOCATION_BASED_FAILURE_ENABLED and newSvcLoc in self.failed_APs:
                    user_reward += common_FAILED_AP_SERV_LOC_REW
                    serv_loc_down = True
                    
                rare_event_occured = random_nos_aps[prevSvcLoc] < actualEPS
                
                if not serv_loc_down and rare_event_occured and LOCATION_BASED_FAILURE_ENABLED:
                    self.insert_failed_location(newSvcLoc)

                #if random number is smaller than the sampling prob, rare event has occured
                if serv_loc_down or rare_event_occured:
                    failureLoc= newSvcLoc
                    # check if there is backup
                    if backupLoc==self.noBackup:
                        # print("NO BACKUP: State: ")
                        user_reward += NEGATIVE_REWARD-migration_cost
                    else:
                        if backupLoc==failureLoc or backupLoc in self.failed_APs: # we dont want this to happen.
                            # print("BACKUP==FAILURE: State: ")#, stateList0[state], "\t Action: ", actList0[action])
                            user_reward += NEGATIVE_REWARD-migration_cost-storCost-backup_migration_cost
                        else:
                            user_reward += -self.commDelay_online[newUserLoc][backupLoc]-self.migrationCost_online[prevSvcLoc][newSvcLoc]-storCost-backup_migration_cost
                            delay=-self.commDelay_online[newUserLoc][backupLoc]
                            self.REdelay_online.append(delay)
                    
                    self.REstorage_online.append(-storCost)
                    self.REreward_online.append(user_reward)
                    migr=-migration_cost-backup_migration_cost
                    self.REmigration_online.append(migr)
                    nextState=get_state_id(newUserLoc, newSvcLoc, 1, backupLoc)
                    user_anomalies.append(True)
                else: # no rare event has occured.
                    
                    MigrationCost=migration_cost+backup_migration_cost
                    CommDelayCost=self.commDelay_online[newUserLoc][newSvcLoc]
                    
                    user_reward += (-MigrationCost-CommDelayCost)-storCost
                    self.NSreward_online.append(user_reward)
                    self.NSmigration_online.append(-MigrationCost)
                    self.NSdelay_online.append(-CommDelayCost)
                    self.NSstorage_online.append(-storCost)
                    nextState=get_state_id(newUserLoc, newSvcLoc, 0, backupLoc)
                    user_anomalies.append(False)

                all_rewards.append(user_reward)
                back_up_locs.append(backupLoc)
                all_users_new_state.append(nextState)

                self.find_optimal_backup_location(NUM_ACCESS_POINTS,self.storageCost_online, self.commDelay_online, newUserLoc)

            for u in range(NUM_USERS-self.users_to_care_about):
                if t==0:
                  print("Lesser users to care about")
                new_service_locs.append(0)
                back_up_locs.append(0)
                all_users_new_state.append(0)

        
        total_count = len(self.combined_data)
        save_folder = rf"C:\Users\Jenny\Desktop\MultiUser-FIRE-main\Figures\{self.commDelay_online_name}_{self.storageCost_name}"
        filenames = {
            "ratio_plot": "latency_storage_ratio.png",
            "decision_plot": "Backup_Distribution.png",
            "probability_plot": "Latency_BackupDecision_vs_Location.png",
        }

        self.plot_and_save_all(self.combined_data, total_count, self.backup_counts, save_folder, filenames)

        self.gen_results()


    def find_optimal_backup_location(self, NUM_ACCESS_POINTS, storage_costs_online, comm_delay_online, new_user_loc):
        """
        Function to identify the optimal backup location for a user based on the comparison of 
        storage and latency costs. It also updates the backup count and stores combined data for valid and 
        non-valid locations.

        Parameters:
            NUM_ACCESS_POINTS (int): The total number of available access points.
            storage_costs_online (list): A list of storage costs for each access point.
            comm_delay_online (list of lists): A 2D list of communication delays between the user and each access point. 
                                            Each inner list corresponds to a user's communication delays to all access points.
            new_user_loc (int): The index of the user's current location.

        Returns:
            int or None: 
                - The index of the optimal backup location (if a valid location is found where latency > storage cost).
                - `None` if no valid backup location exists (i.e., latency is not greater than storage cost for any access point).

        """
        storage_costs = np.array(storage_costs_online)
        latency_costs = np.array(comm_delay_online[new_user_loc])

        cost_differences = latency_costs - storage_costs
        valid_indices = np.where(latency_costs > storage_costs)[0]  # Only consider indices where latency > storage

        # if valid_indices.size == 0:
        #     # No valid backup location
        #     return None
        if valid_indices.size != 0:
            optimal_index = valid_indices[np.argmax(cost_differences[valid_indices])]  # Optimal index

            # Update backup counts for the optimal location
            self.backup_counts[optimal_index] += 1
        else:
            optimal_index = None

        # Store combined data for all valid locations and mark the optimal one
        self.combined_data.extend(
            [
                [latency_costs[i], storage_costs[i], 1, i, 
                "Optimal" if i == optimal_index else "Non-Optimal"]
                for i in valid_indices
            ]
        )

        # Add non-valid locations to combined_data (optional)
        self.combined_data.extend(
            [
                [latency_costs[i], storage_costs[i], 0, i, "Non-Valid"]
                for i in range(NUM_ACCESS_POINTS) if i not in valid_indices
            ]
        )

        return optimal_index

    def plot_latency_to_storage_ratio(self, combined_data, save_folder, filename):
        """
        Plots the ratio of latency cost over storage cost versus backup location as a scatter plot.
        Highlights optimal placements in gold, ratios > 1 in dark green, and ratios <= 1 in firebrick red.

        Parameters:
            combined_data (list of lists): Each sublist contains [latency_cost, storage_cost, backup_decision, location_index, "Optimal"/"Non-Optimal"].
            save_folder (str): Folder to save the plot.
            filename (str): Filename to save the plot.
        """
        # Extract data
        locations = [entry[3] for entry in combined_data]
        latency_costs = [entry[0] for entry in combined_data]
        storage_costs = [entry[1] for entry in combined_data]
        optimal_flags = [entry[4] for entry in combined_data]  # "Optimal" or "Non-Optimal"

        # Calculate ratios (latency_cost / storage_cost)
        ratios = [
            latency_cost / storage_cost if storage_cost != 0 else float('inf')
            for latency_cost, storage_cost in zip(latency_costs, storage_costs)
        ]

        # Separate points based on their properties
        optimal_locations = [locations[i] for i in range(len(optimal_flags)) if optimal_flags[i] == "Optimal"]
        optimal_ratios = [ratios[i] for i in range(len(optimal_flags)) if optimal_flags[i] == "Optimal"]

        # Non-optimal locations: these are locations that are not optimal, but latency > storage (valid backups).
        non_optimal_above_1_locations = [
            locations[i] for i in range(len(optimal_flags)) if optimal_flags[i] == "Non-Optimal" and ratios[i] > 1
        ]
        non_optimal_above_1_ratios = [
            ratios[i] for i in range(len(optimal_flags)) if optimal_flags[i] == "Non-Optimal" and ratios[i] > 1
        ]

        # Non-valid locations: locations where latency cost is not greater than storage cost
        non_valid_locations = [
            locations[i] for i in range(len(optimal_flags)) if latency_costs[i] <= storage_costs[i]
        ]
        non_valid_ratios = [
            ratios[i] for i in range(len(optimal_flags)) if latency_costs[i] <= storage_costs[i]
        ]

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(
            non_optimal_above_1_locations,
            non_optimal_above_1_ratios,
            color='darkgreen',
            label='Non-Optimal Backup (Ratio > 1)',
            s=100,
            alpha=0.7,
        )

        plt.scatter(
            optimal_locations,
            optimal_ratios,
            color='gold',
            edgecolor='black',
            label='Optimal Backup',
            s=150,
            zorder=5,
            alpha=0.8,
        )
        plt.scatter(
            non_valid_locations,
            non_valid_ratios,
            color='firebrick',
            label='No Backup',
            s=100,
            alpha=0.7,
        )

        plt.xlabel("Backup Location Index")
        plt.ylabel("Latency/Storage Ratio")
        plt.title("Ratio of Latency Cost to Storage Cost vs Backup Location")
        plt.axhline(1, color='r', linestyle='--', label='Ratio = 1 (Equal Costs)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        y_min = 0
        if ratios:
             y_max = max(ratios)
        else:
            y_max = 2
        plt.ylim(y_min - 0.5, y_max + 0.5)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, filename))
        plt.close()



    def plot_backup_decisions(self, combined_data, save_folder, filename):
        combined_data.sort(key=lambda x: x[0])
        latency_costs = [item[0] for item in combined_data]
        backup_decisions = [item[2] for item in combined_data]
        location_indices = [item[3] for item in combined_data]
        optimal_flags = [item[4] for item in combined_data]

        optimal_indices = [location_indices[i] for i in range(len(optimal_flags)) if optimal_flags[i] == "Optimal"]
        optimal_latencies = [latency_costs[i] for i in range(len(optimal_flags)) if optimal_flags[i] == "Optimal"]

        indices_with_backup = [
            location_indices[i] for i in range(len(backup_decisions)) if backup_decisions[i] == 1
        ]
        latencies_with_backup = [
            latency_costs[i] for i in range(len(backup_decisions)) if backup_decisions[i] == 1
        ]

        indices_without_backup = [
            location_indices[i] for i in range(len(backup_decisions)) if backup_decisions[i] == 0
        ]
        latencies_without_backup = [
            latency_costs[i] for i in range(len(backup_decisions)) if backup_decisions[i] == 0
        ]

        plt.figure(figsize=(10, 6))
        plt.scatter(
            indices_with_backup,
            latencies_with_backup,
            color='darkgreen',
            label='Backup Decision = 1',
            s=100,
        )
        plt.scatter(
            indices_without_backup,
            latencies_without_backup,
            color='firebrick',
            label='Backup Decision = 0',
            s=100,
        )
        plt.scatter(
            optimal_indices,
            optimal_latencies,
            color='gold',
            edgecolor='black',
            label='Optimal Placement',
            s=150,
            zorder=5,
            alpha=0.7,
        )

        plt.xlabel("Location Index")
        plt.ylabel("Latency Cost")
        plt.title("Latency Costs and Backup Decisions (Highlighting Optimal Placements)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, filename))
        plt.close()


    def plot_backup_probabilities(self, total_count, backup_counts,save_folder,filename):
        """
        Plot the probability of placing a backup at specific indices. Backup distribution graph.

        Parameters:
            total_count (int): Total number of events (e.g., users processed).
            backup_counts (list): List of counts indicating how many backups were placed at each index.
        """
        # Calculate probabilities for each index
        total_backups = sum(backup_counts)  # Total backups placed
        probabilities = [count / total_backups if total_backups > 0 else 0 for count in backup_counts]  # Conditional probabilities
        if total_count:
            overall_probability = total_backups / (total_count/NUM_ACCESS_POINTS) 
        else:
            overall_probability = 0
        

        # Plot: Probability distribution across indices
        locations = list(range(len(backup_counts)))
        plt.figure(figsize=(10, 6))
        plt.bar(locations, probabilities, color="lightgreen", edgecolor="black")
        plt.xlabel("Location Index")
        plt.ylabel("Conditional Probability of Backup Placement")
        plt.title(f"Distribution of optimal backup placement, given that a backup is placed.")
        plt.ylim(0, max(probabilities) * 1.1)  # Adjust y-axis to fit the bars comfortably
        plt.xticks(locations)  # Ensure all locations are labeled
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        plt.text(
            len(backup_counts) - 1,  # Position: near the rightmost bar
            max(probabilities) * 1,  # Position: above the tallest bar
            f"Overall Probability to place backup = {overall_probability:.2f}",
            fontsize=10,
            color="blue",
            ha="right",
            va="center",
            bbox=dict(facecolor="white", edgecolor="blue", boxstyle="round,pad=0.3"),
         )
        
        plt.tight_layout()
        # Ensure the save folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Save the plot
        plt.savefig(os.path.join(save_folder, filename))
        plt.close()


    def plot_and_save_all(self, combined_data, total_count, backup_counts, save_folder, filenames):
        """
        Calls all plotting functions and saves the images to the specified folder with given filenames.

        Parameters:
            combined_data (list): Data as [latency_cost, storage_cost, backup_decision, location_index].
            total_count (int): Total number of events (e.g., users processed).
            backup_counts (list): List of counts indicating how many backups were placed at each index.
            save_folder (str): Folder where the plots should be saved.
            filenames (dict): Dictionary containing filenames for each plot.
                Example: {"ratio_plot": "latency_storage_ratio.png", 
                        "decision_plot": "backup_decisions.png", 
                        "probability_plot": "backup_probabilities.png"}
        """
        self.plot_latency_to_storage_ratio(combined_data, save_folder, filenames["ratio_plot"])
        self.plot_backup_decisions(combined_data, save_folder, filenames["decision_plot"])
        self.plot_backup_probabilities(total_count, backup_counts, save_folder, filenames["probability_plot"])
        print(f"Plots saved successfully in: {save_folder}")
if __name__ == "__main__":
    algo = Combined_Online_Algo()
    
    trained_obj = None  # Replace this with the actual trained model if available

    algo.run(trained_obj)

    algo.gen_results()
    print("Mean Reward Online:", algo.mean_Reward_online)
    print("Mean RE Reward Online:", algo.mean_REreward_online)
    print("Mean NS Reward Online:", algo.mean_NSreward_online)