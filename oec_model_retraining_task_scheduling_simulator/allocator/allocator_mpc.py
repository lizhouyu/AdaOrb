import numpy as np
import os
from .mpc import template_model, template_mpc, template_simulator, set_initial_conditions
from utils.logger import getMyLogger
import time

class MPCAllocator(object):
    def __init__(self, inputs, log_folder='logs'):
        self.inputs = inputs # Store inputs as dictionary

        os.makedirs(log_folder, exist_ok=True)
        self.logger = getMyLogger(os.path.join(log_folder, 'log_mpc_allocator.log'))

        self.prediction_horizon = inputs['prediction_horizon']
        # self.reload(0, [], [], self.prediction_horizon)

        # # Initialize model, mpc, and simulator
        # self.model = template_model(inputs['N_K'])
        # self.logger.info("Model initialized")
        # self.mpc = template_mpc(self.model, inputs['v_sequence'], inputs['b_sequence'], inputs['prediction_horizon'])
        # self.logger.info(f"MPC initialized with initial state: {self.mpc.x0}, acc: {self.mpc.x0['acc']}, down: {self.mpc.x0['down']}, vol: {self.mpc.x0['vol']}") 
        # self.mpc.set_initial_guess()
        # self.mpc.make_step(self.mpc.x0)
        # self.logger.info(f"Initial guess set for MPC with initial state: {self.mpc.x0}, acc: {self.mpc.x0['acc']}, down: {self.mpc.x0['down']}, vol: {self.mpc.x0['vol']})")
        # self.simulator = template_simulator(self.model, inputs['v_sequence'], inputs['b_sequence'], inputs['N_K'])
        # self.logger.info("Simulator initialized with initial state: {self.simulator.x0}, acc: {self.simulator.x0['acc']}, down: {self.simulator.x0['down']}, vol: {self.simulator.x0['vol']}")
        # v_init = np.reshape(inputs['v_sequence'][0], (inputs['N_K'], 1))
        # set_initial_conditions(self.mpc, self.simulator, inputs['N_K'], None, v_init, None)
        # self.logger.info("Initial conditions set for MPC and simulator")

    def reload(self, N_K, v_sequence, b_sequence, prediction_horizon):
        self.model = template_model(N_K)
        self.logger.info("Model reloaded")
        self.mpc = template_mpc(self.model, v_sequence, b_sequence, prediction_horizon)
        self.logger.info(f"MPC reloaded with initial state: {self.mpc.x0}, acc: {self.mpc.x0['acc']}, down: {self.mpc.x0['down']}, vol: {self.mpc.x0['vol']}") 
        self.mpc.set_initial_guess()
        # self.mpc.make_step(self.mpc.x0)
        self.logger.info(f"Initial guess set for MPC with initial state: {self.mpc.x0}, acc: {self.mpc.x0['acc']}, down: {self.mpc.x0['down']}, vol: {self.mpc.x0['vol']})")
        self.simulator = template_simulator(self.model, v_sequence, b_sequence, N_K)
        self.logger.info(f"Simulator reloaded with initial state: {self.simulator.x0}, acc: {self.simulator.x0['acc']}, down: {self.simulator.x0['down']}, vol: {self.simulator.x0['vol']}")
        v_init = np.reshape(v_sequence[0], (N_K, 1))
        set_initial_conditions(self.mpc, self.simulator, N_K, None, v_init, None)
        self.logger.info("Initial conditions set for MPC and simulator")

    def make_downlink_decision(self, time_step, v_sequence, b_sequence, state_vol, state_down, state_acc):
        time_start_making_decision = time.perf_counter()
        self.reload(len(state_down), v_sequence, b_sequence, self.inputs['prediction_horizon'])
        self.logger.info(f"Reloaded MPC with new inputs at time step {time_step}, N_K: {len(state_down)} v_sequence: {v_sequence}, b_sequence: {b_sequence}, state_vol: {state_vol}, state_down: {state_down}, state_acc: {state_acc}")
        # get x0 from simulator
        x0 = self.simulator.x0
        # Update the state of the MPC with real-time data
        x0['acc'] = state_acc
        x0['down'] = state_down
        x0['vol'] = state_vol
        self.simulator.x0 = x0
        self.logger.info(f"Updated MPC state with real-time data at time step {time_step}, acc: {x0['acc']}, down: {x0['down']}, vol: {x0['vol']}, 'this_bandwidth constraint': {self.model.tvp['b_j']}, bandwidth: {self.inputs['b_sequence'][time_step]}")
        # Get the control action from the MPC
        u = self.mpc.make_step(x0)
        self.logger.info(f"MPC control raw action: {u}")
        # round the control action to ensure it is an integer
        decision = [int(np.round(elem[0])) for elem in u]
        # let value smaller than 0 be 0
        decision = [max(0, elem) for elem in decision]
        self.logger.info(f"MPC control action: {decision}, sum downlink before bandwidth check: {np.sum(decision)}, bandwidth: {self.inputs['b_sequence'][time_step]}, state_down: {state_down}")
        # make sure the total downlink does not exceed the available bandwidth
        while np.sum(decision) > self.inputs['b_sequence'][time_step]:
            # get the maximum decision
            max_decision_idx = np.argmax(decision)
            if decision[max_decision_idx] > 0:
                decision[max_decision_idx] -= 1
            else:
                raise ValueError(f"Decision cannot be reduced further, bandwidth: {self.inputs['b_sequence'][time_step]}, decision: {decision}")
        # # get accumulated volume for each task until the current time step
        # accumulated_volume = np.sum(self.inputs['v_sequence'][:time_step], axis=0)
        # # and each decision not surpass the available volume
        # # get task_idx: {decision, decision_available_gap} dict
        # available_volume = accumulated_volume - state_down
        # decision_available_diff = np.array(decision) - available_volume
        # task_idx_decision_num_available_gap_dict = {} # {task_idx: {'decision': decision, 'available_gap': available_gap}}
        # for i in range(len(decision_available_diff)):
        #     task_idx_decision_num_available_gap_dict[i] = {'decision': decision[i], 'available_gap': decision_available_diff[i]}
        # # for every decision that excess the available volume, collect the excess, and reduce the decision to the available volume
        # total_excess = 0
        # for task_idx, task_decision_and_gap in task_idx_decision_num_available_gap_dict.items():
        #     if task_decision_and_gap['available_gap'] > 0:
        #         total_excess += task_decision_and_gap['available_gap']
        #         task_idx_decision_num_available_gap_dict[task_idx]['decision'] = available_volume[task_idx]
        #         task_idx_decision_num_available_gap_dict[task_idx]['available_gap'] = 0
        # # distribute the total excess to the decisions
        # while total_excess > 0 and len(task_idx_decision_num_available_gap_dict) > 0:
        #     # get the task index with minimum decision and available gap < 0
        #     min_task_idx = min(task_idx_decision_num_available_gap_dict, key=lambda x: task_idx_decision_num_available_gap_dict[x]['decision'] if task_idx_decision_num_available_gap_dict[x]['available_gap'] < 0 else float('inf'))
        #     # allocate one unit to the task with minimum decision
        #     task_idx_decision_num_available_gap_dict[min_task_idx]['decision'] += 1
        #     # update the total excess
        #     total_excess -= 1
        #     # update the available gap
        #     task_idx_decision_num_available_gap_dict[min_task_idx]['available_gap'] += 1
        # update the decision list
        # for i in range(len(decision)):
        #     decision[i] = int(task_idx_decision_num_available_gap_dict[i]['decision'])
        self.logger.info(f"MPC control action: {decision}, sum downlink: {np.sum(decision)}, bandwidth: {self.inputs['b_sequence'][time_step]}, state_down: {state_down}")
        # update u with the final decision
        for i in range(len(decision)):
            u[i][0] = decision[i]
        # Make a step in the simulator
        self.simulator.make_step(u)        
        time_end_making_decision = time.perf_counter()
        self.logger.info(f"Time taken to make decision at time step {time_step}: {time_end_making_decision - time_start_making_decision}")
        return decision