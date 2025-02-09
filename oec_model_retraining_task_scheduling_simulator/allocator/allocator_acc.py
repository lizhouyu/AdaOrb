import numpy as np
import os
from utils.logger import getMyLogger

class AccAllocator(object):
    def __init__(self, inputs, log_folder='logs'):
        self.inputs = inputs # Store inputs as dictionary
        self.random_seed = inputs['random_seed']
        os.makedirs(log_folder, exist_ok=True)
        self.logger = getMyLogger(os.path.join(log_folder, 'log_acc_allocator.log'))

        # initialize random seed for reproducibility
        np.random.seed(self.random_seed)

        self.logger.info(f"AccAllocator initialized with random seed: {self.random_seed}")

        # # Initialize model, mpc, and simulator
        # self.model = template_model(inputs['N_K'])
        # self.logger.info("Model initialized")
        # self.mpc = template_mpc(self.model, inputs['v_sequence'], inputs['b_sequence'], inputs['prediction_horizon'])
        # self.logger.info(f"MPC initialized with initial state: {self.mpc.x0}, acc: {self.mpc.x0['acc']}, down: {self.mpc.x0['down']}, vol: {self.mpc.x0['vol']}") 
        # self.mpc.set_initial_guess()
        # self.logger.info(f"Initial guess set for MPC with initial state: {self.mpc.x0}, acc: {self.mpc.x0['acc']}, down: {self.mpc.x0['down']}, vol: {self.mpc.x0['vol']})")
        # self.simulator = template_simulator(self.model, inputs['v_sequence'], inputs['b_sequence'], inputs['N_K'])
        # self.logger.info("Simulator initialized with initial state: {self.simulator.x0}, acc: {self.simulator.x0['acc']}, down: {self.simulator.x0['down']}, vol: {self.simulator.x0['vol']}")
        # v_init = np.reshape(inputs['v_sequence'][0], (inputs['N_K'], 1))
        # set_initial_conditions(self.mpc, self.simulator, inputs['N_K'], None, v_init, None)
        # self.logger.info("Initial conditions set for MPC and simulator")

    def make_downlink_decision(self, time_step, v_sequence, b_sequence, state_vol, state_down, state_acc):
        # get the current v and b sequences
        v_current = v_sequence[0]
        b_current = b_sequence[0]
        # Allocate bandwidth according to the state acc of each task
        ## get 1 - acc for each task as the downlink_urgency
        downlink_urgency = 1 - np.array(state_acc)
        ## compute the allocation ratio for each task as the ratio of the downlink_urgency to the sum of the downlink_urgency
        allocation_ratio = downlink_urgency / np.sum(downlink_urgency)
        ## allocate bandwidth according to the allocation ratio
        allocation = allocation_ratio * b_current
        # rounding the allocated bandwidth
        allocation = np.round(allocation)
        # clip the allocation to ensure it does not exceed the bandwidth
        while np.sum(allocation) > b_current:
            max_decision_idx = np.argmax(allocation)
            if allocation[max_decision_idx] > 0:
                allocation[max_decision_idx] -= 1
            else:
                raise ValueError(f"Allocation cannot be reduced further, bandwidth: {b_current}, allocation: {allocation}")
        # clip the allocation to ensure it does not exceed the accumulated volume minus the already downlinked data
        allocation = np.clip(allocation, 0, np.array(state_vol) - np.array(state_down))
        allocation = [int(x) for x in allocation]
        self.logger.info(f"Downlink decision at time step {time_step}: {allocation}, sum downlink: {np.sum(allocation)}, bandwidth: {b_current}, state_down: {state_down}, state_vol: {state_vol}")
        return allocation


        