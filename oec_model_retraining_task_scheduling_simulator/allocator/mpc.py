import numpy as np
import do_mpc
from casadi import *
import random
import json

# Define the model
def template_model(N_K):
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state variables
    acc = model.set_variable(var_type='_x', var_name='acc', shape=(N_K,1))
    vol = model.set_variable(var_type='_x', var_name='vol', shape=(N_K,1))
    down = model.set_variable(var_type='_x', var_name='down', shape=(N_K,1))

    # Define control input
    d = model.set_variable(var_type='_u', var_name='d', shape=(N_K,1))

    # Parameters for collected data volume (v_j) and data limit (b_j)
    # v_j = model.set_variable(var_type='_p', var_name='v_j', shape=(1,1))  # Collected data volume
    # b_j = model.set_variable(var_type='_p', var_name='b_j', shape=(1,1))  # Data limit
    v_j = model.set_variable(var_type='_tvp', var_name='v_j', shape=(N_K,1))  # Collected data volume
    b_j = model.set_variable(var_type='_tvp', var_name='b_j', shape=(1,1))  # Data limit

    # Accuracy transition function
    # acc_next = (acc * exp(-0.1*d) + (1 - exp(-0.1*d))) / (1 + exp(-0.1*d))
    # acc_next = 0.05488259 + 0.00371186 * (down + d) - (1.81047107e-05) * (down + d) ** 2 + (4.31028326e-08) * (down + d) ** 3 - (4.92517601e-11) * (down + d) ** 4 + (2.15473582e-14) * (down + d) ** 5
    # acc_next = 0.468 / (1 + np.exp(-0.660 * (down + d - 10.233))) # original unet
    # acc_next = 0.351 / (1 + np.exp(-0.419 * (down + d - 15.384))) # pruned unet
    acc_next = 0.05488259 + 0.00371186 * (down + d) - (1.81047107e-05) * (down + d) ** 2 + (4.31028326e-08) * (down + d) ** 3 - (4.92517601e-11) * (down + d) ** 4 + (2.15473582e-14) * (down + d) ** 5

    # Collected data volume transition
    vol_next = vol + v_j

    # Downlinked data volume transition
    down_next = down + d

    # Set the state transitions
    model.set_rhs('acc', acc_next)
    model.set_rhs('vol', vol_next)
    model.set_rhs('down', down_next)

    # Set up the model
    model.setup()

    return model

# Define the MPC controller
def template_mpc(model, v_sequence, b_sequence, prediction_horizon=10):
    mpc = do_mpc.controller.MPC(model)

    # Set up the parameters for MPC
    setup_mpc = {
        'n_horizon': prediction_horizon,  # Prediction horizon
        't_step': 1,  # Time step
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # Define the cost function
    mterm = -sum1(model.tvp['v_j'] * model.x['acc'])
    lterm = -sum1(model.tvp['v_j'] * model.x['acc'])

    # Set the objective
    mpc.set_objective(mterm=mterm, lterm=lterm)

    # Set constraints
    # mpc.bounds['lower', '_u', 'd'] = 0.0  # Downlinking data volume lower bound
    # mpc.bounds['upper', '_u', 'd'] = model.x['vol']  - model.x['down'] # Downlinking data volume upper bound
    # mpc.bounds['upper', '_u', 'd'] = model.tvp['b_j'] # Downlinking data volume upper bound

    # set constraints with equality
    mpc.set_nl_cons('totol_downlink_not_exceed_avaliable', model.u['d'] + model.x['down'] - model.x['vol'], ub=0)
    mpc.set_nl_cons('current_downlink_not_exceed_bandwidth', sum1(model.u['d']) - model.tvp['b_j'], ub=0)
    mpc.set_nl_cons('downlink_is_positive', -model.u['d'], ub=0)
    
    # set time varying parameters
    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_now):
        print("mpc tvp_fun")
        print("t_now type: ", type(t_now))
        if type(t_now) == np.ndarray:
            t_now = t_now[0]
            t_now = int(t_now)
        print("t_now: ", t_now)
        for k in range(mpc.settings.n_horizon):
            tvp_template['_tvp',k,'v_j'] = v_sequence[t_now + k]
            tvp_template['_tvp',k,'b_j'] = b_sequence[t_now + k]
            print("tnow: ", t_now, "k: ", k, "v_j: ", tvp_template['_tvp',k,'v_j'], "b_j: ", tvp_template['_tvp',k,'b_j'])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    # Setup MPC
    mpc.setup()

    return mpc

# Set up the simulator
def template_simulator(model, v_sequence, b_sequence, N_K):
    simulator = do_mpc.simulator.Simulator(model)

    # Set up the simulator
    simulator.set_param(t_step=1)

    # get the tvp template
    tvp_template = simulator.get_tvp_template()

    # define the tvp function
    def tvp_fun(t_now):
        print("simulator tvp_fun")
        print("t_now: ", t_now)
        tvp_template['v_j'] = np.array(v_sequence[int(t_now)]).reshape(N_K, 1)
        # tvp_template['b_j'] = np.array(b_sequence[int(t_now)])
        tvp_template['b_j'] = b_sequence[int(t_now)]
        print("tnow: ", t_now, "v_j: ", tvp_template['v_j'], "b_j: ", tvp_template['b_j'])
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator

# Set up initial conditions
def set_initial_conditions(mpc, simulator, N_K, acc0=None, vol0=None, down0=None):
    if acc0 is None:
        acc0 = np.full((N_K, 1), 0.1)  # Initialize accuracy
    if vol0 is None:
        vol0 = np.zeros((N_K, 1))  # Initialize collected volume
    if down0 is None:
        down0 = np.zeros((N_K, 1))  # Initialize downlinked volume
    x0 = np.vstack((acc0, vol0, down0))  # Combine into initial state

    simulator.x0 = x0
    mpc.x0 = x0
    mpc.set_initial_guess()

# Main function to run the simulation
def run_mpc(v_sequence, b_sequence, N_K):
    model = template_model(N_K)
    print("Model created, now creating MPC")
    mpc = template_mpc(model, v_sequence, b_sequence)
    print("MPC created, now creating simulator")
    simulator = template_simulator(model, v_sequence, b_sequence, N_K)
    print("Simulator created, now setting initial conditions")
    set_initial_conditions(mpc, simulator, N_K)
    print("Initial conditions set, now running MPC")

    # Run MPC for the length of the sequence
    N_steps = len(v_sequence) - 10
    for k in range(N_steps):
        print("Step: ", k, "of ", N_steps)
        print("Get simulator state")
        x0 = simulator.x0
        print("acc: ", simulator.x0['acc'], "vol: ", simulator.x0['vol'], "down: ", simulator.x0['down'])
        print('initial parameters:', model.tvp['v_j'], model.tvp['b_j'])
        print("Get MPC control input by making a step")
        u0 = mpc.make_step(x0)
        print("u0: ", u0)
        print("Make simulator step")
        simulator.make_step(u0)

    # Retrieve the results
    results = simulator.data
    return results

if __name__ == '__main__':
    # N_T = 20  # Number of time steps
    # N_K = 3  # Number of tasks
    # np.random.seed(0)  # For reproducibility
    # # Define known sequences for v_j and b_j
    # v_sequence = np.random.randint(10, 30, size=(N_T, N_K))  # Random collected data volume
    # # randomly set some values to 0
    # for i in range(N_T):
    #     for j in range(N_K):
    #         if random.random() < 0.5: # 50% chance to set to 0
    #             v_sequence[i][j] = 0
    # print("v_sequence: ", v_sequence)
    # # b_sequence = np.random.randint(5, 15, size=(N_T, N_K))  # Random data limit
    # b_sequence = np.random.randint(20, 40, N_T) # Random data limit
    # print("b_sequence: ", b_sequence)

    # read inputs from file
    with open('inputs.json') as f:
        inputs = json.load(f)
    v_sequence = np.array(inputs['v_sequence'])
    b_sequence = np.array(inputs['b_sequence'])
    N_T = inputs['N_T']
    N_K = inputs['N_K']

    results = run_mpc(v_sequence, b_sequence, N_K)

    # Plot the results
    import matplotlib.pyplot as plt
    acc_history = np.array(results['_x'])[:, :N_K]
    vol_history = np.array(results['_x'])[:, N_K:2*N_K]
    down_history = np.array(results['_x'])[:, 2*N_K:]
    down_decision_history = np.array(results['_u'])[:, :N_K]
    captured_data_volume_history = np.array(results['_tvp'])[:, :N_K]
    available_bandwidth_history = np.array(results['_tvp'])[:, N_K]
    print("available_bandwidth_history: ", available_bandwidth_history)
    utility_history = acc_history * captured_data_volume_history
    # sum utility over tasks
    utility_all_history = np.sum(utility_history, axis=1)
    print("utility_all_history: ", utility_all_history)
    # sum decision over tasks
    decision_all_history = np.sum(down_decision_history, axis=1)

    fig, axs = plt.subplots(7, 1, figsize=(10, 20))
    
    for i in range(N_K):
        axs[0].plot(acc_history[:, i], label=f'Accuracy Task {i+1}')
    axs[0].set_title('Accuracy over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    for i in range(N_K):
        axs[1].plot(vol_history[:, i], label=f'Collected Volume {i+1}')
    axs[1].set_title('Collected Volume over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Volume')
    axs[1].legend()

    for i in range(N_K):
        axs[2].plot(down_history[:, i], label=f'Downlinked Volume {i+1}')
    axs[2].set_title('Downlinked Volume over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Volume')
    axs[2].legend()

    for i in range(N_K):
        axs[3].plot(down_decision_history[:, i], label=f'Downlink Decision {i+1}')
    axs[3].set_title('Downlink Decision over Time')
    axs[3].set_xlabel('Time Step')
    axs[3].set_ylabel('Decision')
    axs[3].legend()

    for i in range(N_K):
        axs[4].plot(captured_data_volume_history[:, i], label=f'Captured Data Volume {i+1}')
        axs[4].plot(v_sequence[:, i], label=f'Predefined Volume {i+1}', linestyle='--')
    axs[4].plot(v_sequence, label='Predefined volume', color='orange', linestyle='--')
    axs[4].set_title('Captured Data Volume over Time')
    axs[4].set_xlabel('Time Step')
    axs[4].set_ylabel('Volume')
    axs[4].legend()

    axs[5].plot(available_bandwidth_history, label='Available Bandwidth', color='blue')
    axs[5].plot(b_sequence, label='Predefined bandwidth', color='orange', linestyle='--')
    axs[5].plot(decision_all_history, label='Total Downlink Decision', color='green', linestyle='--')
    axs[5].set_title('Available Bandwidth over Time')
    axs[5].set_xlabel('Time Step')
    axs[5].set_ylabel('Bandwidth')
    axs[5].legend()

    for i in range(N_K):
        axs[6].plot(utility_history[:, i], label=f'Utility Task {i+1}')
    axs[6].plot(utility_all_history, label='Total Utility', color='black', linestyle='--')
    axs[6].set_title('Utility over Time')
    axs[6].set_xlabel('Time Step')
    axs[6].set_ylabel('Utility')
    axs[6].legend()

    plt.tight_layout()
    plt.savefig('mpc_full_results.png')
