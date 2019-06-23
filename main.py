import copy
import numpy as np
import environment
import selfabsorbedagent
import empathyagent
import kSPMDPagent

#  ******************** Initialization Step ********************
NUM_ROBOTS = 2   # num Of robots
ALL_ACTIONS = ['1', '2', '3', '4', '5']    # Actions include: { Stay:1, North:2, South:3, East:4, West:5}
NUM_ROWS = 2
NUM_COLUMNS = 2
ZERO_EPSILON = 0.0000000001
EPSILON_GREEDY_VALUE = ZERO_EPSILON
file_output_id = "-R" + str(NUM_ROBOTS) + "-ksp-k2-" + "5-"
NUMBER_OF_RUNS = 5  # total runs
MAX_ITERATION = 5  # each run should be continued for MAX_ITERATION
TASK_GENERATION_RATE = 0.05  # with this probability every clean cell will become dirty by generate_new_tasks method
method = 'Empathy-kSPMDP'   # should be from {'Self_absorbed' , 'Empathy', 'Empathy-kSPMDP'}
k = 2  # num of tasks to be focused in kSP-MDP model
horizon = 5


# ********************  reading input data:  ********************
print("   reading transition data ...   ")
with open("input-data/2*2kSMDP-k2/transitionProb2-2-kSP-MDP-k2.txt") as f:
    whole_content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in whole_content]
transition_dict = {}
# all_states_set = set()
for line in content:
    data = line.split(",")
    key = (int(data[0]), int(data[1]), int(data[2]))
    value = round(float(data[3]), 10)
    transition_dict[key] = value
#     all_states_set.add(data[0])
# print("  all_states_set:  ", all_states_set, "len: ", len(all_states_set))

print("   reading reward data ...   ")
with open("input-data/2*2kSMDP-k2/reward-data2-2-kSP-MDP-k2.txt") as input_reward_file:  # SP2-2/reward-data2-2-SP-MDP.txt
    whole_rewards = input_reward_file.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
reward_content = [x.strip() for x in whole_rewards]
reward_dict = {}
for line in reward_content:
    data = line.split(",")
    key_rew = (int(data[0]), int(data[1]))
    value_rew = round(float(data[2]), 10)
    reward_dict[key_rew] = value_rew

print("   reading environment states dictionary data ...   ")
with open("input-data/SP2-2/states_data6.txt") as input_states_file:
    whole_states = input_states_file.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
states_content = [x.strip() for x in whole_states]
env_states_dict = {}
for line in states_content:
    data = line.split(",")
    key_sta = int(data[0])
    value_sta = [int(data[1]), int(data[2])]
    env_states_dict[key_sta] = value_sta

print(" env_states_dict:   ", env_states_dict)

if method == 'Empathy-kSPMDP':
    print("   reading kSP-MDP states dictionary data ...   ")
    with open("input-data/2*2kSMDP-k2/states_data5.txt") as input_states_file:
        kSP_states = input_states_file.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    ksp_states_content = [x.strip() for x in kSP_states]
    ksp_states_dict = {}
    for line in ksp_states_content:
        data = line.split(",")
        ksp_key_sta = int(data[0])
        ksp_value_sta = [int(data[1]), int(data[2])]
        ksp_states_dict[ksp_key_sta] = ksp_value_sta

    print(" ksp_states_dict:   ", ksp_states_dict)

# ********************  Starting Runs:  ********************
print("\n  ***************  Starting Runs:  ***********************  \n")

for run_number in range(NUMBER_OF_RUNS):
    print(" \n  ********** run_number: ", run_number, "  **********  \n")
    # instantiating environment
    env = environment.Environment(NUM_ROBOTS, ALL_ACTIONS, NUM_ROWS, NUM_COLUMNS, transition_dict, env_states_dict)
    # Robots' locations initialization; keeping x and y coordinate
    env.make_all_rob_loc_dictionary()
    # env.make_all_states()
    # print(" **** num states:  ", len(env.all_states))
    # print("env.all_states", env.all_states)
    initial_task_status = env.is_dirt

    # converting initial_task_status from 2D list to a string and then a decimal number
    init_task_status_list = []
    for i in range(len(initial_task_status)):
        for j in range(len(initial_task_status[0])):
            init_task_status_list.append(initial_task_status[i][j])

    str_init_task_status = ''.join(init_task_status_list)
    decimal_init_task_status = int(str_init_task_status, 2)
    print(" decimal_init_task_status:   ", decimal_init_task_status)

    # Determining initial position for robots
    all_agents_initial_pos = env.init_rob_loc()
    print("all_agents_initial_pos  ", all_agents_initial_pos)

    # instantiating all agents
    all_agents = []
    if method == 'Self_absorbed':
        for rob_index in range(NUM_ROBOTS):
            all_agents.append(selfabsorbedagent.SelfAbsorbed(env.all_states, ALL_ACTIONS,
                                                             all_agents_initial_pos[rob_index], env.rob_loc_dict,
                                                             env.num_columns, env.num_rows, transition_dict,
                                                             decimal_init_task_status, ZERO_EPSILON, rob_index,
                                                             reward_dict))
    elif method == 'Empathy':
        for rob_index in range(NUM_ROBOTS):
            all_agents.append(empathyagent.Empathy(env.all_states, ALL_ACTIONS, all_agents_initial_pos[rob_index],
                                                   env.rob_loc_dict, env.num_columns, env.num_rows, transition_dict,
                                                   decimal_init_task_status, ZERO_EPSILON, rob_index, reward_dict))

    else:   # method == 'Empathy-kSPMDP':
        for rob_index in range(NUM_ROBOTS):
            all_agents.append(kSPMDPagent.KSPMDP(ksp_states_dict, ALL_ACTIONS, all_agents_initial_pos[rob_index],
                                                 env.rob_loc_dict, env.num_columns, env.num_rows, transition_dict,
                                                 decimal_init_task_status, ZERO_EPSILON, rob_index, reward_dict, k))
        for ag in all_agents:
            ag.make_mapping_from_global_tasks(env.all_states)
            ag.set_initial_state(decimal_init_task_status, env.all_states)

    current_tsk_status_key = decimal_init_task_status

    file_task = open("output-data/tasksStatus" + file_output_id + ".txt", "a+")
    file_reward = open("output-data/reward" + ".txt", "a+")   # + file_output_id

    for iteration in range(MAX_ITERATION):
        print("\n iteration:  ", iteration)
        # print(" robot locations: ", env.all_robs_loc)
        current_tsk_status_key = env.generate_new_tasks(current_tsk_status_key, file_output_id, iteration,
                                                        TASK_GENERATION_RATE)

        # print("current_tsk_status_key: ", current_tsk_status_key)

        selected_actions = []
        for rob in all_agents:
            # print("  ** rob_id:  ", rob.rob_id)
            rob.modify_rob_state_by_tsk_change(current_tsk_status_key, env.all_states)
            selected_actions.append(rob.choose_action(EPSILON_GREEDY_VALUE, file_output_id, env.all_robs_loc,
                                                      env.all_robs_ids))

        # print(" selected_actions:  ", selected_actions)

        for agent_iterator in range(len(all_agents)):
            # print("  rob_id:  ", all_agents[agent_iterator].rob_id)
            st = env.execute_action(selected_actions[agent_iterator], all_agents[agent_iterator].location,
                                    current_tsk_status_key)
            # print("   st:  ", st)
            current_tsk_status_key = st[1]
            all_agents[agent_iterator].set_my_state(st, env.all_states)
            env.set_robot_location(st, all_agents[agent_iterator].rob_id)

        for ite in range(len(all_agents)):
            all_agents[ite].set_task_status(current_tsk_status_key, env.all_states)

        file_task.write(str(selected_actions) + "\t" +
                        np.binary_repr(current_tsk_status_key, width=NUM_ROWS * NUM_COLUMNS) + "\n")

        # print("      current_tsk_status", np.binary_repr(current_tsk_status_key, width=NUM_ROWS * NUM_COLUMNS))

        if method == 'Self_absorbed':
            for count in range(len(all_agents)):
                all_agents[count].update_q(file_output_id, env.all_states)
                all_agents[count].update_values()
        elif method == 'Empathy':
            for counter in range(len(all_agents)):
                all_rob_loc_list = copy.deepcopy(env.all_robs_loc)
                # iteration_file_output_id = file_output_id + str(iteration)
                ag_p_mass_loc = all_agents[counter].compute_presence_mass(env.rob_loc_dict, env.all_states,
                                                                          all_rob_loc_list, current_tsk_status_key,
                                                                          horizon, env.all_robs_ids, file_output_id)
                ag_p_mass_aggregated = all_agents[counter].compute_aggregated_p_mass(ag_p_mass_loc)
                all_agents[counter].update_q_empathy(env.all_states, ag_p_mass_aggregated, horizon)
                all_agents[counter].update_values()

        elif method == 'Empathy-kSPMDP':
            for counter in range(len(all_agents)):
                all_rob_loc_list = copy.deepcopy(env.all_robs_loc)
                # iteration_file_output_id = file_output_id + str(iteration)
                ag_p_mass_loc = all_agents[counter].compute_presence_mass(env.rob_loc_dict, env.all_states,
                                                                          all_rob_loc_list, current_tsk_status_key,
                                                                          horizon, env.all_robs_ids, file_output_id)
                # print(" ag_p_mass_loc:  ", ag_p_mass_loc)
                ag_p_mass_aggregated = all_agents[counter].compute_aggregated_p_mass(ag_p_mass_loc)
                # print(" ag_p_mass_aggregated:  ", ag_p_mass_aggregated)
                all_agents[counter].update_q_empathy(ag_p_mass_aggregated, horizon, file_output_id)
                all_agents[counter].update_values()

        rew_sys = env.calculate_reward(st[1])
        # print(" rew_sys: ", rew_sys)
        file_reward.write(str(rew_sys) + '\t')

    file_reward.write('\n')

file_task.close()
file_reward.close()
