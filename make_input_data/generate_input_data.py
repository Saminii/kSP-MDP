import numpy as np
import copy
import helper


class GenerateInputData:
    """
    Generating input data for simulation tests of the paper:
        Effective Approximations for Multi-Robot Coordination in Spatially Distributed Tasks by Claes, Daniel, et al.
        (https://pdfs.semanticscholar.org/742e/08efade6ed6ddd9092d7eefca60f13fd02f2.pdf)
    Until now simulation is done in n * m environments, no effort for corridor or office environments yet.
    """
    def __init__(self, num_columns, num_rows, num_robots):
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.num_robots = num_robots
        self.all_states = {}
        self.rob_loc_dict = {}
        self.all_robs_loc = []

    def make_all_rob_locations(self):
        rob_loc = []
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                rob_loc.append([i, j])

        rob_loc_dict = {}
        for i in range(len(rob_loc)):
            rob_loc_dict[i] = rob_loc[i]

        self.rob_loc_dict = copy.deepcopy(rob_loc_dict)

    def make_all_states(self):
        #  First making robots' locations
        self.make_all_rob_locations()

        # Second making tasks' locations
        task_status = []
        for i in range(2 ** (self.num_rows * self.num_columns)):
            task_status.append(i)

        # we use a dictionary with elements that are list with length 2;
        # the first robot location and the second task status
        states = {}
        k = 0
        for i in self.rob_loc_dict:
            for j in range(len(task_status)):
                states[k] = [i, task_status[j]]
                k += 1

        states_data = open("states-environment.txt", "+a")
        for sta in states:
            print("sta:  ", sta)
            states_data.write(str(sta) + ',' + str(states[sta][0]) + ',' + str(states[sta][1]) + '\n')
        states_data.close()

        return states
        # self.all_states = copy.deepcopy(states)

    def make_all_states_k_s_mdp(self, num_focused_active_tasks):

        # making tasks' locations
        task_status = []
        for i in range(2 ** (self.num_rows * self.num_columns)):
            task_status.append(i)

        # we use a dictionary with elements that are list with length 2;
        # the first robot location and the second task status
        states = {}
        k = 0
        for i in self.rob_loc_dict:
            for j in range(len(task_status)):
                # print("j: ", j, ",  task_status[j]:  ", task_status[j])
                num_active_tasks = self.compute_number_of_active_tasks(task_status[j])
                # print(" num_active_tasks:  ", num_active_tasks)
                if num_active_tasks <= num_focused_active_tasks:
                    states[k] = [i, task_status[j]]
                    k += 1
        # f_reward_data.write(str(s) + "," + str(act) + "," + str(reward) + "\n")

        states_data = open("states-kSP.txt", "+a")
        for sta in states:
            print("sta:  ", sta)
            states_data.write(str(sta) + ',' + str(states[sta][0]) + ',' + str(states[sta][1]) + '\n')
        states_data.close()

        return states
        # self.all_states = copy.deepcopy(states)

    def compute_number_of_active_tasks(self, tasks):
        tasks_string = np.binary_repr(tasks, width=self.num_columns * self.num_rows)
        # print(" tasks_string:  ", tasks_string)
        num_active_tasks = 0
        for e in tasks_string:
            if e == '1':
                num_active_tasks += 1

        return num_active_tasks

    def compute_trans_prob_uniform_dist_others(self, s, action, s_prime, num_actions, uniform_effect_others_flag):
        """  Computing probability of P(s_i' | s_i, a_i) where s_i = < lambda_i, tau >
            in this method, we make a uniform effect for other robots doing tasks.
            This means we believe tasks are done in each location with probability
                (self.num_robots - 1) / (num_actions * len(self.all_states)).
            In method  compute_trans_prob(self, s, action, s_prime, num_actions), we ignore others completely.
        """

        # First: the robot's task status in its location
        tasks_now = self.all_states[s][1]
        is_dirt_now = np.binary_repr(tasks_now, width=self.num_columns*self.num_rows)

        dirt_list = []
        for i in is_dirt_now:
            dirt_list.append(i)

        current_loc_code = self.all_states[s][0]
        current_loc = self.rob_loc_dict[current_loc_code]

        tasks_later = self.all_states[s_prime][1]
        is_dirt_later = np.binary_repr(tasks_later, width=self.num_rows*self.num_columns)

        dirt_later_list = []
        for i in is_dirt_later:
            dirt_later_list.append(i)

        later_loc_code = self.all_states[s_prime][0]
        later_loc = self.rob_loc_dict[later_loc_code]

        rob_task_index = self.num_rows * self.num_columns - 1 - current_loc_code
        dirt_list_other_loc = dirt_list.copy()
        deleted1 = dirt_list_other_loc.pop(rob_task_index)
        dirt_list_later_other_loc = dirt_later_list.copy()
        deleted2 = dirt_list_later_other_loc.pop(rob_task_index)

        #  Robot's task status in its location:
        p_rob_task = 1.0
        if uniform_effect_others_flag:
            if action == '1':    # stay
                if dirt_later_list[rob_task_index] == '1':  # dirt_later_matrix[current_loc[0]][current_loc[1]] == '1':
                    # p_rob_task = 0.0
                    return 0.0
                elif dirt_later_list[rob_task_index] == '0':
                    p_rob_task = 1.0
            else:   # action = 2 or 3 or 4 or 5
                if dirt_list[rob_task_index] == "0" and dirt_later_list[rob_task_index] == '0':
                    p_rob_task *= 0.95
                elif dirt_list[rob_task_index] == "0" and dirt_later_list[rob_task_index] == '1':
                    p_rob_task *= 0.05
                elif dirt_list[rob_task_index] == "1" and dirt_later_list[rob_task_index] == '0':
                    p_rob_task = (self.num_robots - 1) / (num_actions * len(self.all_states))
                else:   # dirt_list[rob_task_index] == "1" and dirt_later_list[rob_task_index] == '1':
                    p_rob_task = 1 - ((self.num_robots - 1) / (num_actions * len(self.all_states)))

            #  Tasks' status in other locations:
            for iterator in range(len(dirt_list_other_loc)):
                st = dirt_list_other_loc[iterator]
                st_next = dirt_list_later_other_loc[iterator]
                if st == "0" and st_next == '0':
                    p_rob_task *= 0.95
                elif st == "0" and st_next == '1':
                    p_rob_task *= 0.05
                elif st == "1" and st_next == '0':
                    p_rob_task *= (self.num_robots - 1) / (num_actions * len(self.all_states))
                else:
                    p_rob_task *= 1.0 - ((self.num_robots - 1) / (num_actions * len(self.all_states)))
        else:   # uniform_effect_others_flag == False, i.e ignoring others completely
            if action == '1':  # stay
                if dirt_later_list[rob_task_index] == '1':  # dirt_later_matrix[current_loc[0]][current_loc[1]] == '1':
                    # p_rob_task = 0.0
                    return 0.0
                elif dirt_later_list[rob_task_index] == '0':
                    p_rob_task = 1.0
            else:  # action = 2 or 3 or 4 or 5
                if dirt_list[rob_task_index] == "0" and dirt_later_list[rob_task_index] == '0':
                    p_rob_task *= 0.95
                elif dirt_list[rob_task_index] == "0" and dirt_later_list[rob_task_index] == '1':
                    p_rob_task *= 0.05
                elif dirt_list[rob_task_index] == "1" and dirt_later_list[rob_task_index] == '0':
                    p_rob_task = 0.0
                    return 0.0
                else:  # dirt_list[rob_task_index] == "1" and dirt_later_list[rob_task_index] == '1':
                    p_rob_task = 1.0

            #  Tasks' status in other locations:
            for iterator in range(len(dirt_list_other_loc)):
                st = dirt_list_other_loc[iterator]
                st_next = dirt_list_later_other_loc[iterator]
                if st == "0" and st_next == '0':
                    p_rob_task *= 0.95
                elif st == "0" and st_next == '1':
                    p_rob_task *= 0.05
                elif st == "1" and st_next == '0':
                    p_rob_task *= 0.0
                    return 0.0
                else:
                    p_rob_task *= 1.0

        # probability of robot's movement
        # Actions include: { Stay:1, North:2, South:3, East:4, West:5}
        p_rob_move = 1.0
        if action == '1':
            if later_loc == current_loc:
                p_rob_move = 1.0
            else:
                # p_rob_move = 0.0
                return 0.0
        elif action == "2":   # North:2
            if later_loc == current_loc:
                if current_loc[0] == self.num_rows-1:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[1] == current_loc[1] and later_loc[0] == current_loc[0] + 1:
                p_rob_move = 0.9
            else:
                # p_rob_move = 0.0
                return 0.0
        elif action == "3":  # South:3
            if later_loc == current_loc:
                if current_loc[0] == 0:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[1] == current_loc[1] and later_loc[0] == current_loc[0] - 1:
                p_rob_move = 0.9
            else:
                # p_rob_move = 0.0
                return 0.0
        elif action == "4":  # East:4:
            if later_loc == current_loc:
                if current_loc[1] == self.num_columns - 1:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[0] == current_loc[0] and later_loc[1] == current_loc[1] + 1:
                p_rob_move = 0.9
            else:
                # p_rob_move = 0.0
                return 0.0
        elif action == "5":  # West:5:
            if later_loc == current_loc:
                if current_loc[1] == 0:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[0] == current_loc[0] and later_loc[1] == current_loc[1] - 1:
                p_rob_move = 0.9
            else:
                # p_rob_move = 0.0
                return 0.0

        # p_sa = p_rob_task * p_rob_move
        return p_rob_task * p_rob_move

    def compute_trans_prob_phase_mdp(self, s, action, s_prime):
        """
            Computing probability of P(s_i' | s_i, a_i) where s_i = < lambda_i, phase_tau >
        :param s:
        :param action:
        :param s_prime:
        :param num_actions:
        :param uniform_effect_others_flag:
        :return:
        """

        # First: the robot's task status in its location
        tasks_now = self.all_states[s][1]
        is_dirt_now = np.binary_repr(tasks_now, width=self.num_columns*self.num_rows)

        dirt_list = []
        for i in is_dirt_now:
            dirt_list.append(i)

        current_loc_code = self.all_states[s][0]
        current_loc = self.rob_loc_dict[current_loc_code]

        tasks_later = self.all_states[s_prime][1]
        is_dirt_later = np.binary_repr(tasks_later, width=self.num_rows*self.num_columns)

        dirt_later_list = []
        for i in is_dirt_later:
            dirt_later_list.append(i)

        later_loc_code = self.all_states[s_prime][0]
        later_loc = self.rob_loc_dict[later_loc_code]

        rob_task_index = self.num_rows * self.num_columns - 1 - current_loc_code
        dirt_list_other_loc = dirt_list.copy()
        deleted1 = dirt_list_other_loc.pop(rob_task_index)
        dirt_list_later_other_loc = dirt_later_list.copy()
        deleted2 = dirt_list_later_other_loc.pop(rob_task_index)

        #  Robot's task status in its location:
        p_rob_task = 0.0

        if action == '1':  # stay
            if dirt_later_list[rob_task_index] == '1':  # dirt_later_matrix[current_loc[0]][current_loc[1]] == '1':
                # p_rob_task = 0.0
                return 0.0
            elif dirt_later_list[rob_task_index] == '0':
                p_rob_task = 1.0
        else:  # action = 2 or 3 or 4 or 5
            if dirt_list[rob_task_index] == "0" and dirt_later_list[rob_task_index] == '0':
                p_rob_task = 1.0  # 0.95  #
            elif dirt_list[rob_task_index] == "0" and dirt_later_list[rob_task_index] == '1':
                p_rob_task = 0.0  # 0.05  #
            elif dirt_list[rob_task_index] == "1" and dirt_later_list[rob_task_index] == '0':
                p_rob_task = 0.0
                return 0.0
            else:  # dirt_list[rob_task_index] == "1" and dirt_later_list[rob_task_index] == '1':
                p_rob_task = 1.0

        #  Tasks' status in other locations:
        for iterator in range(len(dirt_list_other_loc)):
            st = dirt_list_other_loc[iterator]
            st_next = dirt_list_later_other_loc[iterator]
            if st == "0" and st_next == '0':
                p_rob_task *= 1.0  # 0.95
            elif st == "0" and st_next == '1':
                p_rob_task *= 0.0  # 0.05
            elif st == "1" and st_next == '0':
                p_rob_task *= 0.0
                return 0.0
            else:
                p_rob_task *= 1.0

        # probability of robot's movement
        # Actions include: { Stay:1, North:2, South:3, East:4, West:5}
        p_rob_move = 1.0
        if action == '1':
            if later_loc == current_loc:
                p_rob_move = 1.0
            else:
                # p_rob_move = 0.0
                return 0.0
        elif action == "2":   # North:2
            if later_loc == current_loc:
                if current_loc[0] == self.num_rows-1:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[1] == current_loc[1] and later_loc[0] == current_loc[0] + 1:
                p_rob_move = 0.9
            else:
                # p_rob_move = 0.0
                return 0.0
        elif action == "3":  # South:3
            if later_loc == current_loc:
                if current_loc[0] == 0:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[1] == current_loc[1] and later_loc[0] == current_loc[0] - 1:
                p_rob_move = 0.9
            else:
                # p_rob_move = 0.0
                return 0.0
        elif action == "4":  # East:4:
            if later_loc == current_loc:
                if current_loc[1] == self.num_columns - 1:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[0] == current_loc[0] and later_loc[1] == current_loc[1] + 1:
                p_rob_move = 0.9
            else:
                # p_rob_move = 0.0
                return 0.0
        elif action == "5":  # West:5:
            if later_loc == current_loc:
                if current_loc[1] == 0:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[0] == current_loc[0] and later_loc[1] == current_loc[1] - 1:
                p_rob_move = 0.9
            else:
                # p_rob_move = 0.0
                return 0.0

        # p_sa = p_rob_task * p_rob_move
        return p_rob_task * p_rob_move

    def del_little_prob(self, probability_list, zero_epsilon):
        temp_list = []

        for probability in probability_list:
            if probability < zero_epsilon:
                temp_list.append(0.0)
            else:
                temp_list.append(probability)

        sum_elements = sum(temp_list)
        normalized_new_list = [x / sum_elements for x in temp_list]

        return normalized_new_list

    def compute_future_rew(self, state, ac):
        dirt_status_code = self.all_states[state][1]
        rob_loc = self.all_states[state][0]
        is_dirt = np.binary_repr(dirt_status_code, width=self.num_columns * self.num_rows)
        is_dirt_list = []
        for elemem in is_dirt:
            is_dirt_list.append(elemem)

        rob_tsk_index = self.num_rows * self.num_columns - 1 - rob_loc
        if ac == '1':
            is_dirt_list[rob_tsk_index] = '0'

        rew = 0.0
        for i in range(len(is_dirt_list)):
            if is_dirt_list[i] == '0':
                rew += 1.0
        return rew

    def set_states_dictionary(self, states_dict):
        self.all_states = states_dict


# *****************************************************************************************************
# *****************************     Running Code           *******************************************
# *****************************************************************************************************

#    Initialization Step
NUM_ROBOTS = 1  # num Of robots
ALL_ACTIONS = ['1', '2', '3', '4', '5']  # Actions include: { Stay:1, North:2, South:3, East:4, West:5}
NUM_ROWS = 2
NUM_COLUMNS = 2
ZERO_EPSILON = 0.00000000001
uniform_effect_for_other_robots = False

METHOD = 'kSP-MDP'  # chosen from the set {'S-MDP', 'SP-MDP', 'kSP-MDP'}
# S-MDP: just using Subjective view related to others
# SP-MDP: Subjective view + focusing active tasks
# kSP-MDP: Exactly like SP-MDP but focusing on k-nearest active tasks
# If we need a transition model for environment, it is enough to set METHOD = 'S-MDP'.
# We should avoid phase view to generate transition data.

NUMBER_OF_ACTIVE_TASKS = 2  # k value in k-SMDP model

#  Making all states:
gid = GenerateInputData(NUM_ROWS, NUM_COLUMNS, NUM_ROBOTS)
env_states_dict = gid.make_all_states()  # making environment states and also S-MDP and SP-MDP states

if METHOD == 'kSP-MDP':  # making states for k-SP-MDP model
    states_dict = gid.make_all_states_k_s_mdp(NUMBER_OF_ACTIVE_TASKS)
else:  # making states for 'S-MDP', 'SP-MDP'
    states_dict = copy.deepcopy(env_states_dict)

gid.set_states_dictionary(states_dict)

sorted_all_states = sorted(gid.all_states.items(), key=lambda kv: kv[1])
print("sorted_all_states:  ", sorted_all_states)

file_name_suffix = str(NUM_ROWS) + '-' + str(NUM_COLUMNS) + '-' + METHOD
if METHOD == 'kSP-MDP':
    file_name_suffix = file_name_suffix + '-k' + str(NUMBER_OF_ACTIVE_TASKS)

#  Now making transition probabilities
transition_file_name = 'transitionProb' + file_name_suffix + '.txt'
f = open(transition_file_name, "+a")
print(" generating transition data:   ")
for element in sorted_all_states:
    s_first = element[0]
    print("s: ", s_first)
    for act in ALL_ACTIONS:
        prob_list = []
        for s_pri in gid.all_states:
            if METHOD == 'S-MDP':
                prob = gid.compute_trans_prob_uniform_dist_others(s_first, act, s_pri, len(ALL_ACTIONS),
                                                                  uniform_effect_for_other_robots)
            else:  # if METHOD == 'SP-MDP' or METHOD == 'kSP-MDP':
                prob = gid.compute_trans_prob_phase_mdp(s_first, act, s_pri)
            prob_list.append(prob)
        sum_prob = sum(prob_list)
        new_normalized_prob = []
        if not sum_prob == 0.0:
            normalized_prob = helper.normalize_probability(prob_list)
            prob_after_del_little_prob = gid.del_little_prob(normalized_prob, ZERO_EPSILON)
            new_normalized_prob = copy.deepcopy(helper.normalize_probability(prob_after_del_little_prob))
        if not sum(new_normalized_prob) == 0.0:
            index = 0
            for s_prim in gid.all_states:
                if new_normalized_prob[index] > ZERO_EPSILON:
                    f.write(str(s_first) + "," + str(act) + "," + str(s_prim) + "," + str(new_normalized_prob[index]) + "\n")
                index += 1

f.close()


# ************************************************
# generating reward data
print(" generating reward data:   ")
reward_file_name = 'reward-data' + file_name_suffix + '.txt'
f_reward_data = open(reward_file_name, "+a")
print("sorted_all_states:   ", sorted_all_states)
for elem in sorted_all_states:
    s = elem[0]
    print("s:  ", s)
    for act in ALL_ACTIONS:
        reward = gid.compute_future_rew(s, act)
        f_reward_data.write(str(s) + "," + str(act) + "," + str(reward) + "\n")

f_reward_data.close()



