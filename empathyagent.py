import numpy as np
import copy
import math
import helper


class Empathy:
    def __init__(self, all_states, all_actions, initial_location, loc_dict, num_columns, num_rows,
                 transition_dict, init_task_status, eq_eps, rob_id, reward_dict):
        self.location = initial_location
        self.all_actions = all_actions  # Actions include: { Stay:1, North:2, South:3, East:4, West:5}
        self.location_dictionary = copy.deepcopy(loc_dict)
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.transition_dict = copy.deepcopy(transition_dict)
        self.ZERO_EPSILON = eq_eps
        self.rob_id = rob_id
        self.reward_dict = copy.deepcopy(reward_dict)

        init_loc_key = helper.get_key_from_dict_by_value(initial_location, loc_dict)
        self.my_state = helper.get_key_from_dict_by_value([init_loc_key, init_task_status], all_states)

        self.values = []
        for _ in all_states:
            self.values.append(0.0)

        act_init_val = []
        for _ in self.all_actions:
            act_init_val.append(2.5)  # mean rewards

        self.q_table = []
        for _ in all_states:
            self.q_table.append(act_init_val)

        self.rob_state_mapping = {}

    def compute_future_rew(self, state, action):
        """
        computing future reward for the given action;
            if action == STAY, we made the robot's location state CLEAR
        :param state:
        :param action:
        :return: rew
        """
        dirt_status_code = self.all_states[state][1]
        rob_loc = self.all_states[state][0]
        is_dirt = np.binary_repr(dirt_status_code, width=self.num_columns * self.num_rows)
        is_dirt_list = []
        for elem in is_dirt:
            is_dirt_list.append(elem)

        rob_tsk_index = self.num_rows * self.num_columns - 1 - rob_loc
        if action == '1':
            is_dirt_list[rob_tsk_index] = '0'

        rew = 0.0
        for i in range(len(is_dirt_list)):
            if is_dirt_list[i] == '0':
                rew += 1.0
        return rew

    def get_reward(self, state, all_states_dict):
        all_st = copy.deepcopy(all_states_dict)
        dirt_status_code = all_st[state][1]
        is_dirt_now = np.binary_repr(dirt_status_code, width=self.num_columns * self.num_rows)

        r = 0.0
        for e in is_dirt_now:
            if e == '0':
                r += 1.0

        return r

    def get_transition(self, st, act, next_st):
        key = (st, int(act), next_st)
        p = self.transition_dict.get(key)

        if p is not None:
            return p
        else:
            return 0.0

    def update_values(self):
        values = []
        for i in range(len(self.values)):
            max_s = max(self.q_table[i])
            values.append(max_s)
        self.values = copy.deepcopy(values)

    def modify_rob_state_by_tsk_change(self, tsk_sts_key, all_states_dict):
        """
        changing robot's state when task status changes but robot's position is fixed.
        :param tsk_sts_key:
        :param all_states_dict:
        :return:
        """
        all_st = copy.deepcopy(all_states_dict)
        current_state_key = self.my_state
        current_state_list = all_st[current_state_key]
        current_pos_key = current_state_list[0]
        new_state_list = [current_pos_key, tsk_sts_key]
        self.my_state = helper.get_key_from_dict_by_value(new_state_list, all_st)

    def set_my_state(self, state, all_states_dict):
        all_st = copy.deepcopy(all_states_dict)
        self.my_state = helper.get_key_from_dict_by_value(state, all_st)

        all_locations_dict = copy.deepcopy(self.location_dictionary)
        self.location = all_locations_dict.get(state[0])

    def set_task_status(self, tsk_key, all_states_dict):
        all_st = copy.deepcopy(all_states_dict)
        old_state_key = self.my_state
        old_state = all_st.get(old_state_key)

        old_state[1] = tsk_key
        self.my_state = helper.get_key_from_dict_by_value(old_state, all_st)

    def choose_action(self, epsilon_greedy_value, file_id, all_rob_loc, all_rob_ids):
        """
        If epsilon_greedy_value > 0 we may decide to take a random action (random simulation)
        else we use a greedy action, but it depends on location of others: if two or more agents are in the same
            location, actions are sorted by their values and then selected on the basis of robots ids.
        :param epsilon_greedy_value:
        :param file_id:
        :param all_rob_loc:
        :param all_rob_ids:
        :return: best action
        """

        rand_num = np.random.rand()
        if rand_num < epsilon_greedy_value:
            best_act = np.random.choice(self.all_actions)
        else:  # Greedy action selection
            other_locations = []
            other_ids = []
            for i in range(len(all_rob_loc)):
                if not all_rob_ids[i] == self.rob_id:
                    other_locations.append(all_rob_loc[i])
                    other_ids.append(all_rob_ids[i])

            repetition_flag = helper.is_included_in_list(other_locations, self.location)
            common_indices = helper.find_element_in_list(other_locations, self.location)
            repeated_ids = []
            for j in range(len(common_indices)):
                repeated_ids.append(other_ids[common_indices[j]])

            less_than_id_flag = False
            if repetition_flag:  # We use a social low for that: if two agents are in the same location,
                # the one with smaller id takes the best action and the other choose the second best action.
                less_than_id_flag = helper.is_less_from_all_elements(repeated_ids, self.rob_id)

            if not repetition_flag or less_than_id_flag:
                best_act = self.select_greedy_action()
            else:
                best_act = self.select_ith_best_act(repeated_ids)

        return best_act

    def select_greedy_action(self):
        """
        If there are more than one best action, robot chooses randomly from them.
        :param file_id:
        :return:
        """

        q_for_current_state = self.q_table[self.my_state]
        max_q = -1000000.0
        best_act = None
        for i in range(len(q_for_current_state)):
            q_val = q_for_current_state[i]
            if q_val > max_q:
                max_q = q_val
                best_act = self.all_actions[i]

        # If there are more than one best action, it's better to choose randomly from them
        equal_acts = []
        for i in range(len(q_for_current_state)):
            q_value = q_for_current_state[i]
            if abs(max_q - q_value) < self.ZERO_EPSILON:  # which means (max_q == q_value)
                equal_acts.append(self.all_actions[i])

        if len(equal_acts) > 1:
            best_act = np.random.choice(equal_acts)

        return best_act

    def select_ith_best_act(self, repeated_ids):
        q_current_state = self.q_table[self.my_state]
        sorted_arr = sorted(range(len(q_current_state)), key=lambda k: q_current_state[k])
        sorted_arr.reverse()

        sorted_actions = []
        for elem in sorted_arr:
            sorted_actions.append(str(elem + 1))

        num_less_id = 0
        for e in repeated_ids:
            if e < self.rob_id:
                num_less_id += 1

        best_ith_act = sorted_actions[num_less_id]
        return best_ith_act

    def compute_trans_prob_loc(self, first_state_key, action, next_state_key, all_st_dict):

        current_loc = self.location_dictionary.get(all_st_dict.get(first_state_key)[0])
        later_loc = self.location_dictionary.get(all_st_dict.get(next_state_key)[0])

        if action == '1':
            if later_loc == current_loc:
                p_rob_move = 1.0
            else:
                p_rob_move = 0.0
        elif action == "2":   # North:2
            if later_loc == current_loc:
                if current_loc[0] == self.num_rows-1:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[1] == current_loc[1] and later_loc[0] == current_loc[0] + 1:
                p_rob_move = 0.9
            else:
                p_rob_move = 0.0
        elif action == "3":  # South:3
            if later_loc == current_loc:
                if current_loc[0] == 0:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[1] == current_loc[1] and later_loc[0] == current_loc[0] - 1:
                p_rob_move = 0.9
            else:
                p_rob_move = 0.0
        elif action == "4":  # East:4:
            if later_loc == current_loc:
                if current_loc[1] == self.num_columns - 1:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[0] == current_loc[0] and later_loc[1] == current_loc[1] + 1:
                p_rob_move = 0.9
            else:
                p_rob_move = 0.0
        elif action == "5":  # West:5:
            if later_loc == current_loc:
                if current_loc[1] == 0:
                    p_rob_move = 1.0
                else:
                    p_rob_move = 0.1
            elif later_loc[0] == current_loc[0] and later_loc[1] == current_loc[1] - 1:
                p_rob_move = 0.9
            else:
                p_rob_move = 0.0

        return p_rob_move

    def make_policy_vec_by_q_values(self, state):
        """
        This method uses a boltzmann distribution with q values as exponent,
            i.e, exp(q(s,a))/[Sum(exp(q(s,a)) for each a in A]
        :param state:
        :return: policy_vec
        """

        k = 1               # Boltzmann's constant
        temperature = 1     # thermodynamic temperature
        un_normalized_policy_vec = []

        for element in self.q_table[state]:
            un_normalized_policy_vec.append(np.exp(element/(k * temperature)))
        sum_probability = sum(un_normalized_policy_vec)

        policy_vec = [x / sum_probability for x in un_normalized_policy_vec]
        return policy_vec

    def compute_presence_mass(self, loc_dict, all_st_dict, all_rob_loc_list, tsk_key, horizon, all_rob_ids, file_id):

        others_loc = []
        for counter in range(len(all_rob_loc_list)):
            if not all_rob_ids[counter] == self.rob_id:
                others_loc.append(all_rob_loc_list[counter])

        all_robot_loc_dict = copy.deepcopy(loc_dict)
        all_loc_keys = list(all_robot_loc_dict.keys())
        all_loc_keys.sort()

        others_loc_key = []
        for location in others_loc:
            others_loc_key.append(helper.get_key_from_dict_by_value(location, all_robot_loc_dict))

        others_st_keys = []   # [other_loc_key, tsk_key]
        for loc_key in others_loc_key:
            others_st_keys.append(helper.get_key_from_dict_by_value([loc_key, tsk_key], all_st_dict))

        p0 = []
        for _ in others_loc:
            zero_prob = []
            for _ in all_st_dict:
                zero_prob.append(0.0)
            p0.append(zero_prob)

        for i in range(len(p0)):
            p0[i][others_st_keys[i]] = 1.0

        all_st_keys = list(all_st_dict.keys())
        all_st_keys.sort()

        p_mass = []  # [p0]
        for _ in range(len(others_loc)):
            p_mass.append([])

        for agent_counter in range(len(p_mass)):
            for h in range(horizon):
                probs = []
                for st_next in all_st_keys:
                    sum_probs = 0.0
                    for st in all_st_keys:
                        p0_st = p0[agent_counter][st]

                        policy_vec = self.make_policy_vec_by_q_values(st)

                        trans_vec = [self.get_transition(st, action, st_next) for action in self.all_actions]

                        dp = helper.dot_product(policy_vec, trans_vec)

                        sum_probs += dp * p0_st
                    probs.append(sum_probs)
                sum_probabilities = sum(probs)
                normalized_prob = [x / sum_probabilities for x in probs]
                p_mass[agent_counter].append(normalized_prob)
                p0[agent_counter] = copy.deepcopy(normalized_prob)
        print(" %%%%   p0:  ", p0)

        # computed p_mass is related to states that consist of task status and robot location. However for calculating
        # the best response, the probability of being at a particular location is important. So we sum to get just
        # robot position probability:
        # ************  Initializing
        pmass_loc = []
        for age in range(len(p_mass)):
            p_mass_loc_horizon = []
            for h in range(len(p_mass[0])):
                pmass_loc_init = []
                for i in range(len(all_loc_keys)):
                    pmass_loc_init.append(0.0)
                p_mass_loc_horizon.append(pmass_loc_init)
            pmass_loc.append(p_mass_loc_horizon)
        # ************  Summing
        for ag in range(len(p_mass)):
            for h in range(len(p_mass[0])):
                for key in all_st_keys:
                    pmass_loc[ag][h][all_st_dict.get(key)[0]] += p_mass[ag][h][key]

        # ************  Normalizing
        p_mass_normalized = []
        for agent_counter in range(len(pmass_loc)):
            p_mass_new = []
            for p_horizon in pmass_loc[agent_counter]:
                p_mass_new.append(helper.normalize_probability(p_horizon))
            p_mass_normalized.append(p_mass_new)

        return p_mass_normalized

    def compute_aggregated_p_mass(self, p_mass_loc):
        """
        It is not important who does a specific task, what is important is that a task be done by someone.
            So agents computes aggregated presence mass, i.e., sums over agents for each horizon.
        :param p_mass_loc:
        :return: aggregated_pmass
        """
        p_mass = copy.deepcopy(p_mass_loc)

        # Now we sum over all agents in order to find aggregated presence mass
        init_sum = []
        for i in range(len(p_mass[0][0])):
            init_sum.append(0.0)

        aggregated_pmass = []
        for h in range(len(p_mass[0])):
            sum_list = copy.deepcopy(init_sum)
            for ag in range(len(p_mass)):
                current_list = copy.deepcopy(p_mass[ag][h])

                new_list = helper.get_sum_two_lists(sum_list, current_list)
                sum_list = copy.deepcopy(new_list)
            aggregated_pmass.append(helper.normalize_probability(sum_list))

        return aggregated_pmass

    def update_q_empathy(self, all_states_dict, p_mass_agg, horizon):

        all_st = copy.deepcopy(all_states_dict)

        all_st_keys = []
        for i in all_st.keys():
            all_st_keys.append(i)
        all_st_keys.sort()

        temp_q_table = []

        temp_v = []  # [max(q) for q in self.q_table]
        for _ in range(len(all_st_keys)):
            temp_v.append(0.0)

        max_r_vec = []
        for a in self.all_actions:
            r_vec_a = [self.reward_dict.get((s, int(a))) for s in all_st_keys]
            # print(" len(r_vec_a):   ", len(r_vec_a))
            # print("r_vec_a:  ", r_vec_a)
            max_r_vec.append(max(r_vec_a))

        max_q_vec = []
        for a in self.all_actions:
            q_vec_a = [self.q_table[s][int(a)-1] for s in all_st_keys]
            max_q_vec.append(max(q_vec_a))

        f_constant = []
        for i in range(len(max_r_vec)):
            f_constant.append(max_r_vec[i] / max_q_vec[i])

        for h in range(horizon-1, -1, -1):
            for sta in all_st_keys:
                q_val = []
                act_counter = -1
                for act in self.all_actions:
                    act_counter += 1
                    f_a = f_constant[act_counter]
                    r = self.reward_dict[(sta, int(act))]  # self.compute_future_rew(sta, act)
                    p_s_prime = []
                    for nxt_sta in all_st_keys:
                        p_s_prime.append(self.get_transition(sta, act, nxt_sta))
                    p_mas_agg_h = copy.deepcopy(p_mass_agg[h])
                    p_mass_temp = []
                    for prob in p_mas_agg_h:
                        p_mass_temp.append(1 - f_a * prob)       # ?????
                    modified_v = []

                    for s in all_st_keys:
                        st_loc = all_st.get(s)[0]
                        modified_v.append(p_mass_temp[st_loc] * temp_v[s])

                    product = helper.dot_product(p_s_prime, modified_v)
                    q_val.append(r + product)

                temp_q_table.append(q_val)
            temp_v = [max(q) for q in temp_q_table]
            last_q_table = copy.deepcopy(temp_q_table)
            temp_q_table = []

        self.q_table = copy.deepcopy(last_q_table)

    # def compute_task_dist_to_rob_loc(self, rob_loc_key, tasks_key):
    #     rob_loc_list = self.location_dictionary.get(rob_loc_key)
    #     # print(" rob_loc_list: ", rob_loc_list)
    #     tasks_binary = np.binary_repr(tasks_key, width=self.num_columns * self.num_rows)
    #     tasks_list = []
    #     for t in tasks_binary:
    #         tasks_list.append(t)
    #     # print("tasks_list:  ", tasks_list)
    #     tasks_list.reverse()
    #     # print("tasks_list:  ", tasks_list)
    #     dist_list = []
    #     for i in range(len(tasks_list)):
    #         # print(" i: ", i)
    #         if tasks_list[i] == '0':
    #             dist_list.append(math.inf)
    #         else:
    #             tsk_loc = self.location_dictionary.get(i)
    #             # print("  tsk_loc:  ", tsk_loc)
    #             dist_list.append(abs(tsk_loc[0] - rob_loc_list[0]) + abs(tsk_loc[1] - rob_loc_list[1]))
    #     # print("   dist_list:  ", dist_list)
    #     return dist_list

    # def compute_rob_k_nearest_tasks(self, k, task_dist_list):
    #     sorted_arr = sorted(range(len(task_dist_list)), key=lambda j: task_dist_list[j])
    #     # print("  sorted_arr:  ", sorted_arr)
    #     new_tsk_list = []
    #     for i in range(len(sorted_arr)):
    #         new_tsk_list.append('0')
    #     for j in range(k):
    #         new_tsk_list[sorted_arr[j]] = '1'
    #     # print(" new_tsk_list:  ", new_tsk_list)
    #     # str1 = ''.join(new_tsk_list)
    #     # print(" str1: ", str1)
    #
    #     return new_tsk_list

    # def compute_num_active_tsks(self, tsk_key):
    #     tasks_binary = np.binary_repr(tsk_key, width=self.num_columns * self.num_rows)
    #     num_actives = 0
    #     for t in tasks_binary:
    #         if t == '1':
    #             num_actives += 1
    #     return num_actives
    #
    # def make_mapping_from_global_tasks(self, all_states, k):
    #     rob_state_mapping = {}
    #     for st in all_states:
    #         rob_loc = all_states.get(st)[0]
    #         tsk = all_states.get(st)[1]
    #         # print("\n st: ", st, " rob_loc:  ", rob_loc, "tsk: ", tsk)
    #         num_active_tasks = self.compute_num_active_tsks(tsk)
    #         # print("num_active_tasks:  ", num_active_tasks)
    #         if num_active_tasks <= k:
    #             rob_state_mapping[st] = st
    #         else:
    #             dist_list_tsks = self.compute_task_dist_to_rob_loc(rob_loc, tsk)
    #             # print("   dist_list_tsks:  ", dist_list_tsks)
    #             k_nearest_tsks = self.compute_rob_k_nearest_tasks(k, dist_list_tsks)
    #             # print("   k_nearest_tsks:  ", k_nearest_tsks)
    #             k_nearest_tsks.reverse()
    #             tsk_str = ''.join(k_nearest_tsks)
    #             tsk_decimal = int(tsk_str, 2)
    #             # print(" tsk_decimal: ", tsk_decimal)
    #             # tsk_mapping[tsk] = tsk_decimal
    #             new_state = [rob_loc, tsk_decimal]
    #             rob_state_mapping[st] = helper.get_key_from_dict_by_value(new_state, all_states)
    #
    #     self.rob_state_mapping = copy.deepcopy(rob_state_mapping)
    #
    #     file_name = "output-data/tsk_mapping"  + str(self.rob_id) + ".txt"
    #     f = open(file_name, "a+")
    #     f.write(str(rob_state_mapping) + "\n\n")
    #     f.close()





