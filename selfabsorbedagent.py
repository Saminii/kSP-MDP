import numpy as np
import copy
import helper


class SelfAbsorbed:
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
        rob_state_list = [init_loc_key, init_task_status]
        init_state_key = helper.get_key_from_dict_by_value(rob_state_list, all_states)
        self.my_state = init_state_key

        self.values = []
        for _ in all_states:
            self.values.append(0.0)

        act_init_val = []
        for _ in self.all_actions:
            act_init_val.append(0.0)

        self.q_table = []
        for _ in all_states:
            self.q_table.append(act_init_val)

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

    def update_q(self, file_id, all_states_dict):

        all_st = copy.deepcopy(all_states_dict)

        all_st_keys = []
        for i in all_st.keys():
            all_st_keys.append(i)
        all_st_keys.sort()

        temp_q_table = []
        for sta in all_st_keys:
            q_val = []
            for act in self.all_actions:
                p_s_prime = []
                r = self.reward_dict[(sta, int(act))]  # self.compute_future_rew(sta, act)
                for nxt_sta in all_st_keys:
                    p_s_prime.append(self.get_transition(sta, act, nxt_sta))

                sum_elements = sum(p_s_prime)
                normal_p_sprime = [x / sum_elements for x in p_s_prime]
                product = sum([a * b for a, b in zip(normal_p_sprime, self.values)])
                q_val.append(r + product)
            temp_q_table.append(q_val)
        self.q_table = copy.deepcopy(temp_q_table)

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
        new_state_key = helper.get_key_from_dict_by_value(new_state_list, all_st)
        self.my_state = new_state_key

    def set_my_state(self, state, all_states_dict):
        all_st = copy.deepcopy(all_states_dict)
        self.my_state = helper.get_key_from_dict_by_value(state, all_st)
        self.location = self.location_dictionary[state[0]]

    def set_task_status(self, tsk_key, all_states_dict):
        # print(" *** ** ** ** ** *** ")
        all_st = copy.deepcopy(all_states_dict)
        old_state_key = self.my_state
        old_state = all_st[old_state_key]

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
        else:   # Greedy action selection
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
                best_act = self.select_greedy_action(file_id)
            else:
                best_act = self.select_ith_best_act(repeated_ids)

        return best_act

    def select_greedy_action(self, file_id):
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

    def select_second_best_act(self):
        """
        If there are several best action, each robot chooses randomly from the best action set.
        :param file_id:
        :return:
        """

        q_current_state = self.q_table[self.my_state]

        if q_current_state[0] >= q_current_state[1]:
            max = q_current_state[0]
            best_act = self.all_actions[0]
            second_max = q_current_state[1]
            second_best_act = self.all_actions[1]
        else:
            max = q_current_state[1]
            best_act = self.all_actions[1]
            second_max = q_current_state[0]
            second_best_act = self.all_actions[0]

        for i in range(2, len(q_current_state)):
            if q_current_state[i] >= max:
                second_max = max
                second_best_act = best_act
                max = q_current_state[i]
                best_act = self.all_actions[i]
            else:
                if q_current_state[i] >= second_max:
                    second_max = q_current_state[i]
                    second_best_act = self.all_actions[i]

        equal_acts = []
        for i in range(len(q_current_state)):
            q_value = q_current_state[i]
            if abs(second_max - q_value) < self.ZERO_EPSILON:  # which means (max_q == q_value)
                equal_acts.append(self.all_actions[i])

        if len(equal_acts) > 1:
            second_best_act = np.random.choice(equal_acts)
        return second_best_act

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

