import numpy as np
import copy
import helper


class Environment:
    def __init__(self, num_robots , all_actions, num_rows, num_columns, transition_dict, states_dict):
        self.num_robots = num_robots
        self.all_acts = all_actions
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.transition_dict = transition_dict

        # making a grid world full of tasks
        dirt_row = []
        for i in range(self.num_rows):
            dirt_row.append('1')
        self.is_dirt = []
        for i in range(self.num_columns):
            self.is_dirt.append(dirt_row)

        self.all_states = copy.deepcopy(states_dict)
        self.rob_loc_dict = {}
        self.all_robs_loc = []

        self.all_robs_ids = []
        for counter in range(self.num_robots):
            self.all_robs_ids.append(counter)
        print("self.all_robs_ids:  ", str(self.all_robs_ids))

    def init_rob_loc(self):
        """   Initializing robot positions in the grid  """
        # locations = np.zeros((self.num_robots, 2))
        locations = []
        for _ in range(self.num_robots):
            locations.append([0, 0])

        temp_num_robots = 0
        while temp_num_robots < self.num_robots:
            row = np.random.randint(low=0, high=self.num_rows)
            col = np.random.randint(low=0, high=self.num_columns)
            duplicate_flag = False
            for i in range(self.num_robots):
                if locations[i][0] == row and locations[i][1] == col:
                    duplicate_flag = True
                    break

            if not duplicate_flag:
                locations[temp_num_robots][0] = row
                locations[temp_num_robots][1] = col
                temp_num_robots = temp_num_robots + 1

        self.all_robs_loc = copy.deepcopy(locations)
        return locations

    def make_all_rob_loc_dictionary(self):
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
        rob_loc = []
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                rob_loc.append([i, j])

        rob_loc_dict = {}
        for i in range(len(rob_loc)):
            rob_loc_dict[i] = rob_loc[i]
        self.rob_loc_dict = copy.deepcopy(rob_loc_dict)

        # Second making tasks' locations
        task_loc = []
        for i in range(2**(self.num_rows*self.num_columns)):
            task_loc.append(i)

        # we use a dictionary with elements that are list with length 2;
        # the first robot location and the second task location
        states = {}
        k = 0
        for i in rob_loc_dict:
            for j in range(len(task_loc)):
                states[k] = [i, task_loc[j]]
                k += 1
        self.all_states = copy.deepcopy(states)

    def get_transition(self, st, act, next_st):
        p = self.transition_dict.get((int(st), int(act), int(next_st)))

        if p is not None:
            return p
        else:
            return 0.0

    def generate_new_tasks(self, task_status_key, file_id, iteration, task_generation_rate):

        rob_location_keies = []
        for j in range(len(self.all_robs_loc)):
            # print(" j:  ", j, " ,  self.all_robs_loc[j]:", self.all_robs_loc[j])
            rob_location_keies.append(helper.get_key_from_dict_by_value(self.all_robs_loc[j], self.rob_loc_dict))
        # print(" **********  rob_location_keies:  ", rob_location_keies)

        rob_task_related_indices = []
        for k in range(len(rob_location_keies)):
            rob_task_related_indices.append(self.num_rows * self.num_columns - 1 - rob_location_keies[k])
            #  = self.num_rows * self.num_columns - 1 - rob_loc

        # print("rob_task_related_indices:  ", rob_task_related_indices)

        is_dirt = np.binary_repr(task_status_key, width=self.num_columns * self.num_rows)
        new_task_generated = False
        new_is_dirt = []
        for i in range(len(is_dirt)):
            if is_dirt[i] == '0':
                rand_num = np.random.rand()
                # print("  *********  i:  ", i)
                # print("helper.is_included_in_list(rob_task_related_indices: ", helper.is_included_in_list(rob_task_related_indices, i))
                if rand_num <= task_generation_rate and not helper.is_included_in_list(rob_task_related_indices, i):
                    new_is_dirt.append('1')
                    # print(" $#@#$$#@  rand_num: ", rand_num, "\n")
                    new_task_generated = True
                else:
                    new_is_dirt.append('0')
            else:    # element == '1':
                new_is_dirt.append('1')

        str_new_dirt = ''.join(new_is_dirt)
        new_dirt_key = int(str_new_dirt, 2)
        # if new_task_generated:
        #     file_name = "output-data/newTask" + file_id + ".txt"
        #     fnewtsk = open(file_name, "a+")
        #     fnewtsk.write(" iteration:  " + str(iteration) + "\t \t")
        #     fnewtsk.write(str(new_dirt_key) + "\n")
        #     fnewtsk.close()
        return new_dirt_key

    def execute_action(self, action, rob_loc, tsk_status_deci):
        # print("action, rob_loc, tsk_status_deci: ", action, rob_loc, tsk_status_deci)
        all_st = copy.deepcopy(self.all_states)

        all_robot_loc_dict = copy.deepcopy(self.rob_loc_dict)
        # for key4, val4 in all_robot_loc_dict.items():
        #     if val4 == rob_loc:
        #         loc_key = key4
        #         break
        loc_key = helper.get_key_from_dict_by_value(rob_loc, all_robot_loc_dict)

        # for key8, val8 in all_st.items():
        #     if val8 == [loc_key, tsk_status_deci]:
        #         current_state_key = key8
        #         break

        # Using random simulation to find the next state:
        # s_probs = [self.get_transition(current_state_key, action,  sj) for sj in all_st.keys()]
        # state = all_st.get(helper.draw_arg(s_probs))
        # return state

        # We use this instead of simulation because we need to fix rate of generating new tasks in the system, even
        # when there are multiple robots in the environment
        if action == '1':  # stay in place
            # print(" ******** action == '1'  ******* ")
            tsk_status_bin_str = np.binary_repr(tsk_status_deci, width=self.num_columns * self.num_rows)
            # print(" FIRST tsk_status_bin_str:   ", tsk_status_bin_str)
            tsk_status_bin_list = []
            for elm in tsk_status_bin_str:
                tsk_status_bin_list.append(elm)
            rob_task_index = self.num_rows * self.num_columns - 1 - loc_key
            tsk_status_bin_list[rob_task_index] = '0'
            str_new_task_status = ''.join(tsk_status_bin_list)
            new_task_status = int(str_new_task_status, 2)
            # print(" new_task_status:   ", new_task_status)
            return [loc_key, new_task_status]
        else:
            rnd_number = np.random.randint(low=0, high=10)
            if rnd_number == 0:   # No motion just for random error :(
                # print("rnd_number:  0 ", rnd_number)
                return [loc_key, tsk_status_deci]
            else:
                current_rob_loc = copy.deepcopy(self.rob_loc_dict.get(loc_key))
                # print(" current_rob_loc:   ", current_rob_loc)
                if action == '2':  # North:2
                    if current_rob_loc[0] == self.num_rows - 1:
                        return [loc_key, tsk_status_deci]
                    else:   # if not current_rob_loc[0] == self.num_rows - 1
                        current_rob_loc[0] += 1
                elif action == '3':  # South:3
                    if current_rob_loc[0] == 0:
                        return [loc_key, tsk_status_deci]
                    else:   # if not current_rob_loc[0] == 0
                        current_rob_loc[0] -= 1
                elif action == '4':  # East:4
                    if current_rob_loc[1] == self.num_columns - 1:
                        return [loc_key, tsk_status_deci]
                    else:   # if not current_rob_loc[1] == self.num_columns - 1:
                        current_rob_loc[1] += 1
                elif action == '5':  # West:5
                    if current_rob_loc[1] == 0:
                        return [loc_key, tsk_status_deci]
                    else:   # if not current_rob_loc[1] == 0
                        current_rob_loc[1] -= 1

                for key5, val5 in self.rob_loc_dict.items():
                    if val5 == current_rob_loc:
                        new_loc_key = key5

                # print(" return  [new_loc_key, tsk_status_deci]",  [new_loc_key, tsk_status_deci])
                return [new_loc_key, tsk_status_deci]

    def set_robot_location(self, st, rob_id):
        all_rob_loc_dict = copy.deepcopy(self.rob_loc_dict)
        # print("robot_state:  ", st)
        self.all_robs_loc[rob_id] = all_rob_loc_dict.get(st[0])
        # print("self.all_robs_loc[rob_id]:  ", self.all_robs_loc[rob_id])

    def calculate_reward(self, task_status_key):
        # print(" task_status_key:  ", task_status_key)
        bin_tsk = np.binary_repr(task_status_key, width=self.num_rows * self.num_columns)
        # print(" bin_tsk: ", bin_tsk)
        rew = 0.0
        for tsk_state in bin_tsk:
            if tsk_state == '0':
                rew += 1.0
        return rew


