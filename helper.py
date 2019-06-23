import numpy as np
import copy


def draw_arg(probabs):
    assert (abs(sum(probabs) - 1.0) < 0.00000001)
    probability = np.array(probabs)
    # Do a second normalisation to avoid the problem described here:
    # https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
    my_choice = np.random.choice(list(range(len(probability))), p=probability / probability.sum())
    return my_choice


def dot_product(vector1, vector2):
    product_value = sum([a * b for a, b in zip(vector1, vector2)])
    return product_value


def is_included_in_list(main_list, query):
    for elem in main_list:
        if elem == query:
            return True
    return False


def get_key_from_dict_by_value(value, dictionary):
    temp_dict = copy.deepcopy(dictionary)
    output_key = None
    for key, val in temp_dict.items():
        if val == value:
            output_key = key
            break

    return output_key


def find_element_in_list(main_list, element):  # returns common indices
    indices = []
    for i in range(len(main_list)):
        if main_list[i] == element:
            indices.append(i)
    return indices


def is_less_from_all_elements(main_list, query_element):
    for i in range(len(main_list)):
        if main_list[i] <= query_element:
            return False
    return True


def get_sum_two_lists(list1, list2):
    # if not len(list1) == len(list2)
    assert (len(list1) == len(list2))

    sum_list = []
    for i in range(len(list1)):
        sum_list.append(list1[i] + list2[i])

    return sum_list


def normalize_probability(probability):
    sum_probabilities = sum(probability)
    normalized_prob = [x / sum_probabilities for x in probability]
    return normalized_prob

