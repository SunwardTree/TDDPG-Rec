import numpy as np
import random
import math


# Constant time complexity instead of sample
def constant_sample(o_list, sample_num: int):
    sample_list = []
    in_list = list(o_list).copy()
    for i in range(sample_num):
        r_n = random.randint(0, len(in_list) - 1)
        # print(r_n, len(in_list))
        sample_list.append(in_list[r_n])
        in_list.pop(r_n)
    return sample_list


class RecommendENV:
    def __init__(self, s_num, a_num, state_dim: int, item_vector, boundry_rating,
                 train_user_items_dict, train_user_items_rating_dict, nega_user_items_rating_dict,
                 user_label_list, data_shape, supp_nega_cluster_items):
        self.state_num = s_num
        self.action_num = a_num
        self.state_dim = state_dim
        self.item_vector = item_vector
        self.boundry_rating = boundry_rating
        self.train_user_items_dict = train_user_items_dict
        self.train_user_items_rating_dict = train_user_items_rating_dict
        self.nega_user_items_rating_dict = nega_user_items_rating_dict
        self.supp_nega_cluster_items = supp_nega_cluster_items
        self.user_label_list = user_label_list
        self.data_shape = data_shape

        self.nega_ui_dic = {}
        for u_id in self.nega_user_items_rating_dict.keys():
            try:
                nega_items_list = list((self.nega_user_items_rating_dict[u_id]).keys())
            except KeyError:
                nega_items_list = []
            self.nega_ui_dic[u_id] = nega_items_list

    # Initialize environment
    def init_env(self, user_id: int):
        state_emd = np.zeros((1, self.state_num * self.state_dim))
        t_count = 0
        h_train_items = self.train_user_items_dict[user_id].copy()
        if len(h_train_items) >= self.state_num:
            in_state = constant_sample(h_train_items, self.state_num)
            for i_item in in_state:
                state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                t_count += 1
        else:
            in_state = h_train_items
            while len(in_state) < self.state_num:
                in_state.append(-1)
            for i_item in in_state:
                if i_item == -1:
                    state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                else:
                    state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                t_count += 1
        return in_state, state_emd

    # User status during initialization test
    def init_test_env(self, user_id: int):
        state_emd = np.zeros((1, self.state_num * self.state_dim))
        t_count = 0
        h_train_items = self.train_user_items_dict[user_id].copy()
        if len(h_train_items) >= self.state_num:
            # Last status on user's timeline
            in_state = h_train_items[len(h_train_items) - self.state_num:]
            for i_item in in_state:
                state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                t_count += 1
        else:
            in_state = h_train_items
            while len(in_state) < self.state_num:
                in_state.append(-1)
            for i_item in in_state:
                if i_item == -1:
                    state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                else:
                    state_emd[0][t_count * self.state_dim:(t_count + 1) * self.state_dim] = self.item_vector[i_item]
                t_count += 1
        return in_state, state_emd

    # The environment determines the next state and return through the current state,
    # the weight of actor network output, and randomly generating candidate set
    def step(self, user_id, in_state, in_a_w,
             select_size: int, train_percent: float):
        in_state_ = in_state.copy()
        in_emb_s_ = np.zeros((1, self.state_num * self.state_dim))
        # Select the most suitable item from the candidate set and calculate according to action
        train_num = int(train_percent * select_size)
        if train_num > len(self.train_user_items_dict[user_id]):
            train_num = len(self.train_user_items_dict[user_id])
        train_list = constant_sample(self.train_user_items_dict[user_id], train_num)
        try:
            nega_items_list = self.nega_ui_dic[user_id]
        except KeyError:
            nega_items_list = []
        nega_num = int((select_size - train_num)/2)
        nega_list = self.getNegative(user_id=user_id, nega_num=nega_num,
                                     supp_nega_cluster_items=self.supp_nega_cluster_items[int(self.user_label_list[user_id])])
        random_c_list = train_list + nega_list
        num_random = select_size - train_num - len(nega_list)
        random_list = self.getRandom(random_c_list, num_random)
        random_c_list += random_list
        random.shuffle(random_c_list)
        # print(random_c_list)
        # Select top k items from the candidate set according to weight in a
        c_score_list = list()
        for c_item in random_c_list:
            c_item = int(c_item)
            score = np.sum(np.multiply(in_a_w, self.item_vector[c_item]))
            c_score_list.append([c_item, score])
        # Select a_num as action
        a_t = []
        for ii in range(self.action_num):
            r_item = -1
            max_score = -1
            for c_score in c_score_list:
                c_item = c_score[0]
                score = c_score[1]
                if score > max_score and c_item not in a_t:
                    max_score = score
                    r_item = c_item
            a_t.append(r_item)
        # print('a_t:', a_t)
        reward = 0
        ii = 0
        for item_a_t in a_t:
            # Distinguish the positions that appear, and the ahead is significant
            # position_weight = (len(in_state) - ii)/len(in_state)
            position_weight = 1 / math.log(ii + 2)
            # Every hit, reward + position_weight
            if item_a_t in train_list:
                reward += (self.train_user_items_rating_dict[user_id][item_a_t] - self.boundry_rating)\
                          * position_weight
            elif item_a_t in nega_items_list:
                reward += (self.nega_user_items_rating_dict[user_id][item_a_t] - self.boundry_rating - 1) * position_weight
            elif item_a_t in nega_list:
                reward += -0.5 * position_weight
            # The sliding window is replaced from the front to the back and is not repeated
            if item_a_t not in in_state_:
                in_state_.pop()
                in_state_.insert(0, item_a_t)
            ii += 1
        # Update emb
        ii = 0
        for s_item_ in in_state_:
            in_emb_s_[0][ii * self.state_dim:(ii + 1) * self.state_dim] = self.item_vector[s_item_]
            ii += 1
        # print('state_:', in_state_)
        return in_state_, in_emb_s_, reward

    def step_dqn(self, user_id, in_state, in_a, train_list, nega_list):
        in_state_ = in_state.copy()
        in_emb_s_ = np.zeros((1, self.state_num * self.state_dim))
        try:
            nega_items_list = self.nega_ui_dic[user_id]
        except KeyError:
            nega_items_list = []
        reward = 0
        item_a_t = in_a
        # print(in_a)
        if item_a_t in train_list:
            reward += self.train_user_items_rating_dict[user_id][item_a_t] - self.boundry_rating
        elif item_a_t in nega_items_list:
            reward += self.nega_user_items_rating_dict[user_id][item_a_t] - self.boundry_rating - 1
        elif item_a_t in nega_list:
            reward += -0.5
        # The sliding window is replaced from the front to the back and is not repeated
        if item_a_t not in in_state_:
            in_state_.pop()
            in_state_.insert(0, item_a_t)
        # update emb
        ii = 0
        for s_item_ in in_state_:
            in_emb_s_[0][ii * self.state_dim:(ii + 1) * self.state_dim] = self.item_vector[s_item_]
            ii += 1
        # print('state_:', in_state_)
        return in_state_, in_emb_s_, reward

    # Get negative samples
    def getNegative(self, user_id: int, nega_num: int, supp_nega_cluster_items):
        try:
            nega_items_list = self.nega_ui_dic[user_id].copy()
        except KeyError:
            nega_items_list = []

        if len(nega_items_list) > 0:
            if len(nega_items_list) >= nega_num:
                negative_list = constant_sample(nega_items_list, nega_num)
            else:
                negative_list = nega_items_list
                if nega_num - len(negative_list) >= len(supp_nega_cluster_items):
                    negative_list += list(supp_nega_cluster_items)
                else:
                    negative_list = constant_sample(supp_nega_cluster_items, nega_num - len(negative_list))
        else:
            if nega_num >= len(supp_nega_cluster_items):
                negative_list = list(supp_nega_cluster_items)
            else:
                negative_list = constant_sample(supp_nega_cluster_items, nega_num)
        return negative_list

    # Get random
    def getRandom(self, exit_list, num_random):
        random_list = []
        while True:
            if len(random_list) == num_random:
                break
            one = np.random.randint(self.data_shape[1])
            if (one not in random_list) and \
                    (one not in exit_list):
                random_list.append(one)
        return random_list
