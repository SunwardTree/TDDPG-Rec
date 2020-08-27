# coding: utf-8 -*- DQN model
# A method that replaces DDPG in TDDPG-Rec with DQN, while retains other components the same as that in TDDPG-Rec.
# Select the normalized item vector with the max Q-value as policy vector
import tensorflow as tf
import numpy as np
import random
import math
import time
import os
from datetime import datetime
from env import RecommendENV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def feature_normalize(data):
    max_d = np.max(data)
    min_d = np.min(data)
    return (data - min_d)/(max_d - min_d)


#  DQN
#  hyper parameters
LR = 0.0002  # learning rate
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement


def xavier_init(in_num, out_num, constant=1):
    low = -constant * np.sqrt(6.0 / (in_num + out_num))
    high = constant * np.sqrt(6.0 / (in_num + out_num))
    return tf.random_uniform((in_num, out_num), minval=low, maxval=high, dtype=tf.float32)


class DQN(object):
    def __init__(self, sess, item_vector, r_s_dim, r_s_num, batch_size, MEMORY_CAPACITY):
        self.sess = sess
        self.item_vector = item_vector

        self.s_dim = r_s_dim
        self.s_num = r_s_num
        self.i_dim = self.s_dim
        self.batch_size = batch_size

        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.memory = np.zeros((MEMORY_CAPACITY, self.s_dim * (self.s_num * 2) + self.i_dim + 2), dtype=np.float32)
        self.pointer = 0

        self.keep_rate = 1

        self.S = tf.placeholder(tf.float32, [None, self.s_num * self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_num * self.s_dim], 's_')
        self.item = tf.placeholder(tf.float32, [None, self.i_dim], 'item')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.q = self._build_net(self.S, self.item, scope='eval')
        self.q_ = self._build_net(self.S_, self.item, scope='target')
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')

        q_target = self.R + GAMMA * self.q_
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.train = tf.train.AdamOptimizer(LR).minimize(self.td_error, var_list=self.ae_params)

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea)] for ta, ea in zip(self.at_params, self.ae_params)]

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self, i_s, i_i, scope):
        with tf.variable_scope(scope):
            n_l1 = 100
            self.w1_s = tf.Variable(name='w1_s', initial_value=xavier_init(self.s_dim * self.s_num, n_l1))
            self.w1_i = tf.Variable(name='w1_i', initial_value=xavier_init(self.i_dim, n_l1))
            self.b1 = tf.Variable(name='b1', initial_value=tf.zeros([n_l1]))
            layer = tf.nn.relu(
                tf.nn.dropout((tf.matmul(i_s, self.w1_s) + tf.matmul(i_i, self.w1_i) + self.b1), rate=1 - self.keep_rate))
            # Q(s,a)
            self.w2 = tf.Variable(name='w2', initial_value=xavier_init(n_l1, 1))
            self.b2 = tf.Variable(name='b2', initial_value=tf.zeros([1]))
            q_value = tf.matmul(layer, self.w2) + self.b2
            return q_value

    def choose_action(self, i_emb_s, c_list):
        item = np.zeros((len(c_list), self.i_dim))
        i = 0
        for c_item in c_list:
            item[i] = feature_normalize(self.item_vector[c_item])
            i += 1
        i_emb_s = np.tile(i_emb_s, [len(c_list), 1])
        q_ = self.sess.run(self.q, {self.S: i_emb_s, self.item: item})
        max_index = np.argmax(q_)
        # print(max_index, c_list, q_)
        return c_list[max_index]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim * self.s_num]
        bi = bt[:, self.s_dim * self.s_num: self.s_dim * self.s_num + self.i_dim]
        br = bt[:, -(self.s_dim * self.s_num) - 1: -(self.s_dim * self.s_num)]
        bs_ = bt[:, -(self.s_dim * self.s_num):]

        _, it_td_error = self.sess.run([self.train, self.td_error],
                                       feed_dict={self.S: bs, self.item: bi, self.R: br, self.S_: bs_})
        return it_td_error

    def store_transition(self, i_s, i_i, i_a, i_r, i_s_):
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old memory with new memory
        transition = np.hstack((i_s[0], i_i, i_a, i_r, i_s_[0]))
        self.memory[index, :] = transition
        self.pointer += 1

    def set_keep_rate(self, keep_rate):
        self.keep_rate = keep_rate


class RlProcess:
    def __init__(self, the_data_path, the_data_name, data_method, epochs, state_num, action_num, one_u_steps,
                 test_top_k: list, is_use_history=True):
        self.the_data_path = the_data_path
        self.the_data_name = the_data_name
        self.data_method = data_method
        self.epochs = epochs
        self.state_num = state_num
        self.a_num = action_num
        self.one_u_steps = one_u_steps
        self.test_top_k = sorted(test_top_k)
        self.is_use_history = is_use_history
        self.item_vector = np.load(self.the_data_path + self.the_data_name + "_embeddings.npy")
        self.user_label_list, self.user_label_num = None, None
        self.cluster_items, self.cluster_users = None, None
        self.train_user_items_dict, self.train_user_items_rating_dict = None, None
        self.test_user_items_rating_dict = None
        self.nega_user_items_rating_dict = None
        self.supp_nega_cluster_items = None
        self.old_user2new, self.old_item2new = None, None
        self.data_shape, self.max_dis_dict = None, None
        self.test_dict = None
        # Data initialization
        self.data_prepare()

    # Data preparation
    def data_prepare(self):
        with open(self.the_data_path + "user_label_predict.txt", 'r') as l_file:
            self.user_label_list = l_file.read().split(',')  # str
        self.user_label_num = len(set(self.user_label_list)) - 1

        with open(self.the_data_path + 'train_user_items_dict.txt', 'r') as train_ui_dict:
            self.train_user_items_dict = eval(train_ui_dict.read())
        with open(self.the_data_path + 'train_user_items_rating_dict.txt', 'r') as train_uir_dict:
            self.train_user_items_rating_dict = eval(train_uir_dict.read())
        with open(self.the_data_path + 'test_user_items_rating_dict.txt', 'r') as test_uir_dict:
            self.test_user_items_rating_dict = eval(test_uir_dict.read())
        with open(self.the_data_path + 'nega_user_items_rating_dict.txt', 'r') as nega_uir_dict:
            self.nega_user_items_rating_dict = eval(nega_uir_dict.read())

        with open(self.the_data_path + 'cluster_users.txt', 'r') as c_us:
            self.cluster_users = eval(c_us.read())['cluster_users']

        with open(self.the_data_path + 'old_user2new.txt', 'r', encoding="utf-8") as f:
            self.old_user2new = eval(f.read())
        with open(self.the_data_path + 'old_item2new.txt', 'r', encoding="utf-8") as f:
            self.old_item2new = eval(f.read())
        self.data_shape = [len(self.old_user2new.keys()), len(self.old_item2new.keys())]

        # Obtain positive and negative samples for testing directly
        with open(self.the_data_path + 'test_dict.txt', 'r') as test_dict_file:
            self.test_dict = eval(test_dict_file.read())

        # Gets the list of classes that appear in the current class but not the farthest from the current class
        with open(self.the_data_path + 'supp_nega_cluster_items.txt', 'r') as nega_ci_file:
            self.supp_nega_cluster_items = eval(nega_ci_file.read())

    # hit_rate nDCG precision recall -- pre-user
    def result_evaluate(self, user_id: int, top_k_list: list, the_model, in_emb_s):
        one_hit, one_ndcg, one_precision, one_recall = [], [], [], []
        h_test_items = self.test_dict[str(user_id) + '_p'].copy()
        test_candidate_items = h_test_items + self.test_dict[str(user_id) + '_n'].copy()
        # print('True Percent;', len(h_test_items) / len(test_candidate_items))
        random.shuffle(test_candidate_items)

        # Select the first few with the largest Q value
        item = np.zeros((len(test_candidate_items), self.item_vector.shape[1]))
        i = 0
        for c_item in test_candidate_items:
            item[i] = feature_normalize(self.item_vector[c_item])
            i += 1
        i_emb_s = np.tile(in_emb_s, [len(test_candidate_items), 1])
        q = the_model.sess.run(the_model.q, {the_model.S: i_emb_s, the_model.item: item})
        q = -1 * q.reshape(1, len(q))[0]
        q_sort_index = np.argsort(q)
        a_t = np.array(test_candidate_items)[q_sort_index].tolist()[:top_k_list[-1]]
        # print(a_t)
        # print(test_item)
        # print(test_candidate_items)

        for top_k in top_k_list:
            hit_count = 0
            hit_list = []
            dcg = 0
            idcg = 0
            for k in range(len(a_t[:top_k])):
                t_item = a_t[k]
                if t_item in h_test_items:
                    hit_count += 1
                    t_rating = self.test_user_items_rating_dict[user_id][t_item] - 2
                    dcg += t_rating / math.log(k + 2)
                    hit_list.append(t_rating)
            hit_list.sort(reverse=True)
            # print(hit_list)
            kk = 0
            for t_rating in hit_list:
                idcg += t_rating / math.log(kk + 2)
                kk += 1
            if hit_count > 0:
                one_hit.append(1)
                one_ndcg.append(dcg / idcg)
                one_precision.append(hit_count / top_k)
                one_recall.append(hit_count / len(h_test_items))
            else:
                one_hit.append(0)
                one_ndcg.append(0)
                one_precision.append(0)
                one_recall.append(0)
        return one_hit, one_ndcg, one_precision, one_recall

    def runProcess(self):
        start_time = time.process_time()
        emb_size = self.item_vector.shape[1]
        s_dim, a_w_dim = emb_size, emb_size
        if not os.path.exists('./reinforce_log/'):
            os.makedirs('./reinforce_log/')
        reinforce_log = open('./reinforce_log/' + the_data_name + '_dqn_cluster-' + self.data_method
                             + '_cluster' + str(self.user_label_num + 1)
                             + '_state' + str(self.state_num)
                             + '_action' + str(self.a_num)
                             + '_alpha' + str_alpha + '_'
                             + datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt', 'w')

        BATCH_SIZE = 32
        MEMORY_CAPACITY = 100
        boundry_rating = 2
        print('MEMORY_CAPACITY:', MEMORY_CAPACITY)
        env = RecommendENV(s_num=self.state_num, a_num=self.a_num, state_dim=s_dim,
                           item_vector=self.item_vector, supp_nega_cluster_items=self.supp_nega_cluster_items,
                           boundry_rating=boundry_rating, train_user_items_dict=self.train_user_items_dict,
                           train_user_items_rating_dict=self.train_user_items_rating_dict,
                           nega_user_items_rating_dict=self.nega_user_items_rating_dict,
                           user_label_list=self.user_label_list, data_shape=self.data_shape)

        # Control training times parameter setting
        MAX_STEPS = MEMORY_CAPACITY * 2  # Maximum training steps
        MIN_STEPS = MEMORY_CAPACITY * 1  # Minimum training steps, greater than or equal to memory
        once_show_num = 10
        # The convergence stop indicator,
        # stops when the percentage of the sub average value to the original value is less than or equal to this value
        stop_line = 0.1
        c_select_size = 50  # Number of randomly selected candidate set items, make sure select_size < len(c_items_list)
        o_train_percent = 0.1  # Select the proportion of users in the training set to see in the training

        # result_evaluate
        total_hits, total_ndcgs, total_precisions, total_recalls = [], [], [], []
        for _ in self.test_top_k:
            total_hits.append([])
            total_ndcgs.append([])
            total_precisions.append([])
            total_recalls.append([])
        cluster_w = []
        # t_sun = 0
        for i in range(self.user_label_num + 1):
            cluster_w.append(len(self.cluster_users[i]) / self.data_shape[0])
            # t_sun += cluster_w[i]
        # print(cluster_w, t_sun)

        total_cluster_steps = 0
        for i in range(self.user_label_num + 1):
            # user_cluster
            cluster_hits, cluster_ndcgs, cluster_precisions, cluster_recalls = [], [], [], []
            for _ in self.test_top_k:
                cluster_hits.append([])
                cluster_ndcgs.append([])
                cluster_precisions.append([])
                cluster_recalls.append([])

            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as o_sess:
                # Each cluster corresponds to a graph
                dqn = DQN(o_sess, self.item_vector, s_dim, self.state_num, BATCH_SIZE, MEMORY_CAPACITY)
                o_sess.run(tf.global_variables_initializer())
                # Create a saver.
                cluster_saver = tf.train.Saver()
                the_saver_path = './reinforce_log/' + the_data_name + '_dqn_cluster' \
                                 + '/state' + str(self.state_num) \
                                 + '/action' + str(self.a_num) \
                                 + '/' + self.data_method + '/c' + str(i)
                meta_path = the_saver_path + '/model.meta'
                if self.is_use_history:
                    if os.path.exists(meta_path) \
                            and os.path.exists(the_saver_path):
                        cluster_saver = tf.train.import_meta_graph(meta_path)
                        cluster_saver.restore(o_sess, tf.train.latest_checkpoint(the_saver_path))
                        print('Filled with', the_saver_path)
                explore_var = 1  # Initial value of exploration item
                user_size = len(self.cluster_users[i])  # Transboundary control

                # Initialize memory buffer
                for ii in range(MEMORY_CAPACITY):
                    user_id = self.cluster_users[i][int(ii % user_size)]

                    # Construct candidate set once per selection
                    train_num = int(o_train_percent * c_select_size)
                    if train_num > len(self.train_user_items_dict[user_id]):
                        train_num = len(self.train_user_items_dict[user_id])
                    train_list = random.sample(self.train_user_items_dict[user_id], train_num)
                    nega_num = int((c_select_size - train_num) / 2)
                    nega_list = env.getNegative(user_id=user_id, nega_num=nega_num,
                                                supp_nega_cluster_items=self.supp_nega_cluster_items[int(self.user_label_list[user_id])])
                    random_c_list = train_list + nega_list
                    num_random = c_select_size - train_num - len(nega_list)
                    random_list = env.getRandom(random_c_list, num_random)
                    random_c_list += random_list
                    random.shuffle(random_c_list)

                    s, emb_s = env.init_env(user_id=user_id)
                    a_ = dqn.choose_action(emb_s, random_c_list)
                    item_vec = self.item_vector[a_]
                    s_, emb_s_, r = env.step_dqn(user_id=user_id,
                                                 in_state=s,
                                                 in_a=a_,
                                                 train_list=train_list,
                                                 nega_list=nega_list)
                    dqn.store_transition(emb_s, item_vec, a_, r, emb_s_)
                hit_list, ndcg_list, precision_list, recall_list = [], [], [], []
                for epoch in range(self.epochs):
                    str_cluster = 'Cluster:' + str(i)
                    str_td_loss = 'Loss:'
                    str_reward = 'Mean_Reward:'
                    t_sum_steps = 0
                    step_count = 1
                    once_show_r = 0
                    o_td_error = 0

                    # training
                    dqn.set_keep_rate(keep_rate=0.8)
                    while True:
                        user_id = self.cluster_users[i][random.randint(0, user_size - 1)]
                        s, emb_s = env.init_env(user_id=user_id)
                        # A certain number of training for each user
                        for _ in range(self.one_u_steps):
                            train_num = int(o_train_percent * c_select_size)
                            if train_num > len(self.train_user_items_dict[user_id]):
                                train_num = len(self.train_user_items_dict[user_id])
                            train_list = random.sample(self.train_user_items_dict[user_id], train_num)
                            nega_num = int((c_select_size - train_num) / 2)
                            nega_list = env.getNegative(user_id=user_id, nega_num=nega_num,
                                                        supp_nega_cluster_items=self.supp_nega_cluster_items[int(self.user_label_list[user_id])])
                            random_c_list = train_list + nega_list
                            num_random = c_select_size - train_num - len(nega_list)
                            random_list = env.getRandom(random_c_list, num_random)
                            random_c_list += random_list
                            random.shuffle(random_c_list)
                            # print(random_c_list)
                            # print(s, s_emb, s_flag)
                            if explore_var > 0.5:
                                a_ = random.sample(random_c_list, self.a_num)
                                item_vec = self.item_vector[a_]
                            else:
                                a_ = dqn.choose_action(emb_s, random_c_list)
                                item_vec = self.item_vector[a_]
                            s_, emb_s_, r = env.step_dqn(user_id=user_id,
                                                         in_state=s,
                                                         in_a=a_,
                                                         train_list=train_list,
                                                         nega_list=nega_list)
                            dqn.store_transition(emb_s, item_vec, a_, r, emb_s_)

                            s = s_
                            emb_s = emb_s_
                            once_show_r += r

                            # train
                            o_td_error = dqn.learn()
                            # print(o_td_error)

                        if step_count % once_show_num == 0:
                            # print('State_flag:', s_flag)
                            explore_var *= 0.9
                            # print(explore_var)
                            # print('State:', s)
                            new_loss = o_td_error / (once_show_num * self.one_u_steps)
                            str_td_loss += str(new_loss) + ' '
                            str_reward += str(once_show_r / (once_show_num * self.one_u_steps)) + ' '

                            if step_count >= MIN_STEPS:
                                # Take absolute value to prevent division by 0
                                if np.abs(old_loss - new_loss) / (np.abs(old_loss) + 0.000001) < stop_line:
                                    break

                            old_loss = new_loss
                            once_show_r = 0
                            o_td_error = 0
                        # print(t_td_error)
                        if step_count >= MAX_STEPS:
                            break
                        step_count += 1
                    t_sum_steps += step_count
                    str_steps = 'Steps:' + str(t_sum_steps * self.one_u_steps)
                    str_train_log = str_cluster + '\n' + str_td_loss + '\n' + str_reward + '\n' \
                                    + str_steps
                    print(str_train_log)
                    reinforce_log.write(str_train_log + '\n')
                    reinforce_log.flush()
                    total_cluster_steps += step_count

                    # Test and use the parameters of the corresponding class before changing the cluster
                    dqn.set_keep_rate(keep_rate=1)
                    for t_user_id in self.cluster_users[i]:
                        try:
                            self.test_dict[str(t_user_id) + '_p']
                        except KeyError:
                            continue
                        # Initialize test environment
                        s, emb_s = env.init_test_env(t_user_id)
                        # print(s)
                        # test
                        one_hit, one_ndcg, one_precision, one_recall = self.result_evaluate(
                            user_id=t_user_id,
                            top_k_list=self.test_top_k,
                            the_model=dqn,
                            in_emb_s=emb_s)
                        # print(t_user_id, one_hit, one_ndcg, one_precision, one_recall)
                        kk = 0
                        for _ in self.test_top_k:
                            cluster_hits[kk].append(one_hit[kk])
                            cluster_ndcgs[kk].append(one_ndcg[kk])
                            cluster_precisions[kk].append(one_precision[kk])
                            cluster_recalls[kk].append(one_recall[kk])
                            kk += 1
                    # print(len(cluster_hits))
                    str_rate = 'Evaluate of cluster ' + str(i) + ', Epoch ' + str(epoch)
                    kk = 0
                    hit_t, ndcg_t, precision_t, recall_t = [], [], [], []
                    for top_k in self.test_top_k:
                        if len(cluster_hits) > 0:
                            cluster_hit = np.array(cluster_hits[kk]).mean()
                            cluster_ndcg = np.array(cluster_ndcgs[kk]).mean()
                            cluster_precision = np.array(cluster_precisions[kk]).mean()
                            cluster_recall = np.array(cluster_recalls[kk]).mean()
                        else:
                            cluster_hit = 0
                            cluster_ndcg = 0
                            cluster_precision = 0
                            cluster_recall = 0
                        cluster_f1 = 2 * cluster_precision * cluster_recall / (
                                cluster_precision + cluster_recall + 0.000001)
                        hit_t.append(cluster_hit)
                        ndcg_t.append(cluster_ndcg)
                        precision_t.append(cluster_precision)
                        recall_t.append(cluster_recall)
                        str_rate += '\nTop ' + str(top_k) + \
                                    '. Hit_rate:' + str(cluster_hit) + \
                                    ' nDCG:' + str(cluster_ndcg) + \
                                    ' Precision:' + str(cluster_precision) + \
                                    ' Recall:' + str(cluster_recall) + \
                                    ' F1:' + str(cluster_f1)
                        kk += 1
                    hit_list.append(hit_t)
                    ndcg_list.append(ndcg_t)
                    precision_list.append(precision_t)
                    recall_list.append(recall_t)
                    print(str_rate)
                    reinforce_log.write(str_rate + '\n')
                    reinforce_log.flush()
                best_pos = 0
                for ii in range(1, len(hit_list)):
                    if hit_list[ii][0] > hit_list[best_pos][0]:
                        best_pos = ii
                kk = 0
                for _ in self.test_top_k:
                    total_hits[kk].append(hit_list[best_pos][kk] * cluster_w[i])
                    total_ndcgs[kk].append(ndcg_list[best_pos][kk] * cluster_w[i])
                    total_precisions[kk].append(precision_list[best_pos][kk] * cluster_w[i])
                    total_recalls[kk].append(recall_list[best_pos][kk] * cluster_w[i])
                    kk += 1
                # Save model each class has its own model
                if not os.path.exists(the_saver_path):
                    os.makedirs(the_saver_path)
                cluster_saver.save(o_sess, os.path.join(the_saver_path, 'model'))
            # Clear variables previously defined in the default graph
            tf.reset_default_graph()

        str_log = 'dqn_rec'
        kk = 0
        for top_k in self.test_top_k:
            total_hr, total_ndcg, total_precision, total_recall = 0, 0, 0, 0
            # print(total_hits)
            for i in range(self.user_label_num + 1):
                total_hr += total_hits[kk][i]
                total_ndcg += total_ndcgs[kk][i]
                total_precision += total_precisions[kk][i]
                total_recall += total_recalls[kk][i]
            total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 0.000001)
            str_log += '\nTTop ' + str(top_k) + \
                       '. HR:' + str(total_hr) + \
                       ' nDCG:' + str(total_ndcg) + \
                       ' Precision:' + str(total_precision) + \
                       ' Recall:' + str(total_recall) + \
                       ' F1:' + str(total_f1)
            kk += 1
        str_steps = 'Total train steps:' + str(total_cluster_steps * self.one_u_steps)
        end_time = time.process_time()
        str_time = "Cost time is %f" % (end_time - start_time)
        reinforce_log.write(str_log + '\n' + str_steps + ' ' + str_time)
        reinforce_log.flush()
        reinforce_log.close()
        print(str_log + '\n' + str_steps + ' ' + str_time)


if __name__ == '__main__':
    # glove mf
    data_method = 'glove'
    emb_size = 100
    # 'Digital_Music' 'Beauty' 'Clothing_Shoes_and_Jewelry'
    the_data_name = 'Digital_Music'
    state_num = 10  # Number of items in the action
    one_u_steps = 10  # Training times per user
    test_top_k = [10, 20]  # Top_k during test
    str_alpha = '0.5'  # Proportion of product description
    epochs = 3  # Number of training rounds

    the_data_path = './Amazon/' + the_data_name + '/' + data_method + '/' + str(emb_size) + '/'
    rl_model = RlProcess(the_data_path=the_data_path,
                         the_data_name=the_data_name,
                         data_method=data_method,
                         epochs=epochs,
                         state_num=state_num,
                         action_num=1,
                         one_u_steps=one_u_steps,
                         test_top_k=test_top_k,
                         is_use_history=False)
    rl_model.runProcess()
