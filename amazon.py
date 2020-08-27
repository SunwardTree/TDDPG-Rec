# To do with amazon data
# http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/

import os
import time
import random
import numpy as np
from sklearn.cluster import KMeans


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


# using pmf to get item embedding
def mf_with_bias(data_shape, emb_size, rating_list, lr=1e-2, l2_factor=1e-2, max_step=1000, train_rate=0.95, max_stop_count=30):
    rating = np.array(rating_list)
    user_num = data_shape[0]
    item_num = data_shape[1]
    boundry_user_id = int(user_num * 0.8)
    print('training pmf...')

    data = np.array(list(filter(lambda x: x[0] < boundry_user_id, rating)))
    np.random.shuffle(data)

    t = int(len(data)*train_rate)
    dtrain = data[:t]
    dtest = data[t:]

    user_embeddings = tf.Variable(tf.truncated_normal([user_num, emb_size], mean=0, stddev=0.01))
    item_embeddings = tf.Variable(tf.truncated_normal([item_num, emb_size], mean=0, stddev=0.01))
    item_bias = tf.Variable(tf.zeros([item_num, 1], tf.float32))

    user_ids = tf.placeholder(tf.int32, shape=[None])
    item_ids = tf.placeholder(tf.int32, shape=[None])
    ys = tf.placeholder(tf.float32, shape=[None])

    user_embs = tf.nn.embedding_lookup(user_embeddings, user_ids)
    item_embs = tf.nn.embedding_lookup(item_embeddings, item_ids)
    ibias_embs = tf.nn.embedding_lookup(item_bias, item_ids)
    dot_e = user_embs * item_embs

    ys_pre = tf.reduce_sum(dot_e, 1)+tf.squeeze(ibias_embs)

    target_loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre))
    loss = target_loss + l2_factor * (tf.reduce_mean(tf.square(user_embs) + tf.square(item_embs)))

    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.shuffle(dtrain)
        rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss], feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
        rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (0, rmse_train, rmse_test, loss_v, target_loss_v))
        pre_rmse_test = 100.0
        stop_count = 0
        stop_count_flag = False
        for i in range(max_step):
            feed_dict = {user_ids: dtrain[:, 0],
                         item_ids: dtrain[:, 1],
                         ys: np.float32(dtrain[:, 2])}
            sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss], feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1], ys: np.float32(dtrain[:, 2])})
            rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
            print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (i + 1, rmse_train, rmse_test, loss_v, target_loss_v))
            if rmse_test>pre_rmse_test:
                stop_count += 1
                if stop_count==max_stop_count:
                    stop_count_flag = True
                    break
            pre_rmse_test = rmse_test

        return sess.run(item_embeddings)


def clean_data(comment: str):
    comment = comment.lower()
    comment = comment.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').replace('*', ' ') \
        .replace('-', '').replace('......', ' ').replace('...', ' ').replace('?', ' ').replace(':', ' ') \
        .replace(';', ' ').replace(',', ' ').replace('.', ' ').replace('!', ' ').replace("/", ' ') \
        .replace('" ', ' ').replace("' ", ' ').replace(' "', ' ').replace(" '", ' ').replace("=", ' ') \
        .replace('  ', ' ').replace('  ', ' ')
    return comment


def sorted_dict_keys(old_dict):
    n_dict = {}
    keys = sorted(old_dict.keys())
    for key in keys:
        n_dict[key] = old_dict[key]
    return n_dict


def restore_data(d_name: str, f_name: str, save_root, positive_ratings: int, get_true_scale=False):
    old_user2new = {}
    old_item2new = {}
    new_data = []
    ne_datas = []
    user_items = {}
    nega_user_items = {}
    nega_user_ratings = {}
    item_reviews = {}

    current_u_index = 0
    current_i_index = 0
    rating_counts = 0
    user_list = []
    item_list = []
    valid_rating = 0
    with open('./Amazon/' + d_name + '/' + f_name, 'r', encoding="utf-8") as f:
        while True:
            temp_str = f.readline()
            if temp_str:
                temp_dict = eval(temp_str)
                o_user = temp_dict['reviewerID']
                o_item = temp_dict['asin']
                if get_true_scale:
                    if o_user not in user_list:
                        user_list.append(o_user)
                    if o_item not in item_list:
                        item_list.append(o_item)
                rating = float(temp_dict['overall'])
                rating_counts += 1
                if rating >= positive_ratings:
                    valid_rating += 1
                    try:
                        n_user = old_user2new[o_user]
                        # print(n_user)
                    except KeyError:
                        n_user = current_u_index
                        old_user2new[o_user] = current_u_index
                        current_u_index += 1
                    try:
                        n_item = old_item2new[o_item]
                        # print(n_item)
                    except KeyError:
                        n_item = current_i_index
                        old_item2new[o_item] = current_i_index
                        current_i_index += 1

                    times = int(temp_dict['unixReviewTime'])
                    try:
                        reviews = clean_data(str(temp_dict['reviewText']))
                        reviews += clean_data(str(temp_dict['summary']))
                    except KeyError:
                        try:
                            reviews = clean_data(str(temp_dict['summary']))
                        except KeyError:
                            continue
                    try:
                        reviews_str = item_reviews[n_item]
                        item_reviews[n_item] = reviews_str + ',' + reviews
                    except KeyError:
                        item_reviews[n_item] = reviews
                    new_data.append((n_user, n_item, rating, times))
                else:
                    ne_datas.append((o_user, o_item, rating))
            else:
                break
    new_data = sorted(new_data, key=lambda x: (x[0], x[3]))  # Sort data by user_id and time
    with open(save_root + '/n_rating.txt', 'w', encoding="utf-8") as f:
        t_str = str(new_data)
        t_str = t_str[:len(t_str) - 2].replace('[', '').replace(']', '').replace('(', '').replace(' ', '') \
            .replace('),', '\n')
        f.write(t_str)
    for t_d in new_data:
        try:
            user_items[t_d[0]] = user_items[t_d[0]] + ',' + str(t_d[1]) + ':' + str(t_d[2])
        except KeyError:
            user_items[t_d[0]] = str(t_d[1]) + ':' + str(t_d[2])
    with open(save_root + '/user_items.txt', 'w', encoding="utf-8") as f:
        f.write(str(user_items))

    for ne_data in ne_datas:
        try:
            new_user_id = old_user2new[ne_data[0]]
        except KeyError:
            continue
        try:
            new_item_id = old_item2new[ne_data[1]]
        except KeyError:
            continue
        try:
            nega_user_items[new_user_id] = nega_user_items[new_user_id] + ',' + str(new_item_id)
            nega_user_ratings[new_user_id] = nega_user_ratings[new_user_id] + ',' + str(new_item_id) + ':' + str(ne_data[2])
        except KeyError:
            nega_user_items[new_user_id] = str(new_item_id)
            nega_user_ratings[new_user_id] = str(new_item_id) + ':' + str(ne_data[2])
    with open(save_root + '/nega_user_items.txt', 'w', encoding="utf-8") as f:
        f.write(str(nega_user_items))
    with open(save_root + '/nega_user_ratings.txt', 'w', encoding="utf-8") as f:
        f.write(str(nega_user_ratings))

    with open(save_root + '/old_user2new.txt', 'w', encoding="utf-8") as f:
        f.write(str(old_user2new))
    with open(save_root + '/old_item2new.txt', 'w', encoding="utf-8") as f:
        f.write(str(old_item2new))
    if get_true_scale:
        print('user_counts:', len(user_list))
        print('item_counts:', len(item_list))
    print('valid_user:', current_u_index)
    print('valid_item:', current_i_index)
    print('rating_counts', rating_counts)
    print('valid_rating:', valid_rating)
    return new_data, user_items, item_reviews, old_user2new, old_item2new, current_u_index, current_i_index, nega_user_ratings


def get_descriptions(d_name: str, old_item2new):
    item_descriptions = {}
    count = 0
    des_path = './Amazon/' + d_name + '/meta_' + d_name + '.json'
    with open(des_path, 'r', encoding="utf-8") as f:
        while True:
            temp_str = f.readline()
            if temp_str:
                temp_dict = eval(temp_str)
                t_asin = temp_dict['asin']
                try:
                    n_item = old_item2new[t_asin]
                    t_descriptions = clean_data(str(temp_dict['description']))
                    t_descriptions += clean_data(str(temp_dict['categories']))
                except KeyError:
                    try:
                        n_item = old_item2new[t_asin]
                        t_descriptions = clean_data(str(temp_dict['categories']))
                    except KeyError:
                        continue
                try:
                    descriptions_str = item_descriptions[n_item]
                    item_descriptions[n_item] = descriptions_str + ',' + t_descriptions
                except KeyError:
                    item_descriptions[n_item] = t_descriptions
            else:
                break
            count += 1
    print('Get ', count, ' ' + d_name + ' descriptions.')
    return item_descriptions


def get_train_test(user_items_dict, select_percent: float, more_than_one: bool):
    train_user_items = {}
    test_user_items = {}
    for i in range(0, len(user_items_dict)):
        user_items = user_items_dict[i].split(',')
        select_num = int(np.round(len(user_items) * select_percent))
        t_s_num = len(user_items) - select_num
        if more_than_one and len(user_items) > 1:
            if t_s_num == 0:
                t_s_num = 1
                select_num = len(user_items) - t_s_num
        if select_num < len(user_items):
            test_user_items[i] = user_items[len(user_items) - t_s_num:]
            test_user_items[i] = str(test_user_items[i]). \
                replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
        train_user_items[i] = user_items[:len(user_items) - t_s_num]
        train_user_items[i] = str(train_user_items[i]). \
            replace('[', '').replace(']', '').replace("'", '').replace(' ', '')
    print('Get train and test split by time-line.')
    return train_user_items, test_user_items


def get_stop_word(stop_word_path: str):
    with open(stop_word_path) as stop_word_file:
        i_stop_words_list = (stop_word_file.read()).split()
    return i_stop_words_list


def get_glove_dict(glove_dict_path: str):
    with open(glove_dict_path, 'r', encoding="utf-8") as glove_file:
        in_glove_dict = {}
        t_list = []
        for line in glove_file.readlines():
            t_list = line.split()
            if len(t_list) > 1:
                tt_list = []
                for number in t_list[1:]:
                    tt_list.append(float(number))
                in_glove_dict[t_list[0]] = np.array(tt_list)
    return in_glove_dict


def get_vector(item_texts_dict: dict, in_glove_dict, embedding_size, stop_word_list: list,
               item_num: int, save_path):
    item_embeddings = np.zeros((item_num, embedding_size))
    t_count = 0
    # print(item_num)
    for i in range(item_num):
        item_emb = np.zeros(embedding_size)
        try:
            word_str = str(item_texts_dict[i])
            word_list = word_str.split(" ")
            # print(word_list)
            t_div = 1
            for word in word_list:
                if word not in stop_word_list:
                    try:
                        word_glove_vector = in_glove_dict[word]
                        item_emb = item_emb + word_glove_vector
                    except KeyError:
                        continue
                    t_div += 1
                else:
                    continue
            # print(t_div, item_emb, item_emb / t_div)
            item_embeddings[i] = item_emb / t_div  # normalise
            t_count += 1
        except KeyError:
            continue
    if save_path != '':
        np.save(save_path, item_embeddings)
    print(item_embeddings.shape, "Get embeddings.")
    return item_embeddings


def get_user_vector(train_user_movies: dict, u_num: int, i_movie_vectors, embedding_size,
                    save_str, is_weight=False):
    user_embeddings = np.zeros((u_num, embedding_size))
    user_emb = np.zeros(embedding_size)
    for user in train_user_movies.keys():
        item_list = train_user_movies[user].split(',')
        for item in item_list:
            items = item.split(':')
            if is_weight:
                user_emb = user_emb + i_movie_vectors[int(items[0])] * (float(items[1])/5)
            else:
                user_emb = user_emb + i_movie_vectors[int(items[0])]
        user_embeddings[user] = user_emb / len(item_list)  # normalise
        # print(user_embeddings[user])
        user_emb = np.zeros(embedding_size)
    print('All user embeddings done.')
    # print(user_embeddings.shape)
    if save_str != '':
        np.save(save_str + "_embeddings.npy", user_embeddings)
    return user_embeddings


def get_user_cluster(i_user_data, cluster_num: int, save_root, plot=False):
    estimator = KMeans(n_clusters=cluster_num, max_iter=500)  # Construct cluster
    estimator.fit(i_user_data)
    label_predict = estimator.labels_  # Get cluster labels
    class_center = estimator.cluster_centers_
    # Save K-means results
    # print(len(label_predict))
    with open(save_root + "/user_label_predict.txt", 'w+') as l_file:
        str_label = str(list(label_predict)).replace('[', '').replace(']', '').replace(' ', '')
        l_file.write(str_label)
    # Calculate the distance between the classification center
    # select the one furthest from the cluster center, and save it
    # Calculate the distance between class pairs
    dis_pair_list = list()
    i = 0
    while i < class_center.shape[0] - 1:
        j = i + 1
        while j < class_center.shape[0]:
            dis_pair_list.append([calculate_distance(class_center[i], class_center[j]), [i, j]])
            j += 1
        i += 1
    # print(dis_pair_list)
    # Get the maximum distance for each center
    large_dis_center = list()
    i = 0
    while i < class_center.shape[0]:
        large_pair = None
        max_dis = -1
        for one_pair in dis_pair_list:
            if i in one_pair[1]:
                if one_pair[0] > max_dis:
                    max_dis = one_pair[0]
                    large_pair = one_pair[1]
        large_dis_center.append(large_pair)
        i += 1
    # print(np.array(large_dis_center))
    max_dis_dict = {}
    i = 0
    for t_cluster in large_dis_center:
        if i == t_cluster[0]:
            max_dis_dict[i] = t_cluster[1]
        else:
            max_dis_dict[i] = t_cluster[0]
        i += 1
    # print(class_center)
    np.save(save_root + '/class_center.npy', class_center)
    # print(np.array(large_dis_center))
    with open(save_root + '/max_dis_pair.txt', 'w', encoding="utf-8") as f:
        f.write(str(max_dis_dict))
    if plot:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        t_pca = PCA(n_components=2)
        low_dim_embs = t_pca.fit_transform(i_user_data)
        # Draw K-means results
        for i in range(0, cluster_num):
            x = low_dim_embs[label_predict == i]
            plt.scatter(x[:, 0], x[:, 1], label='cluster' + str(i))
        plt.legend(loc='best')
        plt.show()
        plt.close()
    return label_predict, max_dis_dict


def prepare_data(user_label_list, train_user_items, test_user_items, max_dis_dict, nega_user_ratings, num_item, save_root):
    user_label_num = len(set(user_label_list)) - 1

    train_user_items_dict = {}
    train_user_items_rating_dict = {}
    for u_id in train_user_items.keys():
        train_items_list = []
        train_items_rating_dict = {}
        if train_user_items[u_id] != '':
            for item in train_user_items[u_id].split(','):
                items = item.split(':')
                train_items_list.append(int(items[0]))
                train_items_rating_dict[int(items[0])] = float(items[1])
        train_user_items_dict[u_id] = train_items_list.copy()
        train_user_items_rating_dict[u_id] = train_items_rating_dict.copy()
    # print(train_user_items_rating_dict)
    # print(train_user_items_dict)
    with open(save_root + '/train_user_items_dict.txt', 'w') as train_ui_dict:
        train_ui_dict.write(str(train_user_items_dict))
    with open(save_root + '/train_user_items_rating_dict.txt', 'w') as train_uir_dict:
        train_uir_dict.write(str(train_user_items_rating_dict))

    nega_user_items_rating_dict = {}
    for u_id in nega_user_ratings.keys():
        nega_items_rating_dict = {}
        if nega_user_ratings[u_id] != '':
            for item in nega_user_ratings[u_id].split(','):
                items = item.split(':')
                nega_items_rating_dict[int(items[0])] = float(items[1])
        nega_user_items_rating_dict[u_id] = nega_items_rating_dict.copy()
    # print(nega_user_items_rating_dict[0].keys())
    with open(save_root + '/nega_user_items_rating_dict.txt', 'w') as nega_uir_dict:
        nega_uir_dict.write(str(nega_user_items_rating_dict))

    cluster_items = []  # int, record of items for per cluster
    cluster_users = []  # int, record of users for per cluster
    for i in range(0, user_label_num + 1):  # Initialization
        cluster_users.append(list())
        cluster_items.append(set())
    for user in train_user_items_dict.keys():
        t_label = int(user_label_list[user])
        cluster_users[t_label].append(user)
        for item in train_user_items_dict[user]:
            cluster_items[t_label].add(item)
    with open(save_root + '/cluster_users.txt', 'w') as c_us:
        c_us.write(str({'cluster_users': cluster_users}))

    # Gets the list of classes that appear in the current class but not the farthest from the current class
    supp_nega_cluster_items = {}
    for user_cluster in range(user_label_num + 1):
        train_cluster_items_list = cluster_items[user_cluster].copy()
        max_dis_cluster = max_dis_dict[user_cluster]
        # print(len(cluster_items), len(cluster_items[int_cluster]))
        supp_nega_cluster_items[user_cluster] = cluster_items[max_dis_cluster].copy()
        for train_item in train_cluster_items_list:
            if train_item in cluster_items[max_dis_cluster]:
                supp_nega_cluster_items[user_cluster].remove(train_item)
    with open(save_root + '/supp_nega_cluster_items.txt', 'w') as nega_ci_file:
        nega_ci_file.write(str(supp_nega_cluster_items))

    # Get all test samples in advance for all methods
    test_dict = {}
    test_user_items_rating_dict = {}
    for user_id in test_user_items.keys():
        user_cluster = int(user_label_list[user_id])
        test_items = set()
        test_items_rating_dict = {}
        for te_i_r in test_user_items[user_id].split(','):
            items = te_i_r.split(':')
            test_items.add(int(items[0]))
            test_items_rating_dict[int(items[0])] = float(items[1])
        test_user_items_rating_dict[user_id] = test_items_rating_dict.copy()
        test_dict[str(user_id) + '_p'] = list(test_items)

        num_nega = int(len(test_items) / test_percent)
        if num_nega < max_k * 2 - len(test_items):
            num_nega = max_k * 2 - len(test_items)
        # Avoid dead circulation
        if num_nega > num_item - len(train_user_items_dict[user_id]):
            num_nega = num_item - len(train_user_items_dict[user_id])

        if num_nega >= len(supp_nega_cluster_items[user_cluster]):
            negative_list = list(supp_nega_cluster_items[user_cluster].copy())
            while True:
                if len(negative_list) == num_nega:
                    break
                one_negative = np.random.randint(num_item)
                if one_negative not in negative_list:
                    negative_list.append(one_negative)
        else:
            negative_list = constant_sample(supp_nega_cluster_items[user_cluster], num_nega)

        test_dict[str(user_id) + '_n'] = negative_list
    with open(save_root + '/test_dict.txt', 'w') as test_f:
        test_f.write(str(test_dict))
    # print(test_user_items_rating_dict[0].keys())
    with open(save_root + '/test_user_items_rating_dict.txt', 'w') as test_uir_dict:
        test_uir_dict.write(str(test_user_items_rating_dict))


def calculate_distance(vector1, vector2):
    # theta = 0.0000001
    # dist = np.sum(vector1*vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2) + theta)  # cosin
    dist = np.sum(np.square(vector1 - vector2))  # Euclidean distance
    return dist


# Matrix data is limited to 0-1 per row
def normalize_data(matrix_data):
    for ii in range(matrix_data.shape[0]):
        matrix_data[ii] = \
            (matrix_data[ii] - matrix_data[ii].min()) / (matrix_data[ii].max() - matrix_data[ii].min() + 0.0000001)


if __name__ == '__main__':
    start_time = time.process_time()  # Starting time

    # 'Digital_Music' 'Beauty' 'Clothing_Shoes_and_Jewelry'
    data_name = 'Digital_Music'
    # glove mf
    reduce_dim_method = 'glove'
    emb_size = 100  # experimental set: 'Digital_Music：100' 'Beauty：100' 'Clothing_Shoes_and_Jewelry：200'
    alpha = 0.5
    cluster_num = 10  # experimental set: 'Digital_Music：10' 'Beauty：10' 'Clothing_Shoes_and_Jewelry：15'
    test_percent = 0.1
    max_k = 20
    file_name = data_name + '_5.json'
    meta_path = './Amazon/' + data_name + '/meta_' + data_name + '.json'  # descriptions
    save_root = './Amazon/' + data_name + '/' + reduce_dim_method + '/' + str(emb_size)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    o_new_data, o_user_items, o_item_reviews, o_old_user2new, o_old_item2new, user_num, item_num, nega_user_ratings = \
        restore_data(data_name, file_name, save_root, positive_ratings=3, get_true_scale=False)
    o_train_user_items, o_test_user_items = get_train_test(o_user_items, select_percent=0.9, more_than_one=True)

    if reduce_dim_method == 'glove':
        o_item_descriptions = get_descriptions(data_name, o_old_item2new)
        # get vectors
        o_stop_word_list = get_stop_word(stop_word_path='./resource/stop_words.txt')
        glove_dict = get_glove_dict(glove_dict_path='./resource/glove/glove.6B.' + str(emb_size) + 'd.txt')
        description_vectors = get_vector(o_item_descriptions, glove_dict, emb_size,
                                         o_stop_word_list, item_num, "")
        review_vectors = get_vector(o_item_reviews, glove_dict, emb_size,
                                    o_stop_word_list, item_num, "")
        item_vectors = alpha * description_vectors + (1 - alpha) * review_vectors
    else:
        import tensorflow as tf
        # get vectors
        if reduce_dim_method == 'mf':
            item_vectors = mf_with_bias([user_num, item_num], emb_size, o_new_data)
        else:
            item_vectors = None
            print('Not support this method.')
            exit(-1)
        normalize_data(item_vectors)
        # Make it consistent with the distribution range of text vector data
        item_vectors = item_vectors * 2 - 1
    np.save(save_root + '/' + data_name + '_embeddings.npy', item_vectors)
    user_vectors = get_user_vector(o_train_user_items, user_num, item_vectors, emb_size,
                                   save_str='', is_weight=True)
    # Classify users according to user vector
    user_lables, max_dis_dict = get_user_cluster(user_vectors, cluster_num, save_root)

    # Data preparation
    prepare_data(user_lables, o_train_user_items, o_test_user_items, max_dis_dict, nega_user_ratings, item_num, save_root)

    # End time
    end_time = time.process_time()
    print("Cost time is %f" % (end_time - start_time))
