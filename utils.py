import numpy as np
import random
import tensorflow as tf

def build_pos_pairs_for_id(classid,classid_to_ids):
    traces = classid_to_ids[classid]
    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs

def build_positive_pairs(class_id_range,classid_to_ids):
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id, classid_to_ids)
        for pair in pos:
            listX1 += [pair[0]]
            listX2 += [pair[1]]
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]

def build_similarities(conv, all_imgs):
    _, embs = conv.predict(all_imgs)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    return all_sims

def build_negatives(id_to_classid,anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
    #  如果没有计算相似点，则返回一个随机的负数
    if similarities is None:
        return random.sample(neg_imgs_idx,len(anc_idxs))
    final_neg = []
    # 对于每一对正例
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        sim = similarities[anc_idx, pos_idx]
        # 找出所有的 semi(hard)
        possible_ids = np.where((similarities[anc_idx] + 0.1) > sim)[0]
        possible_ids = list(set(neg_imgs_idx) & set(possible_ids))
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))

    return final_neg

class SemiHardTripletGenerator():
    def __init__(self, id_to_classid, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, conv):
        self.batch_size = batch_size
        self.traces = all_traces
        self.id_to_classid = id_to_classid
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.all_anchors = list(set(Xa_train))
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0

            # fill one batch
            traces_a = self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_n = build_negatives(self.id_to_classid, traces_a, traces_p, self.similarities, self.neg_traces_idx)
            return [self.traces[traces_a], self.traces[traces_p], self.traces[traces_n]]

def _cal_mean_cov(features):

    features = np.array(features)
    mean = np.mean(features, axis=0)

    # Note that bias=1 is set here, which is equivalent to dividing by N in formula (2) instead of N-1,
    # because when N=1, dividing by 0 will result in NaN.
    cov = np.cov(features.T, bias=1)
    return mean, cov

def cal_distributions(data):
    mean = []
    cov = []
    length = []

    for i in range(len(data)):
        f_mean, f_cov = _cal_mean_cov(data[i])
        mean.append(f_mean)
        cov.append(f_cov)
        length.append(data[i].shape[0])

    return mean, cov, length

def cal_global_gd(client_mean, client_cov, client_length):
    g_mean = []
    g_cov = []

    clients = list(client_mean.keys())

    for c in range(len(client_mean[clients[0]])):

        mean_c = np.zeros_like(client_mean[clients[0]][0])
        n_c = 0

        for k in clients:
            n_c += client_length[k][c]

        cov_ck = np.zeros_like(client_cov[clients[0]][0])
        mul_mean = np.zeros_like(client_cov[clients[0]][0])

        for k in clients:
            # local mean
            mean_ck = np.array(client_mean[k][c])
            # global mean
            mean_c += (client_length[k][c] / n_c) * mean_ck

            cov_ck += ((client_length[k][c] - 1) / (n_c - 1)) * np.array(client_cov[k][c])  # first term in equation (4)
            mul_mean = mul_mean + ((client_length[k][c]) / (n_c - 1)) * np.sum(np.dot(mean_ck.T, mean_ck))  # second term in equation (4)

        g_mean.append(mean_c)

        # global covariance
        cov_c = cov_ck + mul_mean - (n_c / (n_c - 1)) * np.dot(mean_c.T, mean_c)  # equation (4)

        g_cov.append(cov_c)

    return g_mean, g_cov

def get_vs_feature(data):
    client_mean = {}
    client_cov = {}
    client_length = {}

    for i in range(len(data)):
        mean, cov, length = cal_distributions(data[i])
        client_mean[i] = mean
        client_cov[i] = cov
        client_length[i] = length

    global_mean, global_cov = cal_global_gd(client_mean, client_cov, client_length)

    # Generate a set of Gc virtual features with ground truth label c from the Gaussian distribution.
    label = []
    num_vr = data[0][0].shape[0]
    vr_list = []
    for i in range(len(global_mean)):
        mean = np.squeeze(np.array(global_mean[i]))
        vr = np.random.multivariate_normal(mean, global_cov[i], num_vr)
        vr_list.append(vr)
        label.extend([i] * num_vr)

    return vr_list, label
