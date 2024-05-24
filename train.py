import random
import numpy as np
from DF_model import DF,generator_model,discriminator_model
from keras.layers import Input, Lambda, Dot, Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
import keras.backend as K
from tensorflow.keras import utils
import keras
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import os
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from copy import deepcopy
from keras.utils.np_utils import to_categorical
from utils import *
np.random.seed(1)

alpha = 0.1
beta = 0.1
lam = 0.5
omiga = 0.5
local_lr = 0.001
global_lr = 0.001
batch_size = 128
emb_size = 128
local_epochs = 10
global_epochs = 30
fed_epoch = 5
trace_len = 5000
client_num = 4
seed_class = 10
seed_num = 10
alpha_value = float(alpha)
has_seed = True

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

binary_loss = tf.keras.losses.BinaryCrossentropy()

def local_loss(emb_a, emb_p, emb_n):
    positive_sim = Dot(axes=-1)([emb_a, emb_p])
    negative_sim = Dot(axes=-1)([emb_a, emb_n])
    triplet_loss = K.maximum(negative_sim - positive_sim + alpha_value, 0.0)
    BDC_loss = Dot(axes=-1)([emb_a, emb_p])
    return K.mean(triplet_loss)

def global_loss(wastein_A, wastein_B):
    pos = Dot(axes=-1)([wastein_A, wastein_B])
    loss = K.maximum(omiga - pos, 0.0)
    return K.mean(loss)

def local_init(data_path, seed_data, seed_label):
    # 读取数据集
    npzfile = np.load(data_path, allow_pickle=True, encoding='latin1')
    ori_data = npzfile['data']
    ori_label = npzfile["labels"]
    npzfile.close()

    all_traces = ori_data
    label = ori_label
    if has_seed:
        for s in range(len(seed_data)):
            all_traces = np.vstack((all_traces, seed_data[s]))
            label = np.hstack((label, seed_label[s]))

    label_list = list(set(label))
    label_list.sort(key=list(label).index)
    num_classes = len(label_list)

    # 构建跟踪和类之间的映射
    classid_to_ids = {}
    for i in range(len(all_traces)):
        key = label_list.index(label[i])
        if key not in classid_to_ids:
            classid_to_ids[key] = []
        classid_to_ids[key].append(i)
    id_to_classid = {v: c for c, traces in classid_to_ids.items() for v in traces}

    all_traces = np.vstack((all_traces))
    all_traces = all_traces[:, :, np.newaxis]

    Xa_train, Xp_train = build_positive_pairs(range(0, num_classes), classid_to_ids)

    return id_to_classid, all_traces, Xa_train, Xp_train

def global_int(seed_path):
    seed_data = []
    seed_label = []
    # 建立server训练的seed数据集
    global_data = []
    global_label = []
    for c in range(client_num):
        npzfile = np.load(seed_path[c], allow_pickle=True, encoding='latin1')
        data = npzfile['data']
        label = npzfile["labels"]
        seed_data.append(data)
        seed_label.append(label)
        npzfile.close()
        if c == 0:
            global_data = data
            global_label = label
        else:
            global_data = np.vstack((global_data, data))
            global_label = np.hstack((global_label, label))

    # 为服务器生成采样器
    label_list = list(set(global_label))
    label_list.sort(key=list(global_label).index)
    num_classes = len(label_list)
    #tr = int(global_data.shape[0] / num_classes)

    # 构建跟踪和类之间的映射
    classid_to_ids = {}
    for i in range(len(global_data)):
        key = label_list.index(global_label[i])
        if key not in classid_to_ids:
            classid_to_ids[key] = []
        classid_to_ids[key].append(i)
    id_to_classid = {v: c for c, traces in classid_to_ids.items() for v in traces}

    all_traces = np.vstack((global_data))
    all_traces = all_traces[:, :, np.newaxis]

    Xa_train, Xp_train = build_positive_pairs(range(0, num_classes), classid_to_ids)

    all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
    gen_hard = SemiHardTripletGenerator(id_to_classid, Xa_train, Xp_train, int(batch_size/2), all_traces,
                                               all_traces_train_idx, None)

    return seed_data, seed_label, gen_hard, all_traces, classid_to_ids

def loacl_train(shared_model, global_gen_hard, id_to_classid, all_traces, Xa_train, Xp_train, client, epoch,file_name):

    all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
    #optimizer = RMSprop(learning_rate=local_lr, decay=0)
    optimizer = SGD(learning_rate=local_lr, decay=1e-6, momentum=0.9, nesterov=True)
    steps = Xa_train.shape[0] // batch_size
    # 在第一阶段，无hard triplets
    if epoch == 0:
        gen_hard = SemiHardTripletGenerator(id_to_classid, Xa_train, Xp_train, batch_size, all_traces,
                                            all_traces_train_idx, None)
        #global_gen_hard.similarities = None
    else:
        gen_hard = SemiHardTripletGenerator(id_to_classid, Xa_train, Xp_train, batch_size, all_traces,
                                            all_traces_train_idx, shared_model)
        #global_gen_hard.similarities = build_similarities(shared_model, global_gen_hard.traces)
    print("-----------------client"+str(client)+"-------------------")
    for e in range(local_epochs):
        train_progbar = utils.Progbar(steps)
        print('\nEpoch {}/{}'.format(e + 1, local_epochs))
        # 按批次训练
        for i in range(steps):
            train_a, train_p, train_n = gen_hard.next_train()
            global_a, global_p, global_n = global_gen_hard.next_train()
            train_a = np.append(train_a, global_a, axis=0)
            train_p = np.append(train_p, global_p, axis=0)
            train_n = np.append(train_n, global_n, axis=0)
            with tf.GradientTape() as tape:
                _, emb_a = shared_model(train_a)
                _, emb_p = shared_model(train_p)
                _, emb_n = shared_model(train_n)
                loss = local_loss(emb_a, emb_p, emb_n)
            grads = tape.gradient(loss, shared_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, shared_model.trainable_variables))
            train_progbar.update(i + 1, [('local loss', loss)])
        gen_hard = SemiHardTripletGenerator(id_to_classid, Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, shared_model)
        #global_gen_hard.similarities = build_similarities(shared_model, global_gen_hard.traces)
    shared_model.save_weights(file_name+"model/"+"local_client"+str(client)+"_epoch"+str(epoch)+".h5")
    return shared_model.get_weights()

def global_train(model, local_weights, global_data, seed_class_dict, G_model, D_model, G_weights, D_weights):
    print("-----------------global-------------------")

    new_weight = local_weights.copy()
    optimizer = SGD(learning_rate=global_lr, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_G = SGD(learning_rate=global_lr, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer_D = SGD(learning_rate=global_lr, decay=1e-6, momentum=0.9, nesterov=True)

    global_data = tf.keras.backend.concatenate(global_data, axis=0)
    global_data = np.reshape(global_data,[global_data.shape[0],global_data.shape[1],1])
    for e in range(global_epochs*100):
        train_progbar_G = utils.Progbar(global_epochs)
        train_progbar_D = utils.Progbar(global_epochs)
        train_acc = utils.Progbar(global_epochs)
        print('\nEpoch {}/{}'.format(e + 1, global_epochs))
        with tf.GradientTape(persistent=True) as tape:
            feature_list = []  # 保存客户端的特征向量
            for i in range(client_num):
                model.set_weights(new_weight[i])
                steps = int(global_data.shape[0] / 40)
                feature = []
                for ss in range(steps):
                    temp, _ = model(global_data[ss*40:(ss+1)*40], training=True)
                    feature.append(temp)
                    del temp
                feature = tf.keras.backend.concatenate(feature, axis=0)
                class_feature_list = []  # 保存每个类的特征向量
                for key in seed_class_dict:
                    index = np.array(seed_class_dict[key])
                    class_feature_list.append(tf.gather(feature, index))
                feature_list.append(class_feature_list)
                del feature, class_feature_list, index
            vistual_feature, vistual_label = get_vs_feature(feature_list)  # 计算均值和方差，生成虚拟特征向量
            vistual_feature = tf.keras.backend.concatenate(vistual_feature, axis=0)

            G_losses = []
            D_losses = []
            for i in range(client_num):  # 每个客户端的特征和虚拟特征尽可能靠近
                feature_c = tf.keras.backend.concatenate(feature_list[i], axis=0)
                G_model.set_weights(G_weights[i])
                D_model.set_weights(D_weights[i])

                steps = int(global_data.shape[0] / 40)
                wastein = []
                for ss in range(steps):
                    temp = G_model(feature_c[ss*40: (ss+1)*40], training=True)
                    wastein.append(temp)
                    del temp
                wastein = tf.keras.backend.concatenate(wastein, axis=0)
                
                steps = int(wastein.shape[0] / 40)
                dism_c = []
                for ss in range(steps):
                    temp = D_model(wastein[ss * 40: (ss + 1) * 40], training=True)
                    dism_c.append(temp)
                    del temp
                dism_fake = tf.keras.backend.concatenate(dism_c, axis=0)
                
                steps = int(vistual_feature.shape[0] / 40)
                dism_vis = []
                for ss in range(steps):
                    temp = D_model(vistual_feature[ss * 40: (ss + 1) * 40], training=True)
                    dism_vis.append(temp)
                    del temp
                dism_real = tf.keras.backend.concatenate(dism_vis, axis=0)

                G_loss = tf.reduce_mean(tf.scalar_mul(-1, dism_fake))
                D_loss = tf.reduce_mean(tf.scalar_mul(-1, dism_real)) + tf.reduce_mean(dism_fake)

                G_losses.append(G_loss)
                D_losses.append(D_loss)

        for i in range(len(G_losses)):
            model.set_weights(local_weights[i])
            G_model.set_weights(G_weights[i])
            D_model.set_weights(D_weights[i])

            if e<4 or 10<e<14 or 20<e<24:
                grads_D = tape.gradient(D_losses[i], D_model.trainable_variables)
                optimizer_D.apply_gradients(zip(grads_D, D_model.trainable_variables))
            else:
                grads = tape.gradient(G_losses[i], model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                grads_G = tape.gradient(G_losses[i], G_model.trainable_variables)
                optimizer_G.apply_gradients(zip(grads_G, G_model.trainable_variables))

            new_weight[i] = model.get_weights()
            G_weights[i] = G_model.get_weights()

            clip_d_op = [var.assign(tf.clip_by_value(var, -0.1, 0.1)) for var in D_model.trainable_variables]
            D_weights[i] = D_model.get_weights()

        train_progbar_G.update(e + 1, [('generator loss', np.mean(G_losses[2]))])
        train_progbar_D.update(e + 1, [('discriminator loss', np.mean(D_losses[2]))])
        del vistual_feature, vistual_label, feature_list, dism_c, dism_vis, wastein
    return new_weight

def test_model(model, fine_path, test_path, client, local, epoch,file_name): #仅测试单个客户端数据
    result = []
    for i in range(len(test_path)):
        npzfile = np.load(fine_path[i], allow_pickle=True, encoding='latin1')
        fine_data = npzfile['data']
        fine_label = npzfile["labels"]
        npzfile.close()

        npzfile = np.load(test_path[i], allow_pickle=True, encoding='latin1')
        test_data = npzfile['data']
        test_label = npzfile["labels"]
        npzfile.close()

        n_shot = [1, 5, 10, 15, 20]
        class_num = len(set(fine_label))
        for shot in n_shot:
            if shot == 20:
                x_train, y_train = fine_data, fine_label
            else:
                x_train, _, y_train, _ = train_test_split(fine_data, fine_label, train_size=shot * class_num,
                                                          random_state=1,stratify=fine_label)
            _,emb_train = model.predict(x_train)

            knn = KNeighborsClassifier(n_neighbors=shot, weights='distance', p=2, metric='cosine', algorithm='brute')
            knn.fit(emb_train, y_train)

            _,emb_test = model.predict(test_data)
            acc_knn_top1 = accuracy_score(test_label, knn.predict(emb_test))
            result.append("{}->{} {}shot : acc = {}".format(client, i, shot, acc_knn_top1))
        result.append("\n")

    with open(file_name + "log.txt", "a") as f:
        f.write("---------------------epoch:{} {}:{}--------------------\n".format(epoch, local, client))
        for res in result:
            f.write(res + "\n")
    f.close()

def test_model_other(model, fine_path, test_path, client, local, epoch,file_name): #测试混合的

    temp_fine = fine_path.copy()
    temp_test = test_path.copy()
    temp_fine.pop(client)
    temp_test.pop(client)

    fine_data = []
    fine_label = []
    test_data = []
    test_label = []

    for c in range(len(temp_fine)):
        npzfile = np.load(temp_fine[c], allow_pickle=True, encoding='latin1')
        data = npzfile['data']
        label = npzfile["labels"]
        npzfile.close()

        if c == 0:
            fine_data = data
            fine_label = label
        else:
            fine_data = np.vstack((fine_data, data))
            fine_label = np.hstack((fine_label, label))

        npzfile = np.load(temp_test[c], allow_pickle=True, encoding='latin1')
        data = npzfile['data']
        label = npzfile["labels"]
        npzfile.close()

        if c == 0:
            test_data = data
            test_label = label
        else:
            test_data = np.vstack((test_data, data))
            test_label = np.hstack((test_label, label))

    result = []
    n_shot = [1, 5, 10, 15, 20]
    class_num = len(set(fine_label))
    for shot in n_shot:
        if shot == 20:
            x_train, y_train = fine_data, fine_label
        else:
            x_train, _, y_train, _ = train_test_split(fine_data, fine_label, train_size=shot * class_num,
                                                      random_state=1, stratify=fine_label)
        _,emb_train = model.predict(x_train)


        knn = KNeighborsClassifier(n_neighbors=shot, weights='distance', p=2, metric='cosine', algorithm='brute')
        knn.fit(emb_train, y_train)

        _,emb_test = model.predict(test_data)
        acc_knn_top1 = accuracy_score(test_label, knn.predict(emb_test))
        result.append("{} {}shot : acc = {}".format(client, shot, acc_knn_top1))
    result.append("\n")

    with open(file_name + "log.txt", "a") as f:
        f.write("---------------------mev epoch:{} {}:{}--------------------\n".format(epoch, local, client))
        for res in result:
            f.write(res + "\n")
    f.close()


def my_train():
    # 建立一个初始化模型
    shared_model = DF((trace_len, 1), emb_size)
    print(shared_model.summary())

    G_model = generator_model((128,), 128)
    print(G_model.summary())

    D_model = discriminator_model((128,), 2)
    print(D_model.summary())

    data_path = ["./dataset/ori/AWF_100w_2500tr.npz", "./dataset/ori/Wang_100w_90tr.npz",
                 "./dataset/ori/DS19_100w_100tr.npz", "./dataset/ori/NoDef_95w_800tr.npz"]

    # 写日志
    file_time = str(datetime.now())[:19]
    file_time = file_time.replace(':', '-')
    file_name = "./result/ours/" + file_time + "/"
    os.mkdir(file_name)
    os.mkdir(file_name + "model")
    os.mkdir(file_name + "dataset")
    train_data_path = []
    fine_data_path = []
    test_data_path = []
    seed_data_path = []
    for j in range(len(data_path)):
        npzfile = np.load(data_path[j], allow_pickle=True, encoding='latin1')
        data = npzfile['data']
        label = npzfile["labels"]
        temp = list(label)
        label = np.array(list(str(i) + "_" + str(j) for i in temp))

        label_num = len(set(label))

        x_train, _, y_train, _ = train_test_split(data, label, train_size=25 * label_num, random_state=1, stratify=label)
        x_fine, x_test, y_fine, y_test = train_test_split(data, label, train_size=20 * label_num, test_size=70 * label_num, random_state=1, stratify=label)

        seed_dict = {}
        seed_index = []
        for index in range(y_train.shape[0]):
            if y_train[index] not in seed_dict:
                if len(seed_dict) < seed_class:
                    seed_dict[y_train[index]] = 1
                    seed_index.append(index)
            else:
                if seed_dict[y_train[index]] < seed_num:
                    seed_dict[y_train[index]] += 1
                    seed_index.append(index)

        x_seed = x_train[seed_index]
        y_seed = y_train[seed_index]

        start = data_path[j].rfind("/")
        end = data_path[j].find("_")
        train_name = file_name + "dataset/" + data_path[j][start+1:end] + "_train.npz"
        fine_name = file_name + "dataset/" + data_path[j][start + 1:end] + "_fine.npz"
        test_name = file_name + "dataset/" + data_path[j][start+1:end] + "_test.npz"
        seed_name = file_name + "dataset/" + data_path[j][start + 1:end] + "_seed.npz"

        np.savez_compressed(train_name, data=x_train, labels=y_train)
        np.savez_compressed(fine_name, data=x_fine, labels=y_fine)
        np.savez_compressed(test_name, data=x_test, labels=y_test)
        np.savez_compressed(seed_name, data=x_seed, labels=y_seed)

        train_data_path.append(train_name)
        fine_data_path.append(fine_name)
        test_data_path.append(test_name)
        seed_data_path.append(seed_name)

    with open(file_name+"log.txt", "w") as f:
        f.write("train_data:" + str(train_data_path) + "\n")
        f.write("fine_data:" + str(fine_data_path) + "\n")
        f.write("test_data:" + str(test_data_path) + "\n")
        f.write("seed_data:" + str(seed_data_path) + "\n")
        f.write("seed_class:" + str(seed_class) + "\n")
        f.write("local epochs:" + str(local_epochs) + "\n")
        f.write("global epochs:" + str(global_epochs) + "\n")
        f.write("local lr:" + str(local_lr) + "\n")
        f.write("global lr:" + str(global_lr) + "\n")
        f.write("fed epochs:" + str(fed_epoch) + "\n")
        f.write("alpha:{}  beta:{}  lam:{}  omiga:{}\n".format(alpha, beta, lam, omiga))
    f.close()

    seed_data, seed_label, global_gen_hard, global_traces, seed_class_dict = global_int(seed_data_path)

    # 保存每个客户端的参数
    id_to_classid = []
    all_traces = []
    Xa_train = []
    Xp_train = []
    local_weights = []
    G_weights = []
    D_weights = []
    for c in range(client_num):
        # 筛选出其他客户端的种子
        temp_seed_data = seed_data.copy()
        temp_seed_label = seed_label.copy()
        temp_seed_data.pop(c)
        temp_seed_label.pop(c)
        # 初始化本地模型的数据和参数
        index, traces, Xa, Xp = local_init(train_data_path[c], temp_seed_data, temp_seed_label)
        id_to_classid.append(index)
        all_traces.append(traces)
        Xa_train.append(Xa)
        Xp_train.append(Xp)
        local_weights.append(shared_model.get_weights())
        G_weights.append(G_model.get_weights())
        D_weights.append(D_model.get_weights())

    # 联邦训练
    for e in range(fed_epoch):
        # 每个客户端分别训练,上传模型
        for c in range(client_num):
            shared_model.set_weights(local_weights[c])
            weights = loacl_train(shared_model, global_gen_hard, id_to_classid[c], all_traces[c], Xa_train[c], Xp_train[c], c, e, file_name)
            local_weights[c] = weights
            # 测试本地模型
            #test_model(shared_model, fine_data_path, test_data_path, c, "local", e, file_name)
            test_model_other(shared_model, fine_data_path, test_data_path, c, "local", e, file_name)
        # 服务器端训练
        new_weights = global_train(shared_model, local_weights, seed_data, seed_class_dict, G_model, D_model, G_weights, D_weights)
        #local_weights = new_weights
        for c in range(client_num):
            shared_model.set_weights(new_weights[c])
            for l in range(0, len(local_weights[c])):
                local_weights[c][l] = (1 - float(lam)) * local_weights[c][l] + float(lam) * new_weights[c][l]
            shared_model.set_weights(local_weights[c])
            test_model_other(shared_model, fine_data_path, test_data_path, c, "global", e, file_name)

my_train()
