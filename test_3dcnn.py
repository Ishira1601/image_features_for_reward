import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from matplotlib import colors
from T3D_keras import densenet161_3D_DropOut, densenet121_3D_DropOut, xception_classifier, c3d_model, \
    c3d_model_feature

from tensorflow.keras.optimizers import Adam
import os

import datetime
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.axes as ax

from pandas.plotting import table
from upr import UPR

def get_file_paths(folders):
    from os import listdir
    from os.path import isfile, join

    file_paths = []
    for folder in folders:
        onlyfiles = [folder+"/"+f for f in listdir(folder) if isfile(join(folder, f))]
        file_paths += onlyfiles
    return file_paths

def training_test_split(X):
    from random import random
    from random import seed
    X_train = []
    X_test = []
    # seed(16)
    for x in X:
        r = random()
        if r<0.8:
            X_train.append(x)
        else:
            X_test.append(x)
    return X_train, X_test

def get_visual_model():
    nb_classes = 2

    pretrained_name = 'visual_weights/C3D_feature_saved_model_weights.hdf5'
    sample_input = np.empty(
        [5, 112, 112, 3], dtype=np.uint8)
    model = c3d_model_feature(sample_input.shape, nb_classes)  # , feature=True)
    # compile model
    optim = Adam(lr=1e-4, decay=1e-6)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
    # load pretrained weights
    if os.path.exists(pretrained_name):
        print('Pre-existing model weights found, loading weights.......')
        model.load_weights( pretrained_name)
        print('Weights loaded')
    else:
        import time
        print("！！！！！NO VIS WEIGHTS FOUND！！！")
        time.sleep(5)

    return model

def read_depth(file, time):
    folder = file.split('/')[0] + '/' + file.split('/')[1] + '_depth/'
    depth_file = folder + time + ".svo-depth.txt"
    f = open(depth_file, "r")
    i = 0
    depth = []
    summed = 0
    j = 0
    for x in f:
        x = x.split()
        if x[0] != '#' and len(x) == 30:
            if i > 5:
                depth.append(float(x[17]))
            i += 1
    return depth

def one_file(file, upr, vis_model, total, yes):
    i = 0
    work_done = 0
    workdone_x = 0
    workdone_y = 0
    a_A = 0.0020
    a_B = 0.0012
    a = 0.0016
    prev_boom = 0
    F0 = 0
    depth = upr.read_depth(file)
    season = file.split("/")[1]
    time = file.split("/")[2].split(".")[0]
    im_features = []
    images = []

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        demonstrations = []

        for row in csv_reader:
            if (len(depth)>i and i>upr.start):
                if season == "autumn" or season == "winter":
                    P_A = float(row[28])*100000
                    P_B = float(row[27])*100000
                    boom = float(row[71])
                    bucket = float(row[72])
                    vx = float(row[62])
                    l = float(row[21])
                if season == "summer":
                    P_A = float(row[9]) * 100000
                    P_B = float(row[8]) * 100000
                    boom = float(row[1])
                    bucket = float(row[2])
                    vx = (depth[i] - prev_depth) * 15
                    prev_depth = depth[i]
                    l = float(row[3])

                F = a_A*P_A-a_B*P_B
                if i == upr.start+1:
                    F0 = F
                F -= F0
                F_C = F * np.array([np.cos(boom), np.sin(boom)])
                boom_dot = (boom - prev_boom) * 15
                prev_boom = boom
                v_C = np.array([vx-l*boom_dot*np.sin(boom)+a, l*boom_dot*np.cos(boom)+a])
                work_done += abs(np.dot(F_C, v_C))/15
                workdone_x += abs(F_C[0] * v_C[0])/15
                workdone_y += abs(F_C[1] * v_C[1])/15

                # observation = [transmission, P_A, depth[i], boom, bucket]
                observation = [boom, bucket, depth[i]]

                frames = get_frames(time, i)
                if frames.shape[1]==5:
                    im_feature, score = vis_model.predict(frames)
                    frame = frames[0, 0, :, :, :]
                    images.append(frame)

                reward_i, segment, terminal = upr.get_intermediate_reward(observation, im_feature)

                # segments.append(segment)
                upr.combine_reward(reward_i, segment, i)
                reward = upr.reward

                data = observation + [segment] + [terminal] + [reward]

                terminal_gt = float(row[82])
                if terminal==terminal_gt:
                    yes += 1

                demonstrations.append(data)
                im_features.append(im_feature[0])

                total += 1
            i += 1

    demonstrations = np.array(demonstrations)
    im_features = np.array(im_features)

    return demonstrations, total, yes, im_features, images

def plot_all_image_features(im_feature, data, depth, time, m):
    n = im_feature.shape[1]
    colours = colors.BASE_COLORS.keys()
    fields = ["wo", "bo", "bu"]
    for i in range(n):
        plt.subplot(4, n, m * n + i + 1)

        im_feature[:, i] = im_feature[:, i] / (max(abs(im_feature[:, i])))
        # plt.plot(depth, im_feature[:, i], label="im")
        plt.plot(im_feature[:, i], label="im")

        depth = np.array(depth)
        depth = depth / max(abs(depth))
        plt.plot(depth, label="di")

        if i == 0 and m == 0:
            plt.legend()

        for p in range(len(fields)):
            data[0:len(depth), p] = data[0:len(depth), p] / (max(abs(data[0:len(depth), p])))
            # plt.plot(depth, data[0:len(depth), p], label=fields[p])
            plt.plot(data[0:len(depth), p], label=fields[p])
            if i == 0 and m == 0:
                plt.legend()

        plt.title(i)
        if i == 0:
            plt.ylabel(time)

def plot_some(im_feature, data, depth, m):
    plt.subplot(2, 2, m + 1)
    im_feature[:, 4] = im_feature[:, 4] / (max(abs(im_feature[:, 4])))
    plt.plot(im_feature[:, 4], label="4")

    im_feature[:, 2] = im_feature[:, 2] / (max(abs(im_feature[:, 2])))
    plt.plot(im_feature[:, 2], label="2")

    data[0:len(depth), 0] = data[0:len(depth), 0] / (max(abs(data[0:len(depth), 0])))
    plt.plot(data[0:len(depth), 0], label="wo")

    # depth = np.array(depth)
    # depth = depth / max(abs(depth))
    # plt.plot(depth, label="di")

def get_frames(time, k):
    frames = []

    window = []
    for i in range(k, k+5):
            file_name = "data/" + time + "_" + str(i) + ".png"
            frame = cv2.imread(file_name)
            if type(frame) == type(None):
                break
            frame = frame[:, 280:1000, :]
            frame = cv2.resize(frame, (112, 112))
            window.append(frame)

    frames.append(window)

    frames = np.array((frames))
    return frames

def plot_image(images, time):
    k = 0
    p = 0
    while k < len(images) and p<15:
        plt.subplot2grid((4, 15), (0, p))
        plt.imshow(images[k])
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.title(time, color="w")
        if p == 0:
            plt.ylabel(time)
        p += 1
        k += 20

def plot_image_vector(im_feature, time):


    for s in range(im_feature.shape[1]):
        im_feature[:, s] = (
                    255 * (im_feature[:, s] - min(im_feature[:, s])) / (max(im_feature[:, s] - min(im_feature[:, s]))))

    # im_feature = np.transpose(im_feature)

    p = 0

    q = 0
    while q < im_feature.shape[0] and p < 15:
        gray_frame = im_feature[q, :].reshape((8, 1))
        q += 20
        plt.subplot2grid((4, 15), (1, p))
        plt.imshow(gray_frame, cmap='gray', vmin=0, vmax=255)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        if p == 0:
            plt.ylabel(time)
        p += 1
    return p

def plot_data(data, labels,p):
    n = data.shape[1]

    plt.subplot2grid((4, 15), (2, 0), colspan=p)
    for u in range(n-3):
        the_min = min(data[:, u])
        the_max = max(data[:, u])
        data_to_plot = (data[:, u] - the_min) / (the_max - the_min)
        plt.plot(data_to_plot, label=labels[u])
    plt.legend()

    plt.subplot2grid((4, 15), (3, 0), colspan=p)
    for v in range(3, 0, -1):
        the_min = min(data[:, n-v])
        the_max = max(data[:, n-v])
        data_to_plot = (data[:, n-v] - the_min) / (the_max - the_min)
        plt.plot(data_to_plot, label=labels[n-v])
    plt.legend()

def test():
    # init 3dcnn extraction model
    vis_model = get_visual_model()

    file_paths = get_file_paths(["data/winter", "data/autumn"])
    X_train, X_test = training_test_split(file_paths)
    upr = UPR(X_train, n_clusters=3)

    # labels = ["Transmission","Telescopic","Distance", "Boom", "Bucket"]
    labels = ["Boom", "Bucket", "Distance", "Segments", "Terminal", "Reward"]
    # create images input
    m = 0
    yes = 0
    total = 0

    with PdfPages('data_plots.pdf') as pdf:
        plt.rc('text', usetex=False)
        fig = plt.figure(figsize=(30, 0.1))
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        title = "Learning Reward from Demonstrations"
        plt.title(title, fontsize=32)
        columns = ['Training Data', 'Testing Data', 'Sensor Values', 'Clustering & Classification', 'Reward']
        cell_text = [['80 % of positive demos: mix of winter and autumn',
                     '20 % of positive demos: mix of winter and autumn',
                     'Boom angle, Bucket angle, Distance to the pile, Segment, Reward',
                     'K-means clustering and KNN Classifier',
                     'Difference R_max and Distance to the next cluster center']]

        table = plt.table(cellText=cell_text, colLabels=columns, loc='bottom')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        for file in file_paths:

            plt.rc('text', usetex=False)
            fig = plt.figure(figsize=(30, 8))
            plt.title(file.split('/')[2])
            time = file.split("/")[2].split(".")[0]
            season = file.split("/")[1]
            data, total, yes, im_feature, images = one_file(file, upr, vis_model, total, yes)

            plot_image(images, time)

            p = plot_image_vector(im_feature, time)

            plot_data(data, labels, p)

            m += 1

            plt.tight_layout()
            title = season + " - " + time
            fig.suptitle(title, fontsize=16, x=0.2)

            if m!=(len(file_paths)-1):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        accuracy = yes/total
        columns = ["Terminal State Classification Accuracy"]
        cell_text = [[accuracy]]
        plt.table(cellText=cell_text, colLabels=columns, loc='bottom')
        pdf.savefig(fig, bbox_inches='tight')
        print(accuracy)
        plt.close()

test()
