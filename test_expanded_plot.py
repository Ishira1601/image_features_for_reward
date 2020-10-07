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
    seed(16)
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
    prev_work_done = 0
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

                ## Populate feature vector
                sensor_values = [P_A, P_B, boom, bucket, vx, l]
                observation, F0 = upr.get_feature_vector(sensor_values, season, depth[i],
                                                         i, prev_boom, prev_work_done, F0)

                frames = get_frames(time, i)
                if frames.shape[1]==5:
                    im_feature, score = vis_model.predict(frames)
                    frame = frames[0, 0, :, :, :]
                    images.append(frame)
                prev_work_done, prev_boom, prev_bucket, prev_distance = observation
                n= len(observation)
                observation = np.array(observation).reshape(1, n)
                observation = np.hstack((observation, im_feature))
                reward, segment, terminal = upr.get_reward(observation, im_feature)



                terminal_gt = int(row[82])
                if terminal==terminal_gt:
                    yes += 1

                data = list(observation[0]) + [terminal] + [terminal_gt]+ [segment] + [reward]

                demonstrations.append(data)
                im_features.append(im_feature[0])

                total += 1
            i += 1

    demonstrations = np.array(demonstrations)
    im_features = np.array(im_features)

    return demonstrations, total, yes, im_features, images

def plot_all_image_features(im_feature, data, depth, time, m):
    n = im_feature.shape[1]

    fields = ["wo", "bo", "bu"]
    for i in range(n):
        plt.subplot(4, n, m * n + i + 1)

        im_feature[:, i] = im_feature[:, i] / (max(abs(im_feature[:, i])))

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
    fontsize = {"fontsize": 11}
    while k < len(images) and p<32:
        plt.subplot2grid((51, 32), (0, p), rowspan=8, colspan=4)
        # images[k] = cv2.resize(images[k], (448, 448))
        plt.imshow(images[k])
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        plt.title(time, color="w")
        if p == 0:
            plt.ylabel("Camera Feed", **fontsize)
        p += 4
        k += 40

def plot_image_vector(im_feature, time):
    fontsize = {"fontsize": 11}
    for s in range(im_feature.shape[1]):
        im_feature[:, s] = (
                    255 * (im_feature[:, s] - min(im_feature[:, s])) / (max(im_feature[:, s] - min(im_feature[:, s]))))

    # im_feature = np.transpose(im_feature)

    p = 0

    q = 0
    while q < im_feature.shape[0] and p < 32:
        gray_frame = im_feature[q, :].reshape((8, 1))
        q += 40
        plt.subplot2grid((51, 32), (8, p), rowspan=8, colspan=4)
        plt.imshow(gray_frame, cmap='gray', vmin=0, vmax=255)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        if p == 0:
            plt.ylabel("Image-vectors", **fontsize)
        p += 4
    return p

def plot_data(data, labels, p):
    fontsize = {"fontsize":9}
    n = data.shape[0]
    time = np.arange(0, n)/15

    for v in range(4):
        plt.subplot2grid((51, 32), (5*v+16, 0), colspan=p, rowspan=5)
        colour = "b-"
        data_to_plot = data[:, v]
        if (v == 1 or v == 2):
            colour = "m-"
        if (v == 3):
            colour = "g-"
            data_to_plot = data_to_plot/100
        plt.plot(time, data_to_plot, colour)
        plt.ylabel(labels[v], **fontsize)
        plt.tick_params(labelbottom=False)
        plt.grid(axis='x')

    plt.subplot2grid((51, 32), (36, 0), colspan=p, rowspan=5)
    for i in range(12, 14):
        data_to_plot = data[:, i]
        plt.plot(time, data_to_plot, label=labels[i - 8])
        plt.grid(axis='x')
        plt.tick_params(labelbottom=False)
    plt.grid(axis='x')
    plt.legend(**fontsize, loc='lower-left')

    for v in range(14, 16):
        plt.subplot2grid((51, 32), (5*v-29, 0), colspan=p, rowspan=5)
        colour = "b-"
        if (v == 15):
            colour = "r-"
        data_to_plot = data[:, v]
        plt.plot(time, data_to_plot, colour)
        if v==14:
            plt.tick_params(labelbottom=False)
        plt.grid(axis='x')

        plt.ylabel(labels[v-8], **fontsize)

    fontsize = {"fontsize": 11}
    plt.tick_params(labelbottom=True)
    plt.xlabel("time / s", **fontsize)


def test():
    # init 3dcnn extraction model
    vis_model = get_visual_model()

    fontsize = {"fontsize":"x-large"}

    file_paths = get_file_paths(["data/winter", "data/autumn"])
    X_train, X_test = training_test_split(file_paths)
    R_max = 1600

    # X_train = get_file_paths(["data/autumn"])
    # X_test = get_file_paths(["data/winter"])

    # X_train = get_file_paths(["data/winter"])
    # X_test = get_file_paths(["data/autumn"])

    upr = UPR(X_train, n_clusters=3, R_max = R_max)

    # labels = ["Transmission","Telescopic","Distance", "Boom", "Bucket"]
    labels = ["Workdone Norm", "Boom / rad", "Bucket / rad ", "Distance / m", "Terminal", "Terminal GT", "Stage", "Reward"]
    # create images input
    m = 0
    yes = 0
    total = 0
    times = []



    for file in X_test:

        plt.rc('text', usetex=False)
        fig = plt.figure(figsize=(32, 51))
        # plt.title(file.split('/')[2], **fontsize)
        time = file.split("/")[2].split(".")[0]
        season = file.split("/")[1]
        a = datetime.datetime.now()
        data, total, yes, im_feature, images = one_file(file, upr, vis_model, total, yes)
        b = datetime.datetime.now()
        c = b-a
        times.append(c.seconds)
        plot_image(images, time)

        p = plot_image_vector(im_feature, time)

        plot_data(data, labels, p)

        m += 1

        plt.tight_layout()
        title = season + " - " + time
        # fig.suptitle(title, fontsize=16, x=0.2)

        if m!=(len(X_test)-1):
            plt.show()
            plt.close()

    accuracy = yes/total
    print("Terminal State Classification Accuracy")

    print(accuracy)

    plt.close()

test()