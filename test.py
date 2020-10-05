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
    ## Get File Paths ffor files in folders
    from os import listdir
    from os.path import isfile, join

    file_paths = []
    for folder in folders:
        onlyfiles = [folder+"/"+f for f in listdir(folder) if isfile(join(folder, f))]
        file_paths += onlyfiles
    return file_paths

def training_test_split(X):
    ## Obtain 80% Training and 20% Testing Data
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

def one_file(file, image_folder, upr, vis_model, total, yes):
    ## Obtain Reward function for one demonstration
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
                ## Get Image Features
                frames = get_frames(image_folder, time, i)
                if frames.shape[1] == 5:
                    im_feature, score = vis_model.predict(frames)
                    frame = frames[0, 0, :, :, :]
                    images.append(frame)

                ## Extract sensor values
                if season == "autumn" or season == "winter":
                    P_A = float(row[28]) * 100000
                    P_B = float(row[27]) * 100000
                    boom = float(row[71])
                    bucket = float(row[72])
                    vx = float(row[62])
                    l = float(row[21])

                ## Populate feature vector
                sensor_values = [P_A, P_B, boom, bucket, vx, l]
                observation, F0 = upr.get_feature_vector(sensor_values, season, depth[i],
                                                       i, prev_boom, prev_work_done, F0)


                reward, segment, terminal = upr.get_reward(observation, im_feature)

                terminal_gt = int(row[82])
                if terminal==terminal_gt:
                    yes += 1

                data = observation + [segment] + [terminal] + [reward] + [terminal_gt]

                demonstrations.append(data)
                im_features.append(im_feature[0])

                prev_work_done, prev_boom, prev_bucket, prev_distance = observation

                total += 1
            i += 1

    demonstrations = np.array(demonstrations)
    im_features = np.array(im_features)

    return demonstrations, total, yes, im_features, images

def get_frames(svo_folder, time, k):
    ## Get frames from time
    frames = []

    window = []
    for i in range(k, k+5):
            file_name = svo_folder + "/" + time + "_" + str(i) + ".png"
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
    ## Plot images
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
    ## Plot 8 image features
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

def plot_data(data, labels, p):
    n = data.shape[1]

    ## Plot Workdone, Boom and Bucket Angle, Distance to the pile
    plt.subplot2grid((4, 15), (2, 0), colspan=p)
    for u in range(n-4):
        the_min = min(data[:, u])
        the_max = max(data[:, u])
        data_to_plot = (data[:, u] - the_min) / (the_max - the_min)
        plt.plot(data_to_plot, label=labels[u])
    plt.legend()

    ## Plot Reward, Stage, Terminal, Terminal GT
    plt.subplot2grid((4, 15), (3, 0), colspan=p)
    for v in range(4, 0, -1):
        the_min = min(data[:, n-v])
        the_max = max(data[:, n-v])
        data_to_plot = (data[:, n-v] - the_min) / (the_max - the_min)
        plt.plot(data_to_plot, label=labels[n-v])
        plt.xlabel("time")
    plt.legend()

def test():
    file_paths = get_file_paths(["data/winter", "data/autumn"])
    X_train, X_test = training_test_split(file_paths)

    # X_train = get_file_paths(["data/autumn"])
    # X_test = get_file_paths(["data/winter"])

    # X_train = get_file_paths(["data/winter"])
    # X_test = get_file_paths(["data/autumn"])


    ## Set up Unsupervised Perceptual Reward
    R_max = 1600
    image_folder = "data"
    upr = UPR(X_train, image_folder, n_clusters=3, R_max=R_max)

    # init 3dcnn extraction model
    vis_model = upr.get_visual_model()

    labels = ["Workdone", "Boom", "Bucket", "Distance", "Segments", "Terminal", "Reward", "Terminal GT"]
    # create images input
    m = 0
    yes = 0
    total = 0
    times = []

    with PdfPages('data_plots.pdf') as pdf:
        plt.rc('text', usetex=False)
        fig = plt.figure(figsize=(30, 0.1))
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        title = "Learning Reward from Demonstrations w/sensors (R_max = " + str(R_max) + ")"
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

        for file in X_test:

            plt.rc('text', usetex=False)
            fig = plt.figure(figsize=(30, 8))
            plt.title(file.split('/')[2])
            time = file.split("/")[2].split(".")[0]
            season = file.split("/")[1]
            a = datetime.datetime.now()
            data, total, yes, im_feature, images = one_file(file, "data", upr, vis_model, total, yes)
            b = datetime.datetime.now()
            c = b-a
            times.append(c.seconds)
            plot_image(images, time)

            p = plot_image_vector(im_feature, time)

            plot_data(data, labels, p)

            m += 1

            plt.tight_layout()
            title = season + " - " + time
            fig.suptitle(title, fontsize=16, x=0.2)

            if m!=(len(X_test)-1):
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        accuracy = yes/total
        print("Terminal State Classification Accuracy")
        pdf.savefig(fig, bbox_inches='tight')
        print(accuracy)
        print("Runtime")
        run_time = np.mean(np.array(times))
        print(run_time)
        plt.close()

test()
