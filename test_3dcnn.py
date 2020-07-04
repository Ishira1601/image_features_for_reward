import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from matplotlib import colors
from T3D_keras import densenet161_3D_DropOut, densenet121_3D_DropOut, xception_classifier, c3d_model, \
    c3d_model_feature

from tensorflow.keras.optimizers import Adam
import os

def get_file_paths(folders):
    from os import listdir
    from os.path import isfile, join

    file_paths = []
    for folder in folders:
        onlyfiles = [folder+"/"+f for f in listdir(folder) if isfile(join(folder, f))]
        file_paths += onlyfiles
    return file_paths

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

def read_depth(time):
    folder = 'data/depth/'
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
                summed += float(x[17])
                j += 1
                if j == 5:
                    summed /= 5
                    depth.append(summed)
                    j = 0
            i += 1
    return depth

def one_file(file):
    i = 0
    work_done = 0
    workdone_x = 0
    workdone_y = 0
    a_A = 0.0020
    a_B = 0.0012
    a = 0.0016
    prev_boom = 0
    F0 = 0
    j = 0
    work_done_sum = 0
    boom_sum = 0
    bucket_sum = 0

    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        demonstrations = []
        # distance_travelled = upr.get_distance_travelled(file)
        for row in csv_reader:
                P_A = float(row[28])*100000
                P_B = float(row[27])*100000
                boom = float(row[71])
                bucket = float(row[72])
                vx = float(row[62])
                l = float(row[21])
                F = a_A*P_A-a_B*P_B
                if i == 0:
                    F0 = F
                F -= F0
                F_C = F * np.array([np.cos(boom), np.sin(boom)])
                boom_dot = (boom - prev_boom) * 15
                prev_boom = boom
                v_C = np.array([vx-l*boom_dot*np.sin(boom)+a, l*boom_dot*np.cos(boom)+a])
                work_done += abs(np.dot(F_C, v_C))/15
                workdone_x += abs(F_C[0] * v_C[0])/15
                workdone_y += abs(F_C[1] * v_C[1])/15
                work_done_sum += work_done
                boom_sum += boom
                bucket_sum += bucket
                j += 1
                if j == 5:
                    work_done_sum /= 5
                    boom_sum /= 5
                    bucket_sum /= 5
                    observation = [work_done_sum, boom_sum, bucket_sum]
                    demonstrations.append(observation)
                    work_done_sum = 0
                    boom_sum = 0
                    bucket_sum = 0
                    j = 0
                i += 1

    data = np.array(demonstrations)

    return data

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

def test():
    # init 3dcnn extraction model    
    vis_model = get_visual_model()

    file_paths = get_file_paths(["data/csv"])
    # create images input
    m = 0
    for file in file_paths:
        if m == 4 :
            plt.show()
            plt.figure()
            m = 0
        time = file.split("/")[2].split(".")[0]
        j = 0
        frames = []
        depth = read_depth(time)
        the_csv = "data/csv/"+time+".csv"
        data = one_file(the_csv)
        while j<len(depth):
            window = []
            for i in range(5):
                k = j*5 + i
                file_name = "data/"+time+"_"+str(k)+".png"
                frame = cv2.imread(file_name)
                frame = frame[:, 280:1000, :]
                frame = cv2.resize(frame, (112, 112))
                window.append(frame)
            frames.append(window)
            j += 1
        frames = np.array((frames))

        im_feature, score = vis_model.predict(frames)

        # plot_some(im_feature, data, depth, m)

        # plot_all_image_features(im_feature, data, depth, time, m)
        m += 1

        plt.title(file)
        plt.legend()
        # plt.show()
test()
