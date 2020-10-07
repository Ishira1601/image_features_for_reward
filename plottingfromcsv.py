import csv

import numpy as np
import matplotlib.pyplot as plt

def plot_image(images):
    k = 0
    p = 0
    fontsize = {"fontsize": 11}
    while k < len(images) and p<32:
        plt.subplot2grid((51, 32), (0, p), rowspan=8, colspan=4)
        # images[k] = cv2.resize(images[k], (448, 448))
        plt.imshow(images[k])
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        if p == 0:
            plt.ylabel("Camera Feed", **fontsize)
        p += 4
        k += 40

def plot_image_vector(im_feature):
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

def plot_low_level(data, p, labels= ["Workdone Norm", "Boom / rad", "Bucket / rad", "Distance / m"]):
    fontsize = {"fontsize":11}
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

def plot_terminal(data, p, labels=["Terminal", "Terminal_GT"]):
    fontsize = {"fontsize": 11}
    plt.subplot2grid((51, 32), (36, 0), colspan=p, rowspan=5)
    for i in range(12, 14):
        data_to_plot = data[:, i]
        plt.plot(time, data_to_plot, label=labels[i - 8])
        plt.grid(axis='x')
        plt.tick_params(labelbottom=False)
    plt.grid(axis='x')
    plt.legend(**fontsize, loc='lower-left')

def plot_reward(data, p, labels=["Stage", "Reward"]):
    fontsize = {"fontsize": 11}
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

def get_file_paths(folders):
    from os import listdir
    from os.path import isfile, join

    file_paths = []
    for folder in folders:
        onlyfiles = [folder+"/"+f for f in listdir(folder) if isfile(join(folder, f))]
        file_paths += onlyfiles
    return file_paths

def from_csv(file):
    data = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row = [float(n) for n in row]
            data.append(row)
    data = np.array(data)
    return data

def from_images_mat(file):
    images = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row = [int(n) for n in row]
            image = np.reshape(row, (112, 112, 3))
            images.append(image)

    return images
files= get_file_paths(["data_results"])
for file in files:

    time = file.split("/")[2]
    img_file = "data_image/" + time
    images =from_images_mat(img_file)
    plot_image(images)

    data = from_csv(file)
    image_features = data[:, 4:12]
    plot_image_vector(image_features)
    low_level_features = data[:, 0:4]
    low_level_features[3] = low_level_features[3]/100

    terminal = data[:, 12:14]

    reward = data[:, 14, 16]