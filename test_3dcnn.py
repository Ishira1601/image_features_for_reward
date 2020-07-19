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

# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
# with PdfPages('multipage_pdf.pdf') as pdf:
#     plt.figure(figsize=(3, 3))
#     x = np.arange(0, 5, 0.1)
#     plt.plot(x, np.sin(x), 'b-')
#     plt.title('Page One')
#     pdf.savefig()  # saves the current figure into a pdf page
#     plt.close()
#
#     # if LaTeX is not installed or error caught, change to `usetex=False`
#     plt.rc('text', usetex=False)
#     plt.figure(figsize=(8, 6))
#     x = np.arange(0, 5, 0.1)
#     plt.plot(x, np.sin(x), 'b-')
#     plt.title('Page Two')
#     pdf.attach_note("plot of sin(x)")  # you can add a pdf note to
#                                        # attach metadata to a page
#     pdf.savefig()
#     plt.close()
#
#     plt.rc('text', usetex=False)
#     fig = plt.figure(figsize=(4, 5))
#     plt.plot(x, x ** 2, 'ko')
#     plt.title('Page Three')
#     pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
#     plt.close()
#
#     # We can also set the file's metadata via the PdfPages object:
#     d = pdf.infodict()
#     d['Title'] = 'Multipage PDF Example'
#     d['Author'] = 'Jouni K. Sepp\xe4nen'
#     d['Subject'] = 'How to create a multipage pdf file and set its metadata'
#     d['Keywords'] = 'PdfPages multipage keywords author title subject'
#     d['CreationDate'] = datetime.datetime(2009, 11, 13)
#     d['ModDate'] = datetime.datetime.today()

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

def one_file(file, time):
    i = 0
    work_done = 0
    workdone_x = 0
    workdone_y = 0
    a_A = 0.0020
    a_B = 0.0012
    a = 0.0016
    prev_boom = 0
    F0 = 0
    depth = read_depth(file, time)
    season = file.split("/")[1]
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        demonstrations = []

        for row in csv_reader:
            if (len(depth)>i):
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

                observation = [depth[i], boom, bucket]
                demonstrations.append(observation)
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

def example_plot(ax, fontsize=12):
    ax.plot([1, 2])

    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

def test():
    # init 3dcnn extraction model    
    vis_model = get_visual_model()

    file_paths = get_file_paths(["data/winter", "data/autumn"])
    labels = ["Distance", "Boom", "Bucket"]
    # create images input
    m = 0
    n = len((file_paths))
    with PdfPages('data_plots.pdf') as pdf:
        for file in file_paths:
            p = 0
            # if m == 4 :
            #     plt.show()
            #     plt.figure()
            #     m = 0
            time = file.split("/")[2].split(".")[0]
            j = 0
            k = 0
            frames = []
            axs = []
            data = one_file(file, time)
            frame = 0
            plt.rc('text', usetex=False)
            fig = plt.figure(figsize=(30, 6))
            plt.title(file.split('/')[2])
            while k < (data.shape[0]):
                window = []
                for i in range(5):
                    k = j*5 + i
                    if k > 24 and k < (data.shape[0]):
                        file_name = "data/"+time+"_"+str(k)+".png"
                        frame = cv2.imread(file_name)
                        if type(frame)==type(None):
                            break
                        frame = frame[:, 280:1000, :]
                        frame = cv2.resize(frame, (112, 112))
                        window.append(frame)
                        if k % 20 == 0 and p < 15:
                            plt.subplot2grid((3, 15), (0, p))
                            plt.imshow(frame)
                            plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                            plt.title(time, color="w")
                            if p == 0:
                                plt.ylabel(time)
                            p += 1
                if k > 24 and k < (data.shape[0]) and type(frame)!=type(None):
                    frames.append(window)
                j += 1
            if p < 15 and type(frame)!=type(None):
                plt.subplot2grid((3, 15), (0, p))
                plt.imshow(frame)
                plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                p += 1

            frames = np.array((frames))

            im_feature, score = vis_model.predict(frames)

            for s in range(im_feature.shape[1]):
                im_feature[:, s] = (255*(im_feature[:, s] - min(im_feature[:, s])) / (max(im_feature[:, s]-min(im_feature[:, s]))))

            # im_feature = np.transpose(im_feature)

            p = 0

            q = 0
            while q < im_feature.shape[0] and p < 15:
                gray_frame = im_feature[q, :].reshape((1, 8))
                q += 4
                plt.subplot2grid((3, 15), (1, p))
                plt.imshow(gray_frame, cmap='gray', vmin=0, vmax=255)
                plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                if p==0:
                    plt.ylabel(time)
                p += 1

            # # plot_all_image_features(im_feature, data, depth, time, m)
            plt.subplot2grid((3, 15), (2, 0), colspan=p)
            for u in range(data.shape[1]):
                the_min = min(data[:, u])
                the_max = max(data[:, u])
                data_to_plot = (data[:, u]-the_min)/(the_max-the_min)
                plt.plot(data_to_plot, label=labels[u])

                if p==0:
                    plt.ylabel(time)

            m += 1
            plt.legend()
            plt.tight_layout()
            fig.suptitle(time, fontsize=16, x=0.2)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    # plt.show()

test()
