import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import warnings
warnings.filterwarnings("ignore")
import math
from T3D_keras import densenet161_3D_DropOut, densenet121_3D_DropOut, xception_classifier, c3d_model, \
    c3d_model_feature
from tensorflow.keras.optimizers import Adam
import os
import cv2

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=20, n_estimators=100, max_features=4),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier()]

class UPR:
    def __init__(self, files, n_clusters, R_max):
        self.files = files
        self.R_max = R_max
        self.reward = 0
        self.demonstrations = []
        self.expert = []
        self.y = []
        self.X = []
        self.data = []
        self.T = 0
        self.start = 5
        self.winter_max = 5000
        self.autumn_max = 2000
        self.load_data()
        self.terminal_state_classifier(len(self.data[0]))
        self.the_stages = []
        self.n_clusters = n_clusters
        self.stages()
        self.step_classifier()



    def load_data(self):
        n = 0
        k = 0
        y = []
        times = []
        autumn_maxes = []
        winter_maxes = []
        for file in self.files:
            i = 0
            depth = self.read_depth(file)
            observations = []
            all_data = []
            season = file.split("/")[1]
            work_done = 0
            a_A = 0.0020
            a_B = 0.0012
            alpha = (20 / 180) * 3.142
            l = 1.5
            a = 0.0016
            prev_boom = 0
            prev_depth = 1.5
            F0 = 0

            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                for row in csv_reader:
                    # distance_to_pile = distance_travelled[-1]-distance_travelled[i]

                    if (len(depth) > i and i>self.start):
                        if season == "autumn" or season == "winter":
                            P_A = float(row[28]) * 100000
                            P_B = float(row[27]) * 100000
                            boom = float(row[71])
                            bucket = float(row[72])
                            vx = float(row[62])
                            l = float(row[21])
                        if season == "summer":
                            P_A = float(row[9]) * 100000
                            P_B = float(row[8]) * 100000
                            boom = float(row[1])
                            bucket = float(row[2])
                            vx = (depth[i] - prev_depth)
                            prev_depth = depth[i]
                            l = float(row[3])

                        F = a_A * P_A - a_B * P_B
                        if i < self.start+1:
                            F0 = F
                        F -= F0
                        F_C = F * np.array([np.cos(boom), np.sin(boom)])
                        if season == "winter":
                            F_C /= 12700
                        elif season == "autumn":
                            F_C /= 6700
                        boom_dot = (boom - prev_boom) * 15
                        prev_boom = boom
                        v_C = np.array([vx - l * boom_dot * np.sin(boom) + a, l * boom_dot * np.cos(boom) + a])
                        work_done += abs(np.dot(F_C, v_C)) / 15

                        observation = [k,
                           work_done, boom, bucket, depth[i]]
                        n = len(observation)
                        # summed += np.array(observation)
                        # if i%5 == 0:
                        #     averaged = list(summed/5)
                        #
                        #     summed = 0
                        observations.append(observation)
                        y.append(float(row[82]))
                    i += 1
                    data = [float(m) for m in row]
                    all_data.append(data)
                # self.plot_all(all_data, file)
            # if season == "winter":
            #     winter_maxes.append(work_done)
            # elif season == "autumn":
            #     autumn_maxes.append(work_done)



            self.data.append(observations)
            self.demonstrations = self.demonstrations + observations

            if k == 0:
                self.T = i
            k+=1

        # winter_maxes = np.array(winter_maxes)
        # autumn_maxes = np.array(autumn_maxes)
        #
        # self.winter_max = np.mean(winter_maxes)
        # self.autumn_max = np.mean(autumn_maxes)
        self.demonstrations = np.array(self.demonstrations)
        self.expert = self.demonstrations[:, 1:n]
        self.X = self.expert

        # self.clf_binary = KNeighborsClassifier()
        # self.clf_binary.fit(self.X, y)

    def read_depth(self, file):
        time = file.split('/')[2].split('.csv')[0]
        folder = file.split('/')[0] + '/' + file.split('/')[1] + '_depth/'
        depth_file = folder + time + ".svo-depth.txt"
        f = open(depth_file, "r")
        i = 0
        depth = []
        vals = []
        for x in f:
            x = x.split()
            if x[0] != '#' and len(x) == 30:
                if i>self.start:
                    vals.append([float(i) for i in x])
                    depth.append(float(x[17]))
                i += 1

        return depth

    def get_distance_travelled(self, file):
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            distance = [0.0]
            for row in csv_reader:
                distance.append(distance[-1]+float(row[62]))
        return distance

    def get_visual_model(self):
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
            model.load_weights(pretrained_name)
            print('Weights loaded')
        else:
            import time
            print("！！！！！NO VIS WEIGHTS FOUND！！！")
            time.sleep(5)

        return model

    def get_frames(self, time):
        k = self.start
        # k = 24
        frames = []

        flag = True

        while flag:
            window = []
            for i in range(k, k + 5):
                file_name = "data/" + time + "_" + str(i) + ".png"
                frame = cv2.imread(file_name)
                if type(frame) == type(None):
                    flag = False
                    break
                frame = frame[:, 280:1000, :]
                frame = cv2.resize(frame, (112, 112))
                window.append(frame)
            if flag:
                frames.append(window)
            k += 1

        frames = np.array((frames))
        return frames

    def terminal_state_classifier(self, n):
        X = None
        y = []
        vis_model = self.get_visual_model()
        for file in self.files:
            time = file.split("/")[2].split(".")[0]
            frames = self.get_frames(time)

            im_feature, score = vis_model.predict(frames)

            if type(X)==type(None):
                X = im_feature
            else:
                X = np.vstack((X, im_feature))
            i = 0
            j = 0
            flag = 0
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')

                for row in csv_reader:
                    if i>self.start and j<im_feature.shape[0]:
                        terminal = int(row[82])
                        y.append(terminal)
                        if terminal==1:
                            flag = 1
                        j += 1
                    i += 1
            if flag==0:
                y[-1] = int(1)
            while j<im_feature.shape[0]:
                last = y[-1]
                y.append(last)
                j+=1
        self.clf_binary = classifiers[4]
        self.clf_binary.fit(X, y)

    def plot_data(self, data, main_title="Training", title="", cluster_centers=np.zeros((1)), js=[]):
        row = 3
        col = 3
        plt.subplot(row, col, 1)
        plt.title(main_title)
        plt.plot(data[:, 0], 'b')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 0], 'r*')
        plt.ylabel('Work done')

        plt.subplot(row, col, 2)
        plt.plot(data[:, 1], 'm')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 1], 'r*')
        plt.ylabel('Boom Angle')

        plt.subplot(row, col, 3)
        plt.plot(data[:, 2], 'm')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 2], 'r*')
        plt.ylabel('Bucket angle')

        plt.subplot(row, col, 4)
        plt.plot(data[:, 3], 'g')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 3], 'r*')
        plt.ylabel('Distance to pile')


        plt.subplot(row, col, 5)
        plt.plot(data[:, -1])
        plt.ylabel('segment')

        plt.xlabel('time')
        plt.show()

    def plot_all(self, all_data, file):
        plt.suptitle(file)
        all_data = np.array(all_data)
        for i in range(30):
            plt.subplot(5, 6, i+1)
            plt.title(i)
            plt.plot(all_data[:, i])
        plt.show()
        plt.figure()
        plt.suptitle(file)
        for i in range(30):
            plt.subplot(5, 6, i + 1)
            plt.title(i+30)
            plt.plot(all_data[:, i+30])
        plt.show()
        plt.figure()
        plt.suptitle(file)
        for i in range(20):
            plt.subplot(4, 5, i + 1)
            plt.title(i+60)
            plt.plot(all_data[:, i+60])
        plt.show()

    def stages(self):
        # cluster_centers, js = self.set_cluster_centers()
        cluster_centers = []
        cluster_centers.append(self.expert[20])
        middle = round(self.T/2)
        middle = 80
        cluster_centers.append(self.expert[middle])
        end = self.T - 5
        end = 190
        cluster_centers.append(self.expert[end])
        cluster_centers = np.array(cluster_centers)
        clusters = KMeans(n_clusters=self.n_clusters, init=cluster_centers).fit(self.X)
        # clusters = AgglomerativeClustering(n_clusters=self.n_clusters).fit(self.X)
        n = self.expert.shape[0]
        self.y = clusters.labels_
        # self.X = np.vstack((self.X, self.x))
        # n_x = self.x.shape[0]
        # y_x = 3 * np.ones((n_x), dtype=int)
        # self.y = np.hstack((self.y, y_x))
        # y = np.array(self.y).reshape((n, 1))
        # a = 0
        # for i in range(len(self.data)):
        #     b = a + len(self.data[i])
        #     only_data = np.array(self.data[i])
        #     only_labels = y[a:b, :]
        #     data = np.hstack((only_data, only_labels))
        #     data = np.delete(data, 0, 1)
        #     self.plot_data(data, "Training", self.files[i], cluster_centers=cluster_centers)
        #     a = b
        self.to_stages(self.y)

    def set_cluster_centers(self):
        i = round(self.T / (2 * (self.n_clusters-1)))
        cluster_centers = []
        k = 1
        j = k * i
        js= []
        while j<self.T:
            cluster_centers.append(self.expert[j])
            js.append(j)
            k += 2
            j = k * i
        cluster_centers.append(self.expert[self.T-8])
        return cluster_centers, js


    def to_stages(self, y):
        n= len(y)
        stages = []
        samples = self.X
        for i in range(n):
            if stages==[]:
                stages.append([samples[i]])
            elif len(stages)>y[i]:
                stages[y[i]].append(samples[i])
            else:
                stages.append([samples[i]])
        the_stages = []
        for stage in stages:
            mu_and_sigma = self.get_mean_and_variance(np.array(stage))
            the_stages.append(mu_and_sigma)
        # self.terminal_state()
        self.the_stages = np.array(the_stages)

    def get_mean_and_variance(self, x):
        mu_x = np.mean(x, axis=0)
        sigma_x = np.std(x, axis=0)
        return np.array([mu_x, sigma_x])

    def reset(self):
        self.reward = 0

    def step_classifier(self):
        self.clf = KNeighborsClassifier()
        self.clf.fit(self.X, self.y)

    def get_intermediate_reward(self, state, im_feature):
        n = len(state)-1
        segment = self.clf.predict([state])[0]
        terminal = int(self.clf_binary.predict([im_feature[0]])[0])
        expert_t = self.the_stages[segment]
        if (segment+1<self.n_clusters):
            expert_t = self.the_stages[segment+1]
        mu_t = expert_t[0]
        sigma_t = expert_t[1]
        summed = 0
        for j in range(n):
            dist = (np.square(state[j] - mu_t[j])) / np.square(sigma_t[j])
            if not math.isnan(dist) and not math.isinf(dist):
                summed = summed + dist
            else:
                continue
        # reward_t = n/summed
        reward_t = self.R_max - summed
        return reward_t, segment, terminal

    def combine_reward(self, reward_i, segment):
        # if (time>300):
        #     self.reward -= time*100

        if segment > 0:
            # self.reward += reward_i*pow(2, segment-1)
            self.reward = reward_i * pow(2, segment)
        else:
            self.reward = reward_i * np.sqrt(2)





