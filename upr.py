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
        self.y = []
        self.X = []
        self.data = []
        self.T = 0
        self.start = 5
        self.X_n = []
        self.winter_max = 12700  # number to divide to normalize winter workdone
        self.autumn_max = 6700  # n number to divide by to normalize autumn workdone
        self.load_data()  # Get Load expert data
        self.terminal_state_classifier()
        self.the_stages = []
        self.n_clusters = n_clusters
        self.stages() # Calculate mu and sigma for stages
        self.step_classifier() # Fit Stage Classifier




    def load_data(self):
        n = 0
        k = 0
        y = []

        for file in self.files:
            i = 0
            depth = self.read_depth(file)
            observations = []
            all_data = []
            season = file.split("/")[1]
            prev_work_done = 0
            a_A = 0.0020
            a_B = 0.0012
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

                        sensor_values = [P_A, P_B, boom, bucket, vx, l]
                        features, F0 = self.get_feature_vector(sensor_values, season, depth[i],
                                                               i, prev_boom, prev_work_done, F0)

                        observation = [k] + features
                        n = len(observation)
                        prev_work_done, prev_boom, prev_bucket, prev_distance = features
                        observations.append(observation)
                        y.append(float(row[82]))
                    i += 1
                    data = [float(m) for m in row]
                    all_data.append(data)
                # self.plot_all(all_data, file)
            self.data.append(observations)
            self.demonstrations = self.demonstrations + observations

            if k == 0:
                self.T = i

            m = len(self.data[-1])
            self.X_n.append(m)
            k+=1


        self.demonstrations = np.array(self.demonstrations)
        self.X = self.demonstrations[:, 1:n]

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

    def get_feature_vector(self, sensor_values, season, distance, i, prev_boom, work_done, F0, a_A=0.0020, a_B=0.0012,
                           a=0.0016):
        P_A, P_B, boom, bucket, vx, l = sensor_values
        F = a_A * P_A - a_B * P_B
        if i < self.start + 1:
            F0 = F
        F -= F0
        F_C = F * np.array([np.cos(boom), np.sin(boom)])
        if season == "winter":
            F_C /= self.winter_max
        elif season == "autumn":
            F_C /= self.autumn_max
        boom_dot = (boom - prev_boom) * 15

        v_C = np.array([vx - l * boom_dot * np.sin(boom) + a, l * boom_dot * np.cos(boom) + a])
        work_done += abs(np.dot(F_C, v_C)) / 15

        observation = [work_done, boom, bucket, distance]

        return observation, F0

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

    def terminal_state_classifier(self):
        X = None
        y = []
        vis_model = self.get_visual_model()
        k = 0
        for file in self.files:
            time = file.split("/")[2].split(".")[0]
            frames = self.get_frames(time)

            im_feature, score = vis_model.predict(frames)
            n = self.X_n[k]
            while n > im_feature.shape[0]:
                last = im_feature[-1, :]
                im_feature = np.vstack((im_feature, last))
            if n < im_feature.shape[0]:
                im_feature = im_feature[0:n, :]
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
            k+=1
        self.image_features = X
        self.clf_binary = classifiers[3]
        self.clf_binary.fit(X, y)

    def stages(self):
        # cluster_centers, js = self.set_cluster_centers()
        cluster_centers = []
        self.X = np.hstack((self.X, self.image_features))
        cluster_centers.append(self.X[20])
        middle = round(self.T/2)
        middle = 90
        cluster_centers.append(self.X[middle])
        end = self.T - 20
        end = 190
        cluster_centers.append(self.X[end])
        cluster_centers = np.array(cluster_centers)
        clusters = KMeans(n_clusters=self.n_clusters, init=cluster_centers).fit(self.X)
        # clusters = AgglomerativeClustering(n_clusters=self.n_clusters).fit(self.X)
        n = self.X.shape[0]
        self.y = clusters.labels_
        y = np.array(self.y).reshape((n, 1))

        self.to_stages(self.y)

    def set_cluster_centers(self):
        i = round(self.T / (2 * (self.n_clusters-1)))
        cluster_centers = []
        k = 1
        j = k * i
        js= []
        while j<self.T:
            cluster_centers.append(self.X[j])
            js.append(j)
            k += 2
            j = k * i
        cluster_centers.append(self.X[self.T-8])
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
        # self.X = np.hstack((self.X, self.image_features))
        self.clf = KNeighborsClassifier()
        self.clf.fit(self.X, self.y)

    def get_reward(self, state, im_feature):
        n = state.shape[1]
        segment = self.clf.predict(state)[0]
        terminal = int(self.clf_binary.predict([im_feature[0]])[0])
        expert_t = self.the_stages[segment]
        if (segment+1<self.n_clusters):
            expert_t = self.the_stages[segment+1]
        mu_t = expert_t[0]
        sigma_t = expert_t[1]
        summed = 0
        for j in range(n):
            dist = (np.square(state[0][j] - mu_t[j])) / np.square(sigma_t[j])
            if not math.isnan(dist) and not math.isinf(dist):
                summed = summed + dist
            else:
                continue
        # reward_t = n/summed
        reward_t = self.R_max - summed
        reward = self.combine_reward(reward_t, segment)
        return reward, segment, terminal

    def combine_reward(self, reward_i, segment):
        if segment>0:
            reward = reward_i * pow(2, segment)
        else:
            reward = reward_i * np.sqrt(2)
        return reward




