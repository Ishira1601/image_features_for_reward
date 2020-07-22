import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
import math
class UPR:
    def __init__(self, files, n_clusters):
        self.files = files
        self.reward = 0
        self.demonstrations = []
        self.expert = []
        self.y = []
        self.X = []
        self.data = []
        self.T = 0
        self.load_data()
        self.the_stages = []
        self.n_clusters = n_clusters
        self.stages()
        self.step_classifier()


    def load_data(self):
        n = 0
        k = 0
        for file in self.files:
            i = 0
            depth = self.read_depth(file)
            observations = []
            all_data = []
            season = file.split("/")[1]
            work_done = 0
            workdone_x = 0
            workdone_y =0
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

                    if (len(depth) > i):
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
                        if i == 0:
                            F0 = F
                        F -= F0
                        F_C = F * np.array([np.cos(boom), np.sin(boom)])
                        boom_dot = (boom - prev_boom) * 15
                        prev_boom = boom
                        v_C = np.array([vx - l * boom_dot * np.sin(boom) + a, l * boom_dot * np.cos(boom) + a])
                        work_done += abs(np.dot(F_C, v_C)) / 15
                        workdone_x += abs(F_C[0] * v_C[0]) / 15
                        workdone_y += abs(F_C[1] * v_C[1]) / 15
                        F_Re = np.linalg.norm(F_C)
                        v_Re = np.linalg.norm(v_C)

                        observation = [k, work_done,
                           boom, bucket, depth[i]]
                        n = len(observation)
                        observations.append(observation)
                        i += 1
                    data = [float(m) for m in row]
                    all_data.append(data)
                # self.plot_all(all_data, file)
            self.data.append(observations)
            self.demonstrations = self.demonstrations + observations
            if k==0:
                self.T = i

            k+=1

        self.demonstrations = np.array(self.demonstrations)
        self.expert = self.demonstrations[:, 1:n]
        X = []
        x = []
        for d in self.data:
            X = X + d[:-3]
            x = x + d[-3:]
        self.X = np.array(X)
        self.X = self.X[:, 1:n]
        self.x = np.array(x)
        self.x = self.x[:, 1:n]

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
                if i>5:
                    vals.append([float(i) for i in x])
                    depth.append(float(x[17]))
                i += 1
        # vals = np.array(vals)
        # for j in range(30):
        #     plt.subplot(6, 5, j+1)
        #     plt.title(j)
        #     plt.plot(vals[:, j])
        # plt.show()

        return depth

    def get_distance_travelled(self, file):
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            distance = [0.0]
            for row in csv_reader:
                distance.append(distance[-1]+float(row[62]))
        return distance

    def terminal_state(self):
        file = "data/winter/14-28-58.csv"
        y = []
        X = []
        for file in self.files:
            with open(file) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    i = 0
                    depth = self.read_depth(file)
                    work_done = 0
                    workdone_x = 0
                    workdone_y = 0
                    a_A = 0.0020
                    a_B = 0.0012
                    a = 0.0016
                    prev_boom = 0
                    F0 = 0
                    for row in csv_reader:
                        if (len(depth) > i):
                            if i<(len(depth)-64):
                                y.append(0)
                            else:
                                y.append(1)
                            P_A = float(row[28]) * 100000
                            P_B = float(row[27]) * 100000
                            boom = float(row[71])
                            bucket = float(row[72])
                            vx = float(row[62])
                            l = float(row[21])
                            F = a_A * P_A - a_B * P_B
                            if i == 0:
                                F0 = F
                            F -= F0
                            F_C = F * np.array([np.cos(boom), np.sin(boom)])
                            boom_dot = (boom - prev_boom) * 15
                            prev_boom = boom
                            v_C = np.array([vx - l * boom_dot * np.sin(boom) + a, l * boom_dot * np.cos(boom) + a])
                            work_done += abs(np.dot(F_C, v_C)) / 15
                            workdone_x += abs(F_C[0] * v_C[0]) / 15
                            workdone_y += abs(F_C[1] * v_C[1]) / 15
                            observation = [work_done,
                                           boom, bucket, depth[i]]
                            X.append(observation)
                            i += 1

        self.clf_binary = SVC()
        self.clf_binary.fit(X, y)
        mu_and_sigma = np.array(self.get_mean_and_variance(np.array(X)))
        self.the_stages.append(mu_and_sigma)

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
        plt.title(title)
        plt.plot(data[:, 1], 'b')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 1], 'r*')
        plt.ylabel('Workdone_x')

        plt.subplot(row, col, 3)
        plt.plot(data[:, 2], 'b')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 2], 'r*')
        plt.ylabel('Workdone_y')

        plt.subplot(row, col, 4)
        plt.plot(data[:, 3], 'm')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 3], 'r*')
        plt.ylabel('Boom Angle')

        plt.subplot(row, col, 5)
        plt.plot(data[:, 4], 'm')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 4], 'r*')
        plt.ylabel('Bucket angle')

        plt.subplot(row, col, 6)
        plt.plot(data[:, 5], 'g')
        if cluster_centers.any():
            plt.plot(js, cluster_centers[:, 5], 'r*')
        plt.ylabel('Distance to pile')


        plt.subplot(row, col, 7)
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
        cluster_centers, js = self.set_cluster_centers()
        cluster_centers = np.array(cluster_centers)
        clusters = KMeans(n_clusters=self.n_clusters, init=cluster_centers).fit(self.X)
        n = self.expert.shape[0]
        self.y = clusters.labels_
        self.X = np.vstack((self.X, self.x))
        n_x = self.x.shape[0]
        y_x = 3 * np.ones((n_x), dtype=int)
        self.y = np.hstack((self.y, y_x))
        y = np.array(self.y).reshape((n, 1))
        # a = 0
        # for i in range(len(self.data)):
        #     b = a + len(self.data[i])
        #     only_data = np.array(self.data[i])
        #     only_labels = y[a:b, :]
        #     data = np.hstack((only_data, only_labels))
        #     data = np.delete(data, 0, 1)
        #     self.plot_data(data, "Training", self.files[i])
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

        for stage in stages:
            mu_and_sigma = self.get_mean_and_variance(np.array(stage))
            self.the_stages.append(mu_and_sigma)
        self.terminal_state()
        self.the_stages = np.array(self.the_stages)

    def get_mean_and_variance(self, x):
        mu_x = np.mean(x, axis=0)
        sigma_x = np.std(x, axis=0)
        return np.array([mu_x, sigma_x])

    def reset(self):
        self.reward = 0

    def step_classifier(self):
        self.clf = KNeighborsClassifier()
        self.clf.fit(self.X, self.y)

    def get_intermediate_reward(self, state):
        n = len(state)-1
        segment = self.clf.predict([state])[0]
        terminal = self.clf_binary.predict([state])[0]
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
        reward_t = 100 - summed
        return reward_t, segment, terminal

    def combine_reward(self, reward_i, segment, time):
        # if (time>300):
        #     self.reward -= time*100

        if segment>0:
            # self.reward += reward_i*pow(2, segment-1)
            self.reward = reward_i * pow(2, segment - 1)
        else:
            self.reward = 0





