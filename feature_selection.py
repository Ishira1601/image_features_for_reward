import numpy as np
import csv
from upr import UPR
from sklearn.cluster import KMeans

def get_file_paths(folders):
    from os import listdir
    from os.path import isfile, join

    file_paths = []
    for folder in folders:
        onlyfiles = [folder+"/"+f for f in listdir(folder) if isfile(join(folder, f))]
        file_paths += onlyfiles
    return file_paths

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
    depth = read_depth(file)
    season = file.split("/")[1] 
    all_data = []
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

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

                j = 0
                data = []
                for m in row:
                    if j<76 and j!=8 and j!=11 and j!=18 and j!=22 and j!=31 and j!=34 and j!=46 and j!=55 and j!=42 and j!=50:
                        data.append(float(m))
                    j+=1
                data.append(work_done)
                data.append(depth[i])
                all_data.append(data)
                i+=1
                
    return all_data

def multiple_files(files):
    from matplotlib import pyplot as plt
    vis_model = get_visual_model()
    X = []
    times = []
    time = 0
    for file in files:
        all_data = one_file(file)
        time = len(all_data)
        times.append(time)


        file_time = file.split("/")[2].split(".")[0]

        frames = get_frames(file_time)
        im_feature, score = vis_model.predict(frames)

        if im_feature.shape[0]>time:
            im_feature = im_feature[0:time, :]
        else:
            all_data = all_data[0:im_feature.shape[0]]
        all_data = np.array(all_data)
        # all_data_and_imgF = np.hstack((all_data, im_feature))
        all_data_and_imgF = im_feature
        if X==[]:
            X =  all_data_and_imgF
        else:
            X = np.vstack((X, all_data_and_imgF))
        print(all_data_and_imgF)
    T = times[0]

    the_stages, y = stages(X, 3, T)

    prev_time = 0
    for time in times:
        new_time = prev_time+time
        plt.plot(y[prev_time:new_time])
        prev_time = new_time
        plt.show()

    n = the_stages.shape[2]

    feature_scores = []
    segment_scores = []
    for i in range(n):
        score, scores = get_feature_score(5, the_stages, i)
        feature_scores.append(score)
        segment_scores.append(scores)


    feature_scores = np.array(feature_scores)

    features = np.argsort(-1 * feature_scores)
    print(features)
    segment_scores = np.array(segment_scores)

    features = np.argsort(-1*segment_scores, axis=0)
    print(features)




def read_depth(file):
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

    return depth

def stages(X, n_clusters, T):
    cluster_centers, js = set_cluster_centers(n_clusters, X, T)
    cluster_centers = np.array(cluster_centers)
    clusters = KMeans(n_clusters=n_clusters, init=cluster_centers).fit(X)
    y = clusters.labels_
    n = len(X)
    y = np.array(y).reshape((n, 1))
    the_stages = stages_mean_and_variance(y, X)

    return the_stages, y

def set_cluster_centers(n_clusters, expert, T):
    i = round(T / (2 * (n_clusters-1)))
    cluster_centers = []
    k = 1
    j = k * i
    js= []
    while j<T:
        cluster_centers.append(expert[j])
        js.append(j)
        k += 2
        j = k * i
    cluster_centers.append(expert[T-8])
    return cluster_centers, js

def stages_mean_and_variance(y, samples):
    n = len(y)
    stages = []
    for i in range(n):
        if stages == []:
            stages.append([samples[i]])
        elif len(stages) > y[i]:
            stages[y[i][0]].append(samples[i])
        else:
            stages.append([samples[i]])
    the_stages = []
    for stage in stages:
        mu_and_sigma = get_mean_and_variance(np.array(stage))
        the_stages.append(mu_and_sigma)
    # self.terminal_state()
    the_stages = np.array(the_stages)
    return the_stages

def get_mean_and_variance(x):
    mu_x = np.mean(x, axis=0)
    sigma_x = np.std(x, axis=0)
    return np.array([mu_x, sigma_x])

def get_feature_score(alpha, the_stages, feature):
    n = the_stages.shape[0]
    total_score = 0
    scores = []
    for segment in range(n):
        mu_poitive = the_stages[segment, 0, feature]
        sigma_positive = the_stages[segment, 1, feature]

        mu_negatives = []
        sigma_negatives = []

        for i in range(n):
            if i!=segment:
                mu_negatives.append(the_stages[i, 0, feature])
                # sigma_negatives.append(the_stages[i, 1, feature])

        mu_negative = np.mean(np.array(mu_negatives))
        sigma_negative = np.std(np.array(mu_negatives))

        total_score += (alpha * abs(mu_poitive-mu_negative) - (sigma_positive+sigma_negative))
        scores.append((alpha * abs(mu_poitive-mu_negative) - (sigma_positive+sigma_negative)))
    return total_score, scores

def get_visual_model():
    from T3D_keras import densenet161_3D_DropOut, densenet121_3D_DropOut, xception_classifier, c3d_model, \
        c3d_model_feature
    from tensorflow.keras.optimizers import Adam
    import os

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

def get_frames(time):
    import cv2
    k = 2
    # k = 24
    frames = []

    flag = True

    while flag:
        window = []
        for i in range(k, k+5):
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
        k+=1

    frames = np.array((frames))
    return frames

files = get_file_paths(["data/autumn", "data/winter"])
multiple_files(files)