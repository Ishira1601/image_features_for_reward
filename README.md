# To run test.py
1. Create a data folder to store data `mkdir data`
2. Add the following folders to the data folder
- "autumn" folder: holds autumn csvs (files named as time.csv ...i.e. 14-28-58.csv)
- "winter" folder: holds winter csvs
- "autumn_depth" folder: holds autumn depth files (files named as time.svo-depth.txt ...i.e. 14-28-58.svo-depth.txt)
- "winter_depth" folder: holds winter depth files

## To convert SVO to Image file 
3. Convert svo file to pngs `./svo2png.py <path_to_svo> <path_to_image_folder>`
- If svo file name is "time.svo" then images will be time_timestep.png where timestep is 0-N
- i.e. for time= 14-28-58
14-28-58_0.png, 14-28-58_1.png, 14-28-58_2.png

## Add a column to csv (column #82)
4. Ground truth for terminal state
- 0 - not terminal state
- 1 - terminal state

## Set up training and testing sets
5. Split training and testing
   `file_paths = get_file_paths(["data/winter"])`
   `X_train, X_test = training_test_split(file_paths)`

## To setup Unsupervised Perceptual Reward 
6. Train Data set
`R_max = 1600
upr = UPR(X_train, image_folder, n_clusters=3, R_max=R_max)`
- X_train: A list of csv files to train
- image_folder: folder containing images related to X_Train
- n_clusters: Number of clusters/stages
- R_max: Maximum possible intermediate reward

7. You should be able to run `test.py` now

# To run one file
The following function is used to test one data set
`data, total, yes, im_feature, images = one_file(file, image_folder, upr, vis_model, total, yes)`
- data: ndarray(n_timesteps, n_features)
- im_feature: ndarray containing 8 image features
- images: list of images as ndarray(112, 112, 3)
- file: name of csv file
- image_folder: name of folder containing images related to file
- upr: Unsupervised Perceptual Reward instance
- vis_model: 3D CNN
- total: total observations to that point (to calculate accuracy)
- yes: total correctly classified task completion states to that point (to calculate accuracy)


# Important Functions
The functions to follow are important when applying to real data

## To populate feature vector
`observation, F0 = upr.get_feature_vector(sensor_values, season, distance,
                                                  i, prev_boom, prev_work_done, F0)`

- observation: feature vector containing [workdone, boom angle, bucket angle, distance to the pile]
- sensor_values: list containing [telescopic pressure A, telescopic pressure B, 
                  boom angle, bucket angle, forward speed, boom length]
- season: autumn/winter
- distance: distance to the pile
- i: time step
- prev_boom: previous boom angle
- prev_work_done: previous work done
- F0: initial force 

## To calculate reward and obtain task completion state
`reward, segment, terminal = upr.get_reward(observation, im_feature)`
- reward: the reward for the observation
- segment: stage for the observation
- terminal: (0/1) is the task completed
- observation: feature vector containing [workdone, boom angle, bucket angle, distance to the pile]
- im_feature: ndarray containing 8 image features



                                  



