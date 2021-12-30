import os 
import cv2
import random 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from model import CNNmodel

def plot_metric(metric_name_1, metric_name_2,plot_name):
    metric_value_1=model_training_history.history[metric_name_1]
    metric_value_2=model_training_history.history[metric_name_2]
    epochs = range(len(metric_value_1))
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)
    plt.title(str(plot_name))
    plt.legend()

def frames_extraction(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    while True:
        success,frame = video_reader.read()
        if not success:
            break
        resized_frame = cv2.resize(frame,(image_height,image_width))
        normalized_frame = resized_frame/255
        frames_list.append(normalized_frame)
    video_reader.release()
    return frames_list

def create_dataset():
    temp_features = [] 
    features = []
    labels = []
    for class_index, class_name in enumerate(classes_list):
        print(f'Extracting Data of Class: {class_name}')
        files_list = os.listdir(os.path.join(dataset_directory, class_name))
        for file_name in files_list:
            video_file_path = os.path.join(dataset_directory, class_name, file_name)
            frames = frames_extraction(video_file_path)
            temp_features.extend(frames)
        features.extend(random.sample(temp_features, max_images_per_class))
        labels.extend([class_index] * max_images_per_class)
        temp_features.clear()

    features = np.asarray(features)
    labels = np.array(labels)  

    return features, labels

seed_constant = 23
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

image_height,image_width = 64,64
max_images_per_class = 8000
dataset_directory = 'UCF50'
classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
model_output_size = len(classes_list)

features,labels = create_dataset()
one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, one_hot_encoded_labels, test_size = 0.2, shuffle = True, random_state = seed_constant)

model = CNNmodel(model_output_size,image_height,image_width)
plot_model(model,to_file='model_structure.png',show_shapes = True, show_layer_names = True)

# Training
early_stopping_callback = EarlyStopping(monitor = 'val_loss',patience = 15, mode = 'min',restore_best_weights=True)
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=["accuracy"])
model_training_history = model.fit(x = features_train, y = labels_train, epochs = 40, batch_size = 4 , shuffle = True, validation_split = 0.2, callbacks = [early_stopping_callback])

# Evaluation
model_evaluation_history = model.evaluate(features_test, labels_test)

# Save Model
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
model_name = f'HAR_Model_Acc_{round(model_evaluation_accuracy*100,2)}.h5'
model.save(model_name)

# Plotting the Total Loss vs Total Validation Loss
plot_metric('loss', 'val_loss', 'Total Loss vs Total Validation Loss')

# Plotting the Total Accuracy vs Total Validation Accuracy
plot_metric('accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')




