import cv2
import numpy as np
import tensorflow as tf


def make_average_predictions(video_file_path, predictions_frames_count, model_output_size, image_height, image_width, model, classes_list):   
    predicted_labels_probabilities_np = np.zeros((predictions_frames_count, model_output_size), dtype = np.float)
    video_reader = cv2.VideoCapture(video_file_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = video_frames_count // predictions_frames_count
    for frame_counter in range(predictions_frames_count): 
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        _ , frame = video_reader.read() 
        resized_frame = cv2.resize(frame, (image_height, image_width))       
        normalized_frame = resized_frame / 255
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities
    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(predicted_labels_probabilities_averaged)[::-1]
    pred_class=[]
    pred_prob=-100
    for predicted_label in predicted_labels_probabilities_averaged_sorted_indexes:
        predicted_class_name = classes_list[predicted_label]
        predicted_probability = predicted_labels_probabilities_averaged[predicted_label]
        print(f"CLASS NAME: {predicted_class_name}   AVERAGED PROBABILITY: {(predicted_probability*100):.2}")
        if predicted_probability>pred_prob:
          pred_class=predicted_class_name
          pred_prob=predicted_probability

    print(" ")   
    print(" ")
    print("Final Prediction : ",pred_class," & Probability : ",round(pred_prob,4))
    video_reader.release()


model = tf.keras.models.load_model('HAR_Model_Acc_98.73.h5') # Loading the pre-trained model

classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
model_output_size = len(classes_list)

image_height,image_width = 64,64

input_video_file_path = 'demo/input.mp4'

make_average_predictions(input_video_file_path, 50, model_output_size, image_height, image_width, model, classes_list)
