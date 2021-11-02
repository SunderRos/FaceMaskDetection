import tensorflow as tf
import cv2
import numpy as np
import os
import dlib
import glob
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('network128v4resfinal.json', 'r') as json_file:
  json_saved_model = json_file.read()
network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('weights128v4resfinal.hdf5')
network_loaded.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


image = cv2.imread('people/8.jpeg')

cnn_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')

detections = cnn_detector(image, 1)
for face in detections:
  x, y, w, h, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
  crop = image[y:h, x:w]
  if crop.size != 0:
    crop = cv2.resize(crop, (128, 128))
    crop = crop / 255
    crop = crop.reshape(-1, 128, 128, 3)
    result = network_loaded(crop)
    # print(result)
    result_np = np.array(result)
    have_mask = str(result_np[0][0]*100)[:5]
    no_mask = str(result_np[0][1]*100)[:5]
    # have_mask = str(result_np[0][0]).split('.')[1][0:4]
    # no_mask = str(result_np[0][1]).split('.')[1][0:4]

    # print(have_mask, no_mask)
    result = np.argmax(result)
    labels, color = ("With Mask: ", (0, 255, 0)) if result == 0 else ( "No Mask: ", (0, 0, 255))
    cv2.putText(image, labels, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
    cv2.rectangle(image, (x, y), (w, h), color, 2)
cv2.imshow("Mask Image Detection", image)
cv2.waitKey(0)
