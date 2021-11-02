import tensorflow as tf
import cv2
import numpy as np
import os
from detect_image import remove_img, detect_img
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
with open('model201224.json', 'r') as json_file:
  json_saved_model = json_file.read()
network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('weightsd201224.hdf5')
network_loaded.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


image = cv2.imread('people/3.png')


detect_img(image)
image_set = "face_detected"
no = 0
with_mask = 0
without_mask = 0
for img in os.listdir(image_set):
  if str(img).split('.')[1][0:3] == "jpg":
    image_path = os.path.join(image_set, img)
    image_ori = cv2.imread(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255
    image = image.reshape(-1, 224, 224, 3)
    result = network_loaded(image)
    # print(result)
    result_np = np.array(result)
    # have_mask100 = str(result_np[0][0]).split('.')[0]
    # no_mask100 = str(result_np[0][1]).split('.')[0]
    # have_mask = str(result_np[0][0]).split('.')[1][0:4]
    # no_mask = str(result_np[0][1]).split('.')[1][0:4]
    have_mask = str(result_np[0][0]*100)[:5]
    no_mask = str(result_np[0][1]*100)[:5]
    # print(have_mask, no_mask)
    result = np.argmax(result)
    no += 1
    if result == 0:
      with_mask += 1
      print("withmask")
      cv2.imshow(str(no) + ".image: " + have_mask +"% + with mask", image_ori)
      cv2.waitKey(0)
    else:
      without_mask += 1
      print("without_mask")
      cv2.imshow(str(no) + ".image: " + no_mask + "% - no mask", image_ori)
      cv2.waitKey(0)
list = os.listdir("face_detected")
num_img = [x for x in list if str(x).split('.')[1][0:4] == "jpg"]
print(len(num_img))
remove_img()
print("Total with mask: ", with_mask)
print("Total without mask: ", without_mask)
