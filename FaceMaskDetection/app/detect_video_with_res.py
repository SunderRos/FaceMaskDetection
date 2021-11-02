from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import os
import imutils
import cv2
import tensorflow as tf
import time
import threading
import IPython
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
with open('network128v4resfinal.json', 'r') as json_file:
  json_saved_model = json_file.read()

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('weights128v4resfinal.hdf5')
network_loaded.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
threshold = 0.5

prototxt_file = file_path + 'Resnet_SSD_deploy.prototxt'
caffemodel_file = file_path + 'Res10_300x300_SSD_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)


# size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out_fps = 20
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter()
# out_path = file_path+'test_out'+os.sep+'example.mp4'
# writer.open(out_path, fourcc, out_fps, size, True)

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS, 20)
fps = FPS().start()

faces = []
vs = VideoStream(src=0).start()
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


while True:
    # ret, frame = cap.read()
    # if not ret: break
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    origin_h, origin_w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    num = 0
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
            x_start, y_start, x_end, y_end = bounding_box.astype('int')
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
            # roi_color = frame[y_start:y_end, x_start:x_end]
            (x_start, y_start) = (max(0, x_start), max(0, y_start))
            (x_end, y_end) = (min(origin_w - 1, x_end), min(origin_h - 1, y_end))

            face = frame[y_start:y_end, x_start:x_end]

            if face.any():
                face = cv2.resize(face, (128, 128))
                face = face / 255
                face = face.reshape(-1, 128, 128, 3)
                faces.append(face)
            # if face.size != 0:
            #     cv2.imwrite("face_detected_video/" + str(num) + '_faces.jpg', face)
            if len(faces) > 0:
                for face in faces:
                    # faces = np.array(faces, dtype="float32")
                    pred = network_loaded(face)
                    pred_np = np.array(pred)
                    have_mask = str(pred_np[0][0] * 100)[:5]
                    no_mask = str(pred_np[0][1] * 100)[:5]
                    # have_mask = str(pred_np[0][0]).split('.')[1][0:4]
                    # no_mask = str(pred_np[0][1]).split('.')[1][0:4]
                    pred = np.argmax(pred)
                    labels, color = ("With Mask: " + have_mask + "%", (0, 255, 0)) if pred == 0 else ("No Mask: " + no_mask + "%", (0, 0, 255))
                    cv2.putText(frame, labels, (x_start+2, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)



            label = '{0:.2f}%'.format(confidence * 100)
            # cv2.rectangle(frame, (x_start, y_start), (x_end, y_end),(0, 0, 255), 2)
            # cv2.rectangle(frame, (x_start, y_start), (x_end, y_start), (0, 0, 255),q -1)
            # cv2.putText(frame, label, (x_start+2, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    faces = []
    fps.update()
    fps.stop()
    text = "FPS: {:.2f}".format(fps.fps())
    cv2.putText(frame, text, (15, int(origin_h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
writer.release()
# cap.release()
cv2.destroyAllWindows()