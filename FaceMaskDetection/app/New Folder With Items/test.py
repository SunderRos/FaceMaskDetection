from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import os
import cv2


file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
threshold = 0.5

prototxt_file = file_path + 'Resnet_SSD_deploy.prototxt'
caffemodel_file = file_path + 'Res10_300x300_SSD_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_file, caffeModel=caffemodel_file)

cap = cv2.VideoCapture(0)
time.sleep(1.0)
fps = FPS().start()

size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out_fps = 20
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter()
out_path = file_path+'test_out'+os.sep+'example.mp4'
writer.open(out_path, fourcc, out_fps, size, True)

while True:
    _, frame = cap.read()
    origin_h, origin_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            bounding_box = detections[0, 0, i, 3:7] * np.array([origin_w, origin_h, origin_w, origin_h])
            x_start, y_start, x_end, y_end = bounding_box.astype('int')

            label = '{0:.2f}%'.format(confidence * 100)
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end),(0, 0, 255), 2)
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_start), (0, 0, 255), -1)
            # cv2.putText(frame, label, (x_start+2, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    fps.update()
    fps.stop()
    text = "FPS: {:.2f}".format(fps.fps())
    cv2.putText(frame, text, (15, int(origin_h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
writer.release()
cap.release()
cv2.destroyAllWindows()