import cv2
from imutils.video import FPS
import time
cap = cv2.VideoCapture('video/1.mp4')
cap.set(cv2.CAP_PROP_FPS, 20)
fps = FPS().start()
writer = cv2.VideoWriter()
frame_rate = 10
prev = 0
while True:
  success, image = cap.read()
  origin_h, origin_w = image.shape[:2]
  fps.update()
  fps.stop()
  text = "FPS: {:.2f}".format(fps.fps())
  cv2.putText(image, text, (15, int(origin_h * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  time_elapsed = time.time() - prev
  if time_elapsed > 1. / frame_rate:
    prev = time.time()
  cv2.imshow("image", image)
  writer.write(image)
  if cv2.waitKey(1) == ord('q'):
    break