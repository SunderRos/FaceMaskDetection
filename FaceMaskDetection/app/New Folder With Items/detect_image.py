import dlib
import cv2
import glob
import os


image = cv2.imread('people/gabriel.png')


cnn_detector = dlib.cnn_face_detection_model_v1('./mmod_human_face_detector.dat')
detections = cnn_detector(image, 1)

def detect_img(img):
    num = 0
    for face in detections:
        num += 1
        l, t, r, b, c = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom(), face.confidence
        print(c)
        cv2.rectangle(image, (l, t), (r, b), (0, 0, 255), 2)
        roi_color = image[t:b, l:r]
        if roi_color.size != 0:
            cv2.imwrite("face_detected/" + str(num) + '_faces.jpg', roi_color)
detect_img(image)
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
folder_path = "face_detected"
def remove_img():
    for filename in os.listdir('./face_detected'):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)

# remove_img()
# pirnt("done")
# for i in range(len(detections)):
#     (x, y, w, h) = detections[i]
#     crop = new_img[y:y + h, x:x + w]
#     crop = cv2.resize(crop, (128, 128))
#     crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
#     mask_result = model.predict(crop)
#     cv2.putText(new_img, mask_label[mask_result.argmax()], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 dist_label[label[i]], 2)
#     cv2.rectangle(new_img, (x, y), (x + w, y + h), dist_label[label[i]], 1)
# plt.figure(figsize=(10, 10))
# plt.imshow(new_img)