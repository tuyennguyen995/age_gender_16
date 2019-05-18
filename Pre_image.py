from __future__ import absolute_import, division, print_function

# Thêm thư viện TensorFlow và tf.keras
import tensorflow
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
#Thư viện open file
from tkinter import filedialog
from tkinter import *

KEY_ESCAPE = 27

# Phat hien khuon mat
def detect_face_open_cv_dnn(net, frame, conf_threshold=0.5):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    ajust = 0.17
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            ajust_x = int((x2 - x1) * ajust)
            ajust_y = int((y2 - y1) * ajust)
            x1 -= ajust_x
            x2 += ajust_x
            y1 -= int(ajust_y * 1.5)
            y2 += int(ajust_y * 0.5)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 350)), 8)
    return frame_opencv_dnn, bboxes


# Load json và tạo model
if __name__ == "__main__":

    # Load model train
    json_file = open("models/train/model.json", "r")
    json_string = json_file.read()
    json_file.close()
    model = model_from_json(json_string)
    model.load_weights("models/train/weights.30.h5")
    names = ["Nam (0-2)", "Nam (15-20)", "Nam (25-32)", "Nam (38-43)", "Nam (4-6)", "Nam (48-53)", "Nam (60-100)", "Nam (8-12)",
             "Nu (0-2)", "Nu (15-20)", "Nu (25-32)", "Nu (38-43)", "Nu (4-6)", "Nu (48-53)", "Nu (60-100)", "Nu (8-12)"]

    # Khoi tao moduls phat hien khuon mat
    model_file = "models/face/opencv_face_detector_uint8.pb"
    config_file = "models/face/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    while (True):
        #Chọn file ảnh
        root = Tk()
        root.withdraw()
        root.filename = filedialog.askopenfilename(initialdir = "/",title = "Chọn file ảnh",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

        # Xử lý ảnh đầu vào
        img_load = cv2.imread(root.filename)  # load ảnh

        w = int(np.shape(img_load)[0] * (1280 / np.shape(img_load)[0]))
        h = int(np.shape(img_load)[1] * (1280 / np.shape(img_load)[0]))
        img_load = cv2.resize(img_load, (h, w))
        print("Ảnh đã resize w: " + str(np.shape(img_load)))

        if np.shape(img_load)[1] > 720:
            w = int(np.shape(img_load)[0] * (720 / np.shape(img_load)[1]))
            h = int(np.shape(img_load)[1] * (720 / np.shape(img_load)[1]))
            img_load = cv2.resize(img_load, (h, w))
            print("Ảnh đã resize h: " + str(np.shape(img_load)))

        plt.imshow(img_load)
        plt.show()
        out_opencv_dnn, bboxes = detect_face_open_cv_dnn(net, img_load)

        for x1, y1, x2, y2 in bboxes:
            try:
                # Lấy khung ảnh
                a = int((x2-x1)*0.1)
                b = a*2
                dd2 = ((y2 - y1) - (x2 - x1))+int(a*0.65)
                dd1 = int(dd2 / 2)+a
                img = cv2.resize(img_load[y1 - a:y2 + b, x1 - dd1:x2 + dd2], (128, 128))

            except Exception as e:
                print('Không nhận dạng được đối tượng')
                dd2 = int(((y2 - y1) - (x2 - x1))*0.8)
                dd1 = int(dd2 / 2)
                img = cv2.resize(img_load[y1:y2,  x1 - dd1:x2 + dd2], (128, 128))
                plt.imshow(img)
                plt.show()

            #Tiền sử lý
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_arr = img_to_array(img)/255
            img_arr = img_arr.reshape((1,) + img_arr.shape)

            #Dự đoán
            prediction = model.predict(img_arr)[0]

            pos = np.argmax(prediction)

            #Tính độ chính xác
            max_percent = np.amax(prediction) * 100

            #Tính tỷ lệ khung Box
            face_width = (x2 - x1) / 260
            box_width = int(face_width * 20)
            marin = int(face_width * 10)

            #Hiển thị khung Box
            cv2.rectangle(out_opencv_dnn, (x1 + marin, y1 - box_width + (y2-y1)), (x2 - marin, y1 + (y2-y1)), (0, 255, 0), box_width)
            cv2.putText(out_opencv_dnn, "  {}".format(names[pos]), (x1, y2),
                          cv2.FONT_HERSHEY_SIMPLEX, face_width, (255, 255, 255), 1, cv2.LINE_4)

        cv2.imshow("Image out", out_opencv_dnn)
        #cv2.show()
        root.destroy()

        # Bat su kien phim
        key = cv2.waitKey(60)
        if key == KEY_ESCAPE:
            break
