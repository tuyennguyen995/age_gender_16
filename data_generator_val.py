import os
import cv2
from keras.preprocessing.image import img_to_array, ImageDataGenerator

# Khai bao bien mac dinh
TEMP = 1

datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             shear_range=0.01,
                             horizontal_flip=True,
                             brightness_range=[0.5, 1.5])

datagen_src = ImageDataGenerator()

# Doc du lieu
data_path = "data_process/validation"
for dir_name in os.listdir(data_path):
    tmp_path = "{}/{}".format(data_path, dir_name)
    if not os.path.isdir(tmp_path):
        continue

    # Tao thu muc, kiem tra
    result_path = "data_train/validation/{}".format(dir_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #Lay duong dan file
    for file_name in os.listdir(tmp_path):
        file_path = "{}/{}".format(tmp_path, file_name)
        if not os.path.isfile(file_path):
            continue

        img_load = cv2.imread(file_path)
        img = cv2.cvtColor(img_load, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        for batch in datagen_src.flow(img_array,
                                  batch_size=1,
                                  save_to_dir=result_path,
                                  save_prefix=os.path.splitext(file_name),
                                  save_format='jpg'):
            break

        counter = 0
        # for batch in datagen.flow(img_array,
        #                           batch_size=1,
        #                           save_to_dir=result_path,
        #                           save_prefix=os.path.splitext(file_name)[0],
        #                           save_format='jpg'):
        #     counter += 1
        #     if counter >= 2:
        #         break

        print(str(TEMP))
        TEMP+=1

