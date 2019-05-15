import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

count = 0
temp = 0
currdict = os.getcwd()

# Tao thu muc data
if not os.path.exists("dataset/Nam_0-2"):
    os.makedirs("dataset/Nam_0-2")
if not os.path.exists("dataset/Nam_4-6"):
    os.makedirs("dataset/Nam_4-6")
if not os.path.exists("dataset/Nam_8-12"):
    os.makedirs("dataset/Nam_8-12")
if not os.path.exists("dataset/Nam_15-20"):
    os.makedirs("dataset/Nam_15-20")
if not os.path.exists("dataset/Nam_25-32"):
    os.makedirs("dataset/Nam_25-32")
if not os.path.exists("dataset/Nam_38-43"):
    os.makedirs("dataset/Nam_38-43")
if not os.path.exists("dataset/Nam_48-53"):
    os.makedirs("dataset/Nam_48-53")
if not os.path.exists("dataset/Nam_60-100"):
    os.makedirs("dataset/Nam_60-100")
    
if not os.path.exists("dataset/Nu_0-2"):
    os.makedirs("dataset/Nu_0-2")
if not os.path.exists("dataset/Nu_4-6"):
    os.makedirs("dataset/Nu_4-6")
if not os.path.exists("dataset/Nu_8-12"):
    os.makedirs("dataset/Nu_8-12")
if not os.path.exists("dataset/Nu_15-20"):
    os.makedirs("dataset/Nu_15-20")
if not os.path.exists("dataset/Nu_25-32"):
    os.makedirs("dataset/Nu_25-32")
if not os.path.exists("dataset/Nu_38-43"):
    os.makedirs("dataset/Nu_38-43")
if not os.path.exists("dataset/Nu_48-53"):
    os.makedirs("dataset/Nu_48-53")
if not os.path.exists("dataset/Nu_60-100"):
    os.makedirs("dataset/Nu_60-100")


with open('data.txt', 'r') as rawdata:
    csv_reader = csv.reader(rawdata, delimiter='\t')
    for row in csv_reader:
        print(temp)
        if count == 0 or temp == 4485 or temp == 8216 or temp == 12111 or temp == 15558:
            count += 1
        else:
            #Lấy các cột dữ liệu
            face_id = row[2]
            image_name = row[1]
            folder_name = row[0]
            age = row[3]
            gender = row[4]

            #Load ảnh
            image_path = currdict + '/aligned/' + folder_name + '/landmark_aligned_face.' + str(face_id) + '.' + image_name
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)

            #Tiến hành phân loại ảnh
            if age == '(0, 2)':
                if gender == 'm':
                    os.chdir(currdict + '/dataset/Nam_0-2')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
                elif gender == 'f':
                    os.chdir(currdict + '/dataset/Nu_0-2')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
            if age == '(4, 6)':
                if gender == 'm':
                    os.chdir(currdict + '/dataset/Nam_4-6')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
                elif gender == 'f':
                    os.chdir(currdict + '/dataset/Nu_4-6')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
            if age == '(8, 12)':
                if gender == 'm':
                    os.chdir(currdict + '/dataset/Nam_8-12')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
                elif gender == 'f':
                    os.chdir(currdict + '/dataset/Nu_8-12')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
            if age == '(15, 20)':
                if gender == 'm':
                    os.chdir(currdict + '/dataset/Nam_15-20')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
                elif gender == 'f':
                    os.chdir(currdict + '/dataset/Nu_15-20')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
            if age == '(25, 32)':
                if gender == 'm':
                    os.chdir(currdict + '/dataset/Nam_25-32')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
                elif gender == 'f':
                    os.chdir(currdict + '/dataset/Nu_25-32')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
            if age == '(38, 43)':
                if gender == 'm':
                    os.chdir(currdict + '/dataset/Nam_38-43')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
                elif gender == 'f':
                    os.chdir(currdict + '/dataset/Nu_38-43')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
            if age == '(48, 53)':
                if gender == 'm':
                    os.chdir(currdict + '/dataset/Nam_48-53')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
                elif gender == 'f':
                    os.chdir(currdict + '/dataset/Nu_48-53')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
            if age == '(60, 100)':
                if gender == 'm':
                    os.chdir(currdict + '/dataset/Nam_60-100')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
                elif gender == 'f':
                    os.chdir(currdict + '/dataset/Nu_60-100')
                    cv2.imwrite(folder_name + '.' + face_id + '.' + image_name, img)
        temp += 1
        print('------------------------'+str(temp))