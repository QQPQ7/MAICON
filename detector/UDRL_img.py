import os
import cv2
# import numpy as np

raw_path = "F:/aiai/baseline/data/train/x"
token_list = os.listdir(raw_path)
data_path = "F:/aiai/baseline/data/train/x"

for token in token_list:
#원본 이미지 경로와 저장할 경로 이미지 지정
    image_path = raw_path + '/' +token 
    save_path = data_path + '/' + "RL_" + token

    image = cv2.imread(image_path)

    image = cv2.flip(image, 1) # 1은 좌우 반전, 0은 상하 반전
    cv2.imwrite(save_path, image)
    print(save_path)

for token in token_list:
#원본 이미지 경로와 저장할 경로 이미지 지정
    image_path = raw_path + '/' +token
    save_path = data_path + '/' + "UD_" + token

    image = cv2.imread(image_path)

    image = cv2.flip(image, 0) # 1은 좌우 반전, 0은 상하 반전
    cv2.imwrite(save_path, image)
    print(save_path)

raw_path = "F:/aiai/baseline/data/train/y"
token_list = os.listdir(raw_path)
data_path = 'F:/aiai/baseline/data/train/y'

for token in token_list:
#원본 이미지 경로와 저장할 경로 이미지 지정
    image_path = raw_path + '/' +token 
    save_path = data_path + '/' + "RL_" + token

    image = cv2.imread(image_path)

    image = cv2.flip(image, 1) # 1은 좌우 반전, 0은 상하 반전
    cv2.imwrite(save_path, image)
    print(save_path)

for token in token_list:
#원본 이미지 경로와 저장할 경로 이미지 지정
    image_path = raw_path + '/' +token
    save_path = data_path + '/' + "UD_" + token

    image = cv2.imread(image_path)

    image = cv2.flip(image, 0) # 1은 좌우 반전, 0은 상하 반전
    cv2.imwrite(save_path, image)
    print(save_path)