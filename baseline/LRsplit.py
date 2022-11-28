import os
import Augmentor
import scipy as sp
import numpy as np
import cv2
from PIL import Image


raw_path = "C:/milai/train/x"  # x 원본사진경로 파일경로 (개인 설정)
token_list = os.listdir(raw_path)
fldata_path = 'C:/milai/flipped X' # 좌우반전 파일 경로(만들어)
RRdata_path = 'C:/milai/roll right X' # 오른쪽 90도 회전 파일 경로(만들어)
RLdata_path = 'C:/milai/roll left X' # 왼쪽 90도 회전 파일 경로 (만들어)

for token in token_list:  # x 이미지 나누기
    image_path = raw_path + '/' + token
    flsave_path = fldata_path + '/' + "fl_" + token  # 수정한 사진 이름 변경
    RRsave_path = RRdata_path + '/' + "rr_" + token  # 수정한 사진 이름 변경
    RLsave_path = RLdata_path + '/' + "rl_" + token  # 수정한 사진 이름 변경

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h0, w0, c0 = img.shape

    # 좌우 사진 자르기
    LC = img[:, 0:(w0//2)].copy()
    RC = img[:, (w0//2):].copy()

    # 뒤집기
    LF = cv2.flip(LC, 1)
    RF = cv2.flip(RC, 1)
    FI = cv2.hconcat([LF, RF])
    cv2.imwrite(flsave_path, FI)
    print(flsave_path)

    # 잘린 왼쪽 사진을 우로 90도 회전 // img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 시계방향으로 90도 회전
    LRR_1 = cv2.rotate(LC, cv2.ROTATE_90_CLOCKWISE)
    RRR_1 = cv2.rotate(RC, cv2.ROTATE_90_CLOCKWISE)
    LRR = LRR_1[:, 0:(w0 // 2)]
    RRR = RRR_1[:, 0:(w0 // 2)]

    RR = cv2.hconcat([LRR, RRR])
    cv2.imwrite(RRsave_path, RR)
    print(RRsave_path)

    # 잘린 오른쪽 사진을 좌로 90도 회전 // img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 반시계방향으로 90도 회전
    LRL_1 = cv2.rotate(LC, cv2.ROTATE_90_COUNTERCLOCKWISE)
    RRL_1 = cv2.rotate(RC, cv2.ROTATE_90_COUNTERCLOCKWISE)
    LRL = LRL_1[:, 0:(w0 // 2)]
    RRL = RRL_1[:, 0:(w0 // 2)]

    RL = cv2.hconcat([LRL, RRL])
    cv2.imwrite(RLsave_path, RL)
    print(RLsave_path)

raw_path = "C:/milai/train/y"  # 원본사진경로
token_list = os.listdir(raw_path)
fldata_path = 'C:/milai/flipped y'  # x좌측사진경로  파일경로는 너가 수정하십쇼 마지막에 L_x라는 폴더 만들어야할거임
RRdata_path = 'C:/milai/roll right y'
RLdata_path = 'C:/milai/roll left y'

for token in token_list:  # x 이미지 나누기
    image_path = raw_path + '/' + token
    flsave_path = fldata_path + '/' + "fl_" + token  # 수정한 사진 이름 변경
    RRsave_path = RRdata_path + '/' + "rr_" + token  # 수정한 사진 이름 변경
    RLsave_path = RLdata_path + '/' + "rl_" + token  # 수정한 사진 이름 변경

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h0, w0, c0 = img.shape

    # 좌우 사진 자르기
    LC = img[:, 0:(w0 // 2)].copy()
    RC = img[:, (w0 // 2):].copy()

    # 뒤집기
    LF = cv2.flip(LC, 1)
    RF = cv2.flip(RC, 1)
    FI = cv2.hconcat([LF, RF])
    cv2.imwrite(flsave_path, FI)
    print(flsave_path)

    # 잘린 왼쪽 사진을 우로 90도 회전 // img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 시계방향으로 90도 회전
    LRR_1 = cv2.rotate(LC, cv2.ROTATE_90_CLOCKWISE)
    RRR_1 = cv2.rotate(RC, cv2.ROTATE_90_CLOCKWISE)
    LRR = LRR_1[:, 0:(w0 // 2)]
    RRR = RRR_1[:, 0:(w0 // 2)]

    RR = cv2.hconcat([LRR, RRR])
    cv2.imwrite(RRsave_path, RR)
    print(RRsave_path)

    # 잘린 오른쪽 사진을 좌로 90도 회전 // img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 반시계방향으로 90도 회전
    LRL_1 = cv2.rotate(LC, cv2.ROTATE_90_COUNTERCLOCKWISE)
    RRL_1 = cv2.rotate(RC, cv2.ROTATE_90_COUNTERCLOCKWISE)
    LRL = LRL_1[:, 0:(w0 // 2)]
    RRL = RRL_1[:, 0:(w0 // 2)]

    RL = cv2.hconcat([LRL, RRL])
    cv2.imwrite(RLsave_path, RL)
    print(RLsave_path)