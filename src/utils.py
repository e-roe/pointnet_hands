import cv2
import mediapipe as mp
import numpy as np
import yaml
import os
import shutil

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


char2int = {
            "a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "k":9, "l":10, "m":11,
            "n":12, "o":13, "p":14, "q":15, "r":16, "s":17, "t":18, "u":19, "v":20, "w":21, "x":22, "y":23
            }

def get_hand_points(img):
    results = hands.process(img)
    points = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                points.append([x, y, z])
    else:
        border_size = 100
        img = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        results = hands.process(img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    points.append([x, y, z])
        else:
            points = None

    if points is not None:
        if len(points) > 21:
            points = points[:21]
        elif len(points) < 21:
            dif = 21 - len(points)
            for i in range(dif):
                points.append([0, 0, 0])

        points = np.array(points)

    return points


def clean_folder(folder):
    shutil.rmtree(folder)


def check_dataset(root):
    width = 400
    height = 400
    for root, dirs, files in os.walk(root):
        for file in files:
            img = np.zeros([height, width, 3], dtype=np.uint8)
            img.fill(255)
            print(file)
            points = np.load(os.path.join(root, file))
            for pp in mp_hands.HAND_CONNECTIONS:
                cv2.line(img, (int((points[pp[0]][0]) * width), int((points[pp[0]][1]) * height)),
                         (int((points[pp[1]][0]) * width), int((points[pp[1]][1]) * height)), (0, 0, 255), 4)
            cv2.imshow('', img)
            cv2.waitKey(0)


def load_config(config_file, config_dir='../configs'):
    config_file = os.path.join(config_dir, config_file)
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    config = load_config('config.yaml')
    print(config)
    dataset = config['dataset']['augmentations']
    name = config['model']['name']
    n_epochs = 10
    print(name)
