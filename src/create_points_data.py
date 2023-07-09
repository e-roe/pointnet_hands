import os
import random
import cv2
import glob
from tqdm import tqdm
from transforms_3d import Scale, Rotate, GaussianNoise
import mediapipe as mp
import numpy as np
from utils import get_hand_points, load_config

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def create(name, root, label, dest_path, augmentations):
    count = 0
    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(os.path.join(dest_path, label), exist_ok=True)
    angles = [ang for ang in range(-15, 15, 1)]
    scales = [s for s in np.arange(1.0, 0.7, -0.05)]
    rot_axes = ['y', 'z']
    scale_axes = ['y', 'x']
    img = cv2.imread(os.path.join(root, name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    points_raw = get_hand_points(img)
    if points_raw is not None:
        min_x = np.min(points_raw[:, 0])
        max_x = np.max(points_raw[:, 0])
        min_y = np.min(points_raw[:, 1])
        max_y = np.max(points_raw[:, 1])
        for i in range(len(points_raw)):
            points_raw[i][0] = (points_raw[i][0] - min_x) / (max_x - min_x)
            points_raw[i][1] = (points_raw[i][1] - min_y) / (max_y - min_y)

        np.save(os.path.join(dest_path, label, name.split('.')[0]), points_raw)
        count += 1
        if 'rot' in augmentations:
            for i in range(9):
                axis = rot_axes[random.randint(0, len(rot_axes) - 1)]
                index = random.randint(0, len(angles) - 1)
                angle = angles[index]
                angles.remove(angle)
                rotate_3d = Rotate(axis='y', angle=angle, prob=1)
                rot = rotate_3d(points_raw.copy())

                np.save(os.path.join(dest_path, label, name.split('.')[0] + f'_Rot_{angle}_Axis_{axis}'), rot)
                count += 1
        if 'scale' in augmentations:
            for i in range(4):
                axis = scale_axes[random.randint(0, len(scale_axes) - 1)]
                index = random.randint(0, len(scales) - 1)
                factor = scales[index]
                scales.remove(factor)
                scale = Scale(axis=axis, factor=factor, prob=1)
                scaled = scale(points_raw.copy())
                np.save(os.path.join(dest_path, label, name.split('.')[0] + f'_Scale_{angle}_Axis_{axis}'), scaled)
                count += 1
        if 'noise' in augmentations:
            for i in range(4):
                amount = 1 / random.randint(1000, 2000)
                scale = GaussianNoise(amount=amount)
                scaled = scale(points_raw.copy())
                np.save(os.path.join(dest_path, label, name.split('.')[0] + f'_Noise_{amount}'), scaled)
                count += 1

    return count


def main():
    config = load_config('config.yaml')
    count = 0
    path = config['dataset']['image_dataset']
    destination_path = config['dataset']['npy_dataset']
    augmentations = config['dataset']['augmentations']
    jpg_files = glob.glob(os.path.join(path, '**/*.jp*'), recursive=True)
    png_files = glob.glob(os.path.join(path, '**/*.png'), recursive=True)
    progress_bar = tqdm(total=len(jpg_files)+len(png_files))
    progress_bar.set_description("Creating dataset")
    to_discard = config['dataset']['to_discard']
    for root, dirs, files in os.walk(path):
        for name in files:
            label = root.split(os.sep)[-1]
            if label in to_discard:
                continue
            c = create(name, root, label, destination_path, augmentations)
            count += c
            progress_bar.update(1)

    print(f'Created {count}')


if __name__ == '__main__':
    main()
