
import os
import numpy as np
import torch
import cv2
import mediapipe as mp
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools
from utils import load_config, get_hand_points, char2int, clean_folder
import matplotlib.pyplot as plt

from tqdm import tqdm

pred = {'a':(370, 385), 'b':(530, 700), 'c':(745, 875), 'd':(945, 1080), 'e':(1125, 1234), 'f':(1285, 1430),
        'g':(1475, 1600), 'h':(1655, 1770), 'i':(1820, 1940), 'j':(2005, 2288), 'k':(2349, 2463), 'l':(2534, 2652),
        'm':(2720, 2813), 'n':(2869, 2990), 'o':(3054, 3136), 'p':(3202, 3357), 'q':(3430, 3560), 'r':(3608, 3725),
        's':(3788, 3890), 't':(3940, 4072), 'u':(4118, 4231), 'v':(4298, 4400), 'w': (4463, 4570), 'x':(4632, 4725),
        'y':(4810, 4864), 'z':(4930, 5180), '1':(100000, 100000), '2':(100000, 100000), '3':(100000, 100000),
        '4':(100000, 100000), '5':(100000, 100000), '6':(100000, 100000), '7':(100000, 100000),
        '8':(100000, 100000), '9':(100000, 100000), '0':(100000, 100000)}

pred = {'a':(370, 385), 'b':(530, 700), 'c':(745, 875), 'd':(945, 1080), 'e':(1125, 1234), 'f':(1285, 1430),
        'g':(1475, 1600), 'h':(1655, 1770), 'i':(1820, 1940), 'k':(2349, 2463), 'l':(2534, 2652),
        'm':(2720, 2813), 'n':(2869, 2990), 'o':(3054, 3136), 'p':(3202, 3357), 'q':(3430, 3560), 'r':(3608, 3725),
        's':(3788, 3890), 't':(3940, 4072), 'u':(4118, 4231), 'v':(4298, 4400), 'w': (4463, 4570), 'x':(4632, 4725),
        'y':(4810, 4864), '1':(100000, 100000), '2':(100000, 100000), '3':(100000, 100000),
        '4':(100000, 100000), '5':(100000, 100000), '6':(100000, 100000), '7':(100000, 100000),
        '8':(100000, 100000), '9':(100000, 100000), '0':(100000, 100000)}


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def predict(model, img):
    model.eval()
    points_raw = get_hand_points(img)
    try:
        points = points_raw.copy()
        min_x = np.min(points_raw[:, 0])
        max_x = np.max(points_raw[:, 0])
        min_y = np.min(points_raw[:, 1])
        max_y = np.max(points_raw[:, 1])
        for i in range(len(points_raw)):
            points[i][0] = (points[i][0] - min_x) / (max_x - min_x)
            points[i][1] = (points[i][1] - min_y) / (max_y - min_y)
    except:
        return None, None

    pointst = torch.tensor([points]).float().to(device)
    label = model(pointst)
    label = label.detach().cpu().numpy()
    label = np.argmax(label)
    label = list(char2int.keys())[list(char2int.values()).index(label)]

    return label, points_raw


def plot_cm(conf_matrix, ):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, file=''):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Sign')
    plt.xlabel('Predicted Sign')
    plt.savefig(file)
    plt.show()


def display_histogram(sign_dict):
    sign_dict = dict(sorted(sign_dict.items()))
    plt.bar(list(sign_dict.keys()), sign_dict.values(), color='steelblue', edgecolor='black', linewidth=1.2)
    plt.title('Distribution of misclassified signs')
    plt.xlabel('Sign')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def predict_images():
    config = load_config('config.yaml')
    model_name = config['model']['name']
    path = config['dataset']['test_dataset']
    missclassified_path = config['paths']['missclassified_path']
    model_path = config['model']['model_path']

    model = torch.load(os.path.join(model_path, model_name))

    os.makedirs(missclassified_path, exist_ok=True)
    clean_folder(missclassified_path)

    actuals = []
    predicteds = []
    ss = set()
    signs = list(char2int.keys())
    wrongs = {}
    for root, dirs, files in os.walk(path):
        gt = root.split(os.sep)[-1].lower()
        print(f'Current sign:{gt}')
        count = 0
        for file in files:
            if gt in signs:
                ss.add(gt)
                img = cv2.imread(os.path.join(root, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img.flags.writeable = False
                predicted_label, points = predict(model, img)
                if predicted_label is not None:
                    predicteds.append(predicted_label)
                    actuals.append(gt)
                    if predicted_label != gt:
                        if predicted_label not in wrongs:
                            wrongs[predicted_label] = 1
                        else:
                            wrongs[predicted_label] = wrongs[predicted_label] + 1
                        os.makedirs(os.path.join(missclassified_path, predicted_label), exist_ok=True)
                        cv2.imwrite(os.path.join(missclassified_path, predicted_label, gt + '_' + file), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                else:
                    print('None')
            count += 1
            if count >= 100:
                break
    print(actuals[:10])
    print(predicteds[:10])
    acc = accuracy_score(actuals, predicteds)
    print(f'Acc {acc}')

    cm = confusion_matrix(actuals, predicteds, labels=signs)
    print(cm)
    plot_cm(cm)

    _, _ = plt.subplots(figsize=(12, 10))
    file = f'D:\\Roe\\Medium\\paper_to\\handShape_pointNet\\figuras\\cm_{2}.png'
    plot_confusion_matrix(cm, signs, normalize=False, file=file, cmap='Purples')

    print(f'{sum(wrongs.values())} misclassified images')
    display_histogram(wrongs)


if __name__ == '__main__':
    predict_images()