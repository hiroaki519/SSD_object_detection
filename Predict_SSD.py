import sys
import os
import torch
import numpy as np
current_folder_path = os.getcwd()
sys.path.append(current_folder_path)

from ssd import SSD  # SSD  クラスをインポート

# VOC2012の正解ラベルのリスト
voc_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train',
    'tvmonitor'
]

# SSDモデルの設定値
ssd_cfg = {
    'classes_num': 21,                      # 背景クラスを含めた合計クラス数
    'input_size': 300,                      # 画像の入力サイズ
    'dbox_num': [4, 6, 6, 6, 4, 4],         # 出力するDBoxのアスペクト比の種類
    'feature_maps': [38, 19, 10, 5, 3, 1],  #各sourceの画像サイズ
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]] 
}

# 推論モードのSSDモデルを生成
net = SSD(phase='test', cfg=ssd_cfg)

# 学習済みの重みを設定
net_weights = torch.load(
    './weights_ite/ssd_weights10000_3.pth',
    # map_location = {'cuda: 0': 'cpu'}
    map_location=torch.device('cpu')
)

# 重みをロードする
net.load_state_dict(net_weights)
print('SSDモデルの準備完了')

import cv2
from voc import DataTransform
import matplotlib.pyplot as plt
# matplotlib inline

#　画像の読み込み
image_file_path = './data/VOCdevkit/VOC2012/JPEGImages/2914.jpg'
img = cv2.imread(image_file_path)
height, width, channels = img.shape

# 画像を出力
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 前処理クラスを生成
color_mean = (104, 117, 123)      # VOC2012の(BGR)の平均値
input_size = 300                  # 画像の入力サイズは300×300
transform = DataTransform(input_size, color_mean)

# 検証用の前処理を実施
phase = 'val'
img_transformed, boxes, labels = transform(
    img,     #画像
    phase,   # 処理モード
    '',      # BBoxの正解座標、正解ラベルはないので
    ''
)
# BGRの並びをRGBの順に変更
img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

# SSDで物体検出を実施
net.eval()              # SSDモデルを推論モードにする
x = img.unsqueeze(0)    # 0の次元を追加してミニバッチ化(1, 3, 300, 300)
detections = net(x)     # SSDモデルに入力
# detection(1,
#           21[クラス数],
#           200[確信度上位のBBox数],
#           5[確信度, xmin, ymin, xmax, ymax])

# 予測値を出力
print(detections)

from ssd_predictions import SSDPredictions

# 予測と予測結果を画像で描画する

ssd = SSDPredictions(eval_categories=voc_classes, net=net)


rgb_img, predict_bbox, pre_dict_label_index, scores = ssd.ssd_predict(image_file_path,confidence_threshold=0.5)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('pre_dict_label_index',predict_bbox)

# BBoxを抽出する際の閾値を0.6にする
ssd.show(image_file_path, confidence_threshold=0.5)
