from PIL import Image
import torch
from pathlib import Path
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
import numpy as np
import pdb

# YOLOv5モデルを読み込む
weights = 'yolov5s.pt'
device = select_device('')
#model = attempt_load(weights, map_location=device)
model = attempt_load(weights, device=device)
stride = int(model.stride.max())

# 画像を読み込む
img_path = 'plant_leaves.png'
img0 = Image.open(img_path)

# 画像をリサイズして正規化
img = img0.convert('RGB')
img = img.resize((640, 640))
img = torch.from_numpy(np.array(img)).to(device)
img = img.float() / 255.0
if img.ndimension() == 3:
    img = img.unsqueeze(0)

# YOLOv5を使用して物体検出を行う
pred = model(img.permute(0,3,1,2))[0]
print(pred)
#pred = non_max_suppression(pred, 0.4, 0.5)
pred = non_max_suppression(pred, 0.1, 0.9)
print(pred)

# 検出された葉を切り出す
for det in pred[0]:
    pdb.set_trace()
    x1, y1, x2, y2, conf, cls = det
    box = [x1, y1, x2-x1, y2-y1]
    box = scale_boxes(img.shape[2:], box, img0.size).int()
    leaf = img0.crop((box[0], box[1], box[0]+box[2], box[1]+box[3]))
    
    # 各葉を別々の画像として保存
    leaf.save(f'leaf_{int(cls)}.jpg')

# 検出された葉にバウンディングボックスを描画して表示（Ultralyticsでは自動で描画される）
