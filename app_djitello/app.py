print('importing library')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import os 
from torch.utils.data import Dataset
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from djitellopy import Tello    # DJITelloPyのTelloクラスをインポート
import time                     # time.sleepを使いたいので
import pdb

model_file_path = "../model/model_ft_gpu.pth"
model_ft = models.resnet18(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 3)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
model_ft.eval()
print('model loaded.')

preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

# メイン関数
def main():
    # 初期化部
    # Telloクラスを使って，tellというインスタンス(実体)を作る
    tello = Tello(retry_count=1)    # 応答が来ないときのリトライ回数は1(デフォルトは3)
    tello.RESPONSE_TIMEOUT = 0.01   # コマンド応答のタイムアウトは短くした(デフォルトは7)

    # Telloへ接続
    tello.connect()

    # 画像転送を有効にする
    tello.streamoff()   # 誤動作防止の為、最初にOFFする
    tello.streamon()    # 画像転送をONに
    frame_read = tello.get_frame_read()     # 画像フレームを取得するBackgroundFrameReadクラスのインスタンスを作る

    current_time = time.time()  # 現在時刻の保存変数
    pre_time = current_time     # 5秒ごとの'command'送信のための時刻変数

    # SDKバージョンを問い合わせ
    sdk_ver = tello.query_sdk_version()

    # モータとカメラの切替フラグ
    motor_on = False                    # モータON/OFFのフラグ
    camera_dir = Tello.CAMERA_FORWARD   # 前方/下方カメラの方向のフラグ
    
    # 前回強制終了して下方カメラかもしれないので
    if sdk_ver == '30':                                     # SDK 3.0に対応しているか？ 
        tello.set_video_direction(Tello.CAMERA_FORWARD)     # カメラは前方に

    time.sleep(0.5)     # 通信が安定するまでちょっと待つ

    # ループ部
    # Ctrl+cが押されるまでループ
    try:
        # 永久ループで繰り返す
        while True:

            # (1) 画像取得
            image = frame_read.frame    # 映像を1フレーム取得しimage変数に格納

            # (2) 画像サイズ変更と、カメラ方向による回転
            small_image = cv2.resize(image, dsize=(480,360) )   # 画像サイズを半分に変更
            small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

            if camera_dir == Tello.CAMERA_DOWNWARD:     # 下向きカメラは画像の向きが90度ずれている
                small_image = cv2.rotate(small_image, cv2.ROTATE_90_CLOCKWISE)      # 90度回転して、画像の上を前方にする

            # (3) ここから画像処理
            frame = small_image
            pil_image = Image.fromarray(frame)
            img_tensor = preprocess(pil_image)
            img_tensor = img_tensor.unsqueeze(0)
            cropped_img = img_tensor.squeeze().permute(1, 2, 0).numpy()
            cv2.imshow('frame2',cropped_img)

            with torch.no_grad():
                pred = model_ft(img_tensor)
                predicted = class_names[pred[0].argmax(0)]
                pred_index = pred[0].argmax(0)
                if(pred_index)!=0:
                    fontColor = (255,0,0)
                if(pred_index)!=1:
                    fontColor = (128,128,128)
                if(pred_index)!=2:
                    fontColor = (255,255,255)

                blank_image = np.zeros((50,400,3), np.uint8)
                cv2.putText(
                    img = blank_image,
                    text = predicted,
                    org = topLeftCornerOfText,
                    fontFace = font,
                    fontScale = fontScale,
                    color = fontColor,
                    thickness = thickness,
                )
                print(f'Predicted: "{predicted}"')
                cv2.imshow('frame3',blank_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # (4) ウィンドウに表示
            cv2.imshow('OpenCV Window', small_image)    # ウィンドウに表示するイメージを変えれば色々表示できる

            # (5) OpenCVウィンドウでキー入力を1ms待つ
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                   # key が27(ESC)だったらwhileループを脱出，プログラム終了
                break
            elif key == ord('t'):           # 離陸
                tello.takeoff()
            elif key == ord('l'):           # 着陸
                tello.land()
            elif key == ord('w'):           # 前進 30cm
                tello.move_forward(30)
            elif key == ord('s'):           # 後進 30cm
                tello.move_back(30)
            elif key == ord('a'):           # 左移動 30cm
                tello.move_left(30)
            elif key == ord('d'):           # 右移動 30cm
                tello.move_right(30)
            elif key == ord('e'):           # 旋回-時計回り 30度
                tello.rotate_clockwise(30)
            elif key == ord('q'):           # 旋回-反時計回り 30度
                tello.rotate_counter_clockwise(30)
            elif key == ord('r'):           # 上昇 30cm
                tello.move_up(30)
            elif key == ord('f'):           # 下降 30cm
                tello.move_down(30)
            elif key == ord('p'):           # ステータスをprintする
                print(tello.get_current_state())
            elif key == ord('m'):           # モータ始動/停止を切り替え
                if sdk_ver == '30':         # SDK 3.0に対応しているか？
                    if motor_on == False:       # 停止中なら始動 
                        tello.turn_motor_on()
                        motor_on = True
                    else:                       # 回転中なら停止
                        tello.turn_motor_off()
                        motor_on = False
            elif key == ord('c'):           # カメラの前方/下方の切り替え
                if sdk_ver == '30':         # SDK 3.0に対応しているか？
                    if camera_dir == Tello.CAMERA_FORWARD:     # 前方なら下方へ変更
                        tello.set_video_direction(Tello.CAMERA_DOWNWARD)
                        camera_dir = Tello.CAMERA_DOWNWARD     # フラグ変更
                    else:                                      # 下方なら前方へ変更
                        tello.set_video_direction(Tello.CAMERA_FORWARD)
                        camera_dir = Tello.CAMERA_FORWARD      # フラグ変更
                    time.sleep(0.5)     # 映像が切り替わるまで少し待つ

            # (6) 10秒おきに'command'を送って、死活チェックを通す
            current_time = time.time()                          # 現在時刻を取得
            if current_time - pre_time > 10.0 :                 # 前回時刻から10秒以上経過しているか？
                tello.send_command_without_return('command')    # 'command'送信
                pre_time = current_time                         # 前回時刻を更新

    except( KeyboardInterrupt, SystemExit):    # Ctrl+cが押されたらループ脱出
        print( "Ctrl+c を検知" )

    # 終了処理部
    cv2.destroyAllWindows()                             # すべてのOpenCVウィンドウを消去
    
    if sdk_ver == '30':                                 # SDK 3.0に対応しているか？
        tello.set_video_direction(Tello.CAMERA_FORWARD) # カメラは前方に戻しておく

    tello.streamoff()                                   # 画像転送を終了(熱暴走防止)
    frame_read.stop()                                   # 画像受信スレッドを止める

    del tello.background_frame_read                     # フレーム受信のインスタンスを削除    
    del tello                                           # telloインスタンスを削除


# "python3 main_core.py"として実行された時だけ動く様にするおまじない処理
if __name__ == "__main__":      # importされると__name_に"__main__"は入らないので，pyファイルが実行されたのかimportされたのかを判断できる．
    main()    # メイン関数を実行


  