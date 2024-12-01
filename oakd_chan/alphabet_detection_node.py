import rclpy #文字を認識するコード
from rclpy.node import Node
from sensor_msgs.msg import Image
import depthai as dai  # DepthAIライブラリ
from cv_bridge import CvBridge
import cv2
import pytesseract
import numpy as np
import math
from collections import Counter
import time

class AlphabetDetectionNode(Node):
    def __init__(self):
        super().__init__('alphabet_detection_node')
        # DepthAIパイプラインの作成
        self.device = self.initialize_pipeline()
        # ROS2のImageメッセージを送信するパブリッシャーの作成
        # self.image_pub = self.create_publisher(Image, 'camera/image_raw', 3)
        
        # 文字認識結果を保存するリスト
        self.recognized_texts = []
        # 特定の文字をカウントするためのカウンタ
        self.char_count = {'a': 0, 'A': 0, 'b': 0, 'B': 0, 'c': 0, 'C': 0}  

        self.start_time = None  # 文字の認識開始時刻を保存
        self.timer_started = False  # タイマーが開始されたかどうかを示すフラグ

        # 30FPSでタイマーを設定し、コールバックを実行
        self.timer = self.create_timer(0.03, self.timer_callback)

        # 5秒後に呼び出すタイマー
        self.later_timer = None

    def initialize_pipeline(self):
        """Initialize the DepthAI pipeline."""
        # 接続されているすべてのデバイス情報を取得
        available_devices = dai.Device.getAllAvailableDevices()

        if not available_devices:
            print("No devices found!")
            exit()

        print("Available devices:")
        for i, device in enumerate(available_devices):
            print(f"{i}: {device.getMxId()} ({device.state.name})")

        # 使用したいデバイスのシリアル番号を指定
        target_serial = "18443010A1D5F50800"  # 任意のシリアル番号に置き換え

        # 対応するデバイスを探す
        target_device_info = None
        for device in available_devices:
            if device.getMxId() == target_serial:
                target_device_info = device
                break

        if target_device_info is None:
            print(f"Device with serial {target_serial} not found!")
            exit()
        # DepthAIパイプラインの作成
        pipeline = dai.Pipeline()
        # 特定のデバイスでパイプラインを実行
        with dai.Device(pipeline, target_device_info) as device:
            print(f"Using device: {device.getMxId()}")
        # カラーカメラノードの作成と設定
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(320, 240)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setFps(30)

        # XLinkOutノードの作成（ホストへのデータ出力用）
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        return dai.Device(pipeline)

    def timer_callback(self):  #コールバック関数

        in_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False).get()

        # OpenCV形式のフレームに変換
        frame = in_rgb.getCvFrame() #表示用
        frame1 = frame.copy()       #認識用
        frame2 = frame.copy()       #歪み補正チェック用

        #箱検出関数 〜 文字検出まで
        self.box_detection(frame, frame1, frame2)

        # メインの画像を表示
        # cv2.imshow("main", frame)
        cv2.waitKey(1)



    ######################
    ##ここから関数ゾーン##
    ######################

    #箱検出関数(紙検出ポリゴン化関数含む)
    def box_detection(self, frame, frame1, frame2):

        # ガンマ補正の適用 日陰補正
        frame = self.adjust_gamma(frame, gamma=1.5)
        


        if frame.dtype != 'uint8':
            frame = cv2.convertScaleAbs(frame)  # 描画用にデータ型を変換

        # 1. 緑色の範囲をHSVで定義
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #1.5 ぼかす
        blurred_image = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 緑色の範囲を定義
        lower_green = np.array([50, 40, 40])   # HSVで緑色の下限
        upper_green = np.array([70, 255, 255])  # HSVで緑色の上限

        # ”青”色の範囲を定義
        # lower_green = np.array([90, 60, 60])   # HSVで青色の下限
        # upper_green = np.array([150, 255, 255])  # HSVで青色の上限
        
        # 緑色部分のマスクを作成
        mask_green = cv2.inRange(blurred_image, lower_green, upper_green)
        
        # 2. マスクを使って緑の箱を検出
        contours_b, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_box = None
        for contour_b in contours_b:
            # 緑色の箱の輪郭を検出
            if cv2.contourArea(contour_b) > 1000:  # 面積が小さいものは無視
                green_box = cv2.boundingRect(contour_b)  # 緑色の箱のバウンディングボックスを取得
                cv2.rectangle(frame, (green_box[0], green_box[1]), 
                            (green_box[0] + green_box[2], green_box[1] + green_box[3]), 
                            (0, 255, 0), 2)  # 緑色の箱に矩形を描画緑
                
        #紙検出関数
        self.paper_detection(green_box, frame, frame1, frame2)


    #紙検出ポリゴン化関数(歪み補正関数含む)
    def paper_detection(self, green_box, frame, frame1, frame2):
        if green_box is not None: #引数：green_box frame1  返し:contours?
            wx, wy, ww, wh = green_box
            # 3. 緑色の箱の内部から白い紙を検出
            roi = frame1[wy:wy+wh, wx:wx+ww] # roi : 緑箱のバウンディングボックス
            # blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 白色の範囲をHSVで定義　日陰だとキツイかも
            lower_white = np.array([0, 0, 120])
            upper_white = np.array([180, 60, 255])#彩度を40から60に上げるとつくばチャレンジの文字の影響が小さくなった。

            ###ここからコピペ
            
            # 5. hsvに変換した画像を　白(1)or白以外(0)　の２値画像に変換
            white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)

            cv2.imshow("white_mask", white_mask)
            cv2.moveWindow("white_mask", 0, 0)    # 紙が四角形で出ているか？

            # 5.5 2値画像の白色部分を拡張 
            # dilated_image = cv2.dilate(white_mask, np.ones((2, 2), np.uint8), iterations=1)

            # 7. 紙の2値画像の輪郭を検出
            contours_p, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #2値画像から紙をポリゴン化

            #コピペゾーン https://qiita.com/sitar-harmonics/items/ac584f99043574670cf3
            for i, cnt in enumerate(contours_p):
                # 輪郭の周囲に比例する精度で輪郭を近似する
                cnt_global = cnt + np.array([[wx, wy]])  # ROIの座標オフセットを追加
                arclen = cv2.arcLength(cnt_global, True)
                approx = cv2.approxPolyDP(cnt_global, arclen*0.1, True) #0.02だと低すぎる。0.1だとかなり余裕。

                #四角形の輪郭は、近似後に4つの頂点があります。
                #比較的広い領域が凸状になります。
                # 凸性の確認 
                
                #歪み補正関数(文字認識関数含む)
                self.distortion_correction(approx, frame, frame2)


    #歪み補正関数(文字認識関数含む)
    def distortion_correction(self, approx, frame, frame2):
        area = abs(cv2.contourArea(approx))
        if approx.shape[0] == 4 and area > 500 and cv2.isContourConvex(approx) :

            maxCosine = 0

            for j in range(2, 5):
                # 辺間の角度の最大コサインを算出
                cosine = abs(angle(approx[j%4], approx[j-2], approx[j-1]))
                maxCosine = max(maxCosine, cosine)

            # すべての角度の余弦定理が小さい場合（すべての角度は約90度です）次に、quandrangeを書き込みます
            # 結果のシーケンスへの頂点
            if maxCosine < 0.3 :#およそ73.24度　上の紙を認識するかはカメラの高さとここが関わってくる。
                # 四角判定!!
                rcnt = approx.reshape(-1,2)
                # cv2.polylines(frame, [rcnt], True, (0,0,255), thickness=2, lineType=cv2.LINE_8)
                cv2.fillPoly(frame, [rcnt], (0,0,255), lineType=cv2.LINE_8)
                cv2.imshow("Debug Frame", frame)

                ## 以降歪み補正
                # 頂点を並び替える関数
                def order_points(pts):
                    rect = np.zeros((4, 2), dtype="float32")
                    s = pts.sum(axis=1)
                    rect[0] = pts[np.argmin(s)]  # 左上
                    rect[2] = pts[np.argmax(s)]  # 右下
                    diff = np.diff(pts, axis=1)
                    rect[1] = pts[np.argmin(diff)]  # 右上
                    rect[3] = pts[np.argmax(diff)]  # 左下
                    return rect

                # 頂点を並び替え
                rect = order_points(approx.reshape(4, 2))

                # A5サイズの縦横比（横長）
                width = 210  # mm
                height = 148  # mm
                aspect_ratio = width / height

                # 横長A5を基準とした新しい座標を計算
                maxWidth = int(500)  # 横方向のサイズを任意に設定
                maxHeight = int(maxWidth / aspect_ratio)  # 縦方向は縦横比に基づいて計算

                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")

                # 透視変換行列を計算
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(frame2, M, (maxWidth, maxHeight))

                # 結果を表示
                cv2.imshow("Warped A5", warped)
                cv2.moveWindow("Waroed A5", 0, 0)   

                #文字を認識する関数
                self.character_detection(warped)


    #文字検出関数
    def character_detection(self, warped):
        """文字を読み取る関数"""
        # グレースケールに変換
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # ノイズ除去 (ガウシアンブラー)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        blurred = cv2.medianBlur(gray, 5)
        # 二値化 (Otsuの手法)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 文字の輪郭を抽出
        contours_c, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 上だけブラー＆文字認識直前に上40%をトリミング
        height_b, width_b = binary.shape[:2]

        # 上部10%の範囲を取得
        top_blur = int(height_b * 0.5)
        top_area = binary[:top_blur, :]  # 上部10%

        # ぼかし処理を適用 (GaussianBlurを使用)
        blurred_top = cv2.GaussianBlur(top_area, (61, 61), 0)

        # 元画像にぼかし部分を反映
        binary[:top_blur, :] = blurred_top

        top_trim = int(height_b * 0.42)   # 紙の上40%をトリミング
        bottom_trim = int(width_b * 0.05)  # 左右5%をトリミング
        bottom_trim = int(height_b * 0.01)  # 下部5%をトリミング

        # トリミング画像の作成
        trimmed_image = binary[top_trim:height_b - bottom_trim, bottom_trim:width_b - bottom_trim]

        cv2.imshow("moji", trimmed_image) #読み取る直前の画像を表示
        cv2.moveWindow("moji", 500, 0) 

        if contours_c:
            # 切り出した部分にOCRを適用
            text = pytesseract.image_to_string(trimmed_image, lang='eng', config = r'--psm 10').strip()
            if text:
                self.get_logger().info(f'Recognized text: {text}')

               # 認識した文字が特定の文字リスト（self.char_count）に含まれている場合のみ処理
                for char in self.char_count:
                    if char in text:
                        # 認識された文字列をリストに追加
                        self.recognized_texts.append(text)

                        # 特定の文字をカウント
                        for char in text:
                            if char in self.char_count:
                                self.char_count[char] += 1

                                # 初めて文字を認識したときにタイマーを開始
                                if not self.timer_started:
                                    self.start_later_timer()


    def start_later_timer(self):
        """5秒後に特定の関数を呼び出すタイマーを開始"""
        if not self.timer_started:
            self.later_timer = self.create_timer(5.0, self.finalize_count)
            self.timer_started = True  # タイマーが開始されたことを記録
            self.get_logger().info("5秒タイマーを開始しました。")

    def finalize_count(self):
        """カウント結果を処理し、大文字小文字を合計して最も多く認識した文字を選ぶ"""
        # 大文字・小文字の合計を計算
        total_counts = {
            'A': self.char_count['a'] + self.char_count['A'],
            'B': self.char_count['b'] + self.char_count['B'],
            'C': self.char_count['c'] + self.char_count['C']
        }

        # 最も多く認識された文字を選択
        if total_counts:
            most_recognized_char = max(total_counts, key=total_counts.get)
            self.get_logger().info(
                f"最も多く認識された文字: {most_recognized_char} (合計: {total_counts[most_recognized_char]})"
            )
        else:
            self.get_logger().info("認識された文字がありません。")

        # タイマーを停止
        if self.later_timer is not None:
            self.later_timer.cancel()
            self.later_timer = None

    def adjust_gamma(self, image, gamma=1.5):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

# pt0-> pt1およびpt0-> pt2からの
# ベクトル間の角度の余弦(コサイン)を算出
def angle(pt1, pt2, pt0) -> float:
    """角度を計算する関数"""
    dx1 = float(pt1[0,0] - pt0[0,0])
    dy1 = float(pt1[0,1] - pt0[0,1])
    dx2 = float(pt2[0,0] - pt0[0,0])
    dy2 = float(pt2[0,1] - pt0[0,1])
    v = math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) )
    return (dx1*dx2 + dy1*dy2)/ v


def main(args=None):
    rclpy.init(args=args)
    node = AlphabetDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()