import rclpy #文字を認識するコード
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import depthai as dai  # DepthAIライブラリ
from cv_bridge import CvBridge
import cv2
import pytesseract
import numpy as np
import math
from collections import Counter
import time

class SearchForPaperNode(Node):
    def __init__(self):
        super().__init__('search_for_paper_node')

        # QoS for sensors
        sensor_qos = rclpy.qos.QoSProfile(depth=10)

         # パブリッシャーの初期化
        self.depth_publisher = self.create_publisher(Float32, '/paper_depth', 10)
        self.angle_publisher = self.create_publisher(Float32, '/paper_angle_x', 10)

                #出力解像度
        self.image_width = 320
        self.image_height = 240

        self.FOV_horizontal = 69  # 水平視野角 (度)
        self.FOV_vertical = 55    # 垂直視野角 (度)

        # 紙との距離関係
        self.depth_values = []  # depth_value を保存するリスト
        self.max_measurements = 10  # 最大測定数（キューの長さ）

        # DepthAIパイプラインの作成
        self.device = self.initialize_pipeline()
        # ROS2のImageメッセージを送信するパブリッシャーの作成
        # self.image_pub = self.create_publisher(Image, 'camera/image_raw', 3)

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


        cam_rgb.setPreviewSize(self.image_width, self.image_height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setFps(30)

        # 左カメラの設定
        left_camera = pipeline.create(dai.node.MonoCamera)
        left_camera.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P) # 720

        # 右カメラの設定
        right_camera = pipeline.create(dai.node.MonoCamera)
        right_camera.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right_camera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        # Depthカメラの設定
        cam_depth = pipeline.create(dai.node.StereoDepth)
        cam_depth.setOutputSize(self.image_width, self.image_height)  # depthの出力解像度を小さくする
        # cam_depth.setInput(left_camera, right_camera)  # 左右のカメラを入力に指定
        cam_depth.initialConfig.setConfidenceThreshold(200)
        cam_depth.setDepthAlign(dai.CameraBoardSocket.RGB)
        cam_depth.setConfidenceThreshold(255)
        cam_depth.setSubpixel(False)
        cam_depth.setRectifyEdgeFillColor(0)  # 黒でエッジを埋める

        # 左右のカメラをStereoDepthノードに接続
        left_camera.out.link(cam_depth.left)
        right_camera.out.link(cam_depth.right)

        # XLinkOutノードの作成（ホストへのデータ出力用）
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        cam_depth.depth.link(xout_depth.input)

        return dai.Device(pipeline)

    def timer_callback(self):  #コールバック関数

        in_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False).get()
        q_depth = self.device.getOutputQueue(name="depth", maxSize=8, blocking=False).get()

        # OpenCV形式のフレームに変換
        frame = in_rgb.getCvFrame() #表示用
        frame1 = frame.copy()       #認識用
        depth_frame = q_depth.getFrame() #深度フレーム

        #箱検出関数 〜 文字検出まで
        self.box_detection(frame, frame1, depth_frame)

        # メインの画像を表示
        cv2.imshow("Image of Detected Lines", frame)
        cv2.waitKey(1)


    ######################
    ##ここから関数ゾーン##
    ######################

    #箱検出関数(紙検出ポリゴン化関数含む)
    def box_detection(self, frame, frame1, depth_frame):

        # ガンマ補正の適用 日陰補正
        frame = self.adjust_gamma(frame, gamma=1.5)

        # 1. 緑色の範囲をHSVで定義
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #1.5 ぼかす
        blurred_image = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 緑色の範囲を定義
        #lower_green = np.array([50, 40, 40])   # HSVで緑色の下限
        #upper_green = np.array([70, 255, 255])  # HSVで緑色の上限

        # ”青”色の範囲を定義
        lower_green = np.array([95, 60, 60])   # HSVで青色の下限
        upper_green = np.array([115, 255, 255])  # HSVで青色の上限
        
        # 緑色部分のマスクを作成
        mask_green = cv2.inRange(blurred_image, lower_green, upper_green)
        
        # 2. マスクを使って緑の箱を検出
        contours_b, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area_g = 0
        max_counter_g = None # 最大面積の輪郭を格納
        green_box = None

        for contour_b in contours_b:
            # 緑色の箱の輪郭を検出
            if cv2.contourArea(contour_b) > 100:  # 面積が小さいものは無視
                area = cv2.contourArea(contour_b)
                if area > max_area_g:  # 最大面積を更新
                    max_area_g = area
                    max_counter_g = contour_b  # 最大面積の輪郭のバウンディングボックス

        # 最大の緑色の箱のバウンディングボックスが見つかった場合
        if max_counter_g is not None:
            green_box = cv2.boundingRect(max_counter_g)  # 緑色の箱のバウンディングボックスを取得
            cv2.rectangle(frame, (green_box[0], green_box[1]), 
                        (green_box[0] + green_box[2], green_box[1] + green_box[3]), 
                        (0, 255, 0), 2)  # 緑色の箱に矩形を描画緑
                
        #紙検出関数
        self.paper_detection(green_box, frame, frame1, depth_frame)


    #紙検出ポリゴン化関数(歪み補正関数含む)
    def paper_detection(self, green_box, frame, frame1, depth_frame):
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

            # 7. 紙の2値画像の輪郭を検出
            contours_p, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #2値画像から紙をポリゴン化
            # 最大面積を持つ輪郭を見つける
            max_area_p = 0
            max_contour = None

            #コピペゾーン https://qiita.com/sitar-harmonics/items/ac584f99043574670cf3
            for cnt in contours_p:
                    if cv2.contourArea(cnt) > 300:  # 面積が小さいものは無視
                        area = cv2.contourArea(cnt)
                        if area > max_area_p:  # 最大面積を更新
                            max_area_p = area
                            max_contour = cnt

            # 最大面積の輪郭のバウンディングボックスを取得
            if max_contour is not None:
               # 紙の輪郭を検出
                paper_position = cv2.boundingRect(max_contour)  # 紙のバウンディングボックスを取得

                # 箱の中心座標
                center_x = paper_position[0] + paper_position[2] // 2
                center_y = paper_position[1] + paper_position[3] // 2 

                center_gx = wx + center_x
                center_gy = wy + center_y

                cv2.circle(frame, (center_gx, center_gy), 5, (0, 0, 255), -1)  # 紙の中心に赤色の点を描画

                depth_value = depth_frame[center_gy, center_gx]  # 紙の中心までの距離

                # 箱の中心までの距離を表示
                #print(f"Distance to the box: {depth_value} mm")
                
                angle_x, angle_y = self.calculate_box_direction(center_gx, center_gy, self.image_width, self.image_height, self.FOV_horizontal, self.FOV_vertical)
                # print(f"紙の方向: 水平方向 {angle_x}度, 垂直方向 {angle_y}度")
                # print(f"紙の方向: 水平方向 {angle_x}度")

                # データをパブリッシュ
                self.publish_depth_and_angle(depth_value, angle_x)
                
                self.measure_depth(depth_value)

                # フレームに矩形を描画 青枠
                cv2.rectangle(frame, (paper_position[0] + wx , paper_position[1] + wy ), 
                            (paper_position[0] + paper_position[2] + wx, paper_position[1] + paper_position[3] + wy ), 
                            (255, 0, 0), 2)
                
                # 距離情報を描画
                distance_text = f"{depth_value:.2f} m, {angle_x:.1f}deg"
                # cv2.putText(frame, distance_text, (paper_position[0], paper_position[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                # 文字をフレーム中央に描画
                cv2.putText(frame, 
                            distance_text, 
                            (5,30),  # 描画位置
                            cv2.FONT_HERSHEY_SIMPLEX,  # フォント
                            1,  # フォントサイズ
                            (255, 255, 255),  # テキスト色（白）
                            2,  # 線の太さ
                            cv2.LINE_AA)  # アンチエイリアス


    #歪み補正関数
    def distortion_correction(self, approx, frame, frame2):
        area = abs(cv2.contourArea(approx))  #映像での紙のサイズ
        if approx.shape[0] == 4 and area > 500 and cv2.isContourConvex(approx) :
            #self.get_logger().info(f"Contour Area: {area}")

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
                cv2.imshow("Debug Frame", frame)
                cv2.imshow("Warped A5", warped)
                cv2.moveWindow("Waroed A5", 0, 0)   

    def adjust_gamma(self, image, gamma=1.5):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def publish_depth_and_angle(self, depth_value, angle_x):
        # depth_valueをパブリッシュ
        depth_msg = Float32()
        depth_msg.data = float(depth_value)  # 明示的にfloat型に変換
        self.depth_publisher.publish(depth_msg)
        self.get_logger().info(f"Published depth: {depth_value}")

        # angle_xをパブリッシュ
        angle_msg = Float32()
        angle_msg.data = float(angle_x)  # 明示的にfloat型に変換
        self.angle_publisher.publish(angle_msg)
        self.get_logger().info(f"Published angle_x: {angle_x}")

    # 水平と垂直方向の角度を計算
    def calculate_box_direction(self, center_x, center_y, image_width, image_height, FOV_horizontal, FOV_vertical):
        
        angle_x = (center_x - image_width / 2) / (image_width / 2) * FOV_horizontal / 2
        angle_y = (center_y - image_height / 2) / (image_height / 2) * FOV_vertical / 2

        return angle_x, angle_y

    # 紙との距離確定
    def measure_depth(self, depth_value):
        # センサーから取得した値をシミュレート (ここは実際のセンサー値取得コードに置き換える)
        self.depth_values.append(depth_value)

        # 古い値を削除して新しい値を追加
        if len(self.depth_values) >= self.max_measurements:
            removed_value = self.depth_values.pop(0)  # 最も古い値を削除
           # self.get_logger().info(f'Removed oldest depth value: {removed_value}')
        
        self.depth_values.append(depth_value)# 新しい数を追加
        #self.get_logger().info(f'Added new depth value: {depth_value}')

        # 平均値を計算してログを表示
        if len(self.depth_values) >= self.max_measurements:
            avg_depth = sum(self.depth_values) / len(self.depth_values)
            #self.get_logger().info(f'Current depth values: {self.depth_values}')
            self.get_logger().info(f'Current average = {avg_depth}')

            # 1500mm以内か？
            if avg_depth < 1500:
                self.get_logger().warn(f'Warning: 1.5m以内だあああ!')


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
    node = SearchForPaperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
