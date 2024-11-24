import rclpy #画面上の箱の位置と距離を計算するコード
from rclpy.node import Node
from sensor_msgs.msg import Image
import depthai as dai  # DepthAIライブラリ
import cv2
import numpy as np
import math
import time

class SearchForBoxNode(Node):
    def __init__(self):
        super().__init__('search_for_box_node')

        #出力解像度
        self.image_width = 320
        self.image_height = 240

        self.FOV_horizontal = 90  # 水平視野角 (度)
        self.FOV_vertical = 69    # 垂直視野角 (度)

        # DepthAIパイプラインの作成
        self.device = self.initialize_pipeline()
        # ROS2のImageメッセージを送信するパブリッシャーの作成
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 3)
        # 30FPSでタイマーを設定し、コールバックを実行
        self.timer = self.create_timer(0.03, self.timer_callback)

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

        # パイプラインを作成
        pipeline = dai.Pipeline()
        # 必要なノード設定をここに記述...

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
        depth_frame = q_depth.getFrame() #深度フレーム

        # 深度画像を表示（OpenCVを使って）
        depth_image = depth_frame.astype(np.uint16)  # 16ビット深度画像
        # cv2.imshow("Depth Image", depth_image)

        #箱検出関数 
        self.box_detection(frame, depth_frame)




        # メインの画像を表示
        # self.get_logger().info("画像出すよ")
        cv2.imshow("Color", frame)
        # cv2.imshow("Depth", depth_frame)
        cv2.waitKey(1)


    ######################
    ##ここから関数ゾーン##
    ######################

    #箱検出関数(紙検出ポリゴン化関数含む)
    def box_detection(self, frame, depth_frame):
        # 1. 緑色の範囲をHSVで定義
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #1.5 ぼかす
        blurred_image = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 緑色の範囲を定義
        # lower_green = np.array([40, 40, 40])   # HSVで緑色の下限
        # upper_green = np.array([80, 255, 255])  # HSVで緑色の上限

        # ”青”色の範囲を定義
        lower_blue = np.array([95, 100, 60])   # HSVで青色の下限  広すぎる。
        upper_blue = np.array([115, 255, 255])  # HSVで青色の上限

        # 2. 画像の下半分を切り出す
        height, width = blurred_image.shape[:2]
        bottom_half = blurred_image[height // 2:, :]  # 高さの半分から下
        
        # 緑色部分のマスクを作成
        mask_blue = cv2.inRange(bottom_half, lower_blue, upper_blue)
        
        # 2. マスクを使って緑の箱を検出
        contours_b, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        blue_box = None
        for contour_b in contours_b:
            # 緑色の箱の輪郭を検出
            if cv2.contourArea(contour_b) > 500:  # 面積が小さいものは無視
                blue_box = cv2.boundingRect(contour_b)  # 青色の箱のバウンディングボックスを取得

                # y座標を元のフレーム基準に調整（下半分に対応）
                adjusted_y = blue_box[1] + height // 2

                # 箱の中心座標
                center_x = min(max(blue_box[0] + blue_box[2] // 2, 0), depth_frame.shape[1] - 1)
                center_y = min(max(blue_box[1] + blue_box[3] // 2 + height // 2, 0), depth_frame.shape[0] - 1)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)  # 箱の中心に赤色の点を描画

                depth_value = depth_frame[center_y, center_x]  # 箱の中心までの距離

                # 箱の中心までの距離を表示
                print(f"Distance to the box: {depth_value} meters")
                
                angle_x, angle_y = self.calculate_box_direction(center_x, center_y, self.image_width, self.image_height, self.FOV_horizontal, self.FOV_vertical)
                # print(f"箱の方向: 水平方向 {angle_x}度, 垂直方向 {angle_y}度")
                print(f"箱の方向: 水平方向 {angle_x}度")
                

                # フレームに矩形を描画 青枠
                cv2.rectangle(frame, (blue_box[0] , adjusted_y), 
                              (blue_box[0] + blue_box[2], adjusted_y + blue_box[3]), 
                              (255, 0, 0), 2)
                
                 # 距離情報を描画
                distance_text = f"{depth_value:.2f} m, {angle_x:.1f}deg"
                # cv2.putText(frame, distance_text, (blue_box[0], adjusted_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                # 文字をフレーム中央に描画
                cv2.putText(frame, 
                            distance_text, 
                            (5,30),  # 描画位置
                            cv2.FONT_HERSHEY_SIMPLEX,  # フォント
                            1,  # フォントサイズ
                            (255, 255, 255),  # テキスト色（白）
                            2,  # 線の太さ
                            cv2.LINE_AA)  # アンチエイリアス

    # 水平と垂直方向の角度を計算
    def calculate_box_direction(self, center_x, center_y, image_width, image_height, FOV_horizontal, FOV_vertical):
        
        angle_x = (center_x - image_width / 2) / (image_width / 2) * FOV_horizontal / 2
        angle_y = (center_y - image_height / 2) / (image_height / 2) * FOV_vertical / 2

        return angle_x, angle_y




def main(args=None):
    rclpy.init(args=args)
    node = SearchForBoxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()