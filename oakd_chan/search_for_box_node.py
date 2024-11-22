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
        # DepthAIパイプラインの作成
        self.device = self.initialize_pipeline()
        # ROS2のImageメッセージを送信するパブリッシャーの作成
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 3)
        # 30FPSでタイマーを設定し、コールバックを実行
        self.timer = self.create_timer(0.03, self.timer_callback)

    def initialize_pipeline(self):
        """Initialize the DepthAI pipeline."""
        # DepthAIパイプラインの作成
        pipeline = dai.Pipeline()
        # カラーカメラノードの作成と設定
        cam_rgb = pipeline.createColorCamera()
        
        cam_rgb.setPreviewSize(320, 240)
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
        cam_depth.setOutputSize(320, 240)  # depthの出力解像度を小さくする
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
        frame1 = q_depth.getFrame()

        # 深度画像を表示（OpenCVを使って）
        depth_image = frame1.astype(np.uint16)  # 16ビット深度画像
        cv2.imshow("Depth Image", depth_image)

        #箱検出関数 
        self.box_detection(frame)

        # メインの画像を表示
        # self.get_logger().info("画像出すよ")
        cv2.imshow("Color", frame)
        # cv2.imshow("Depth", frame1)
        cv2.waitKey(1)


    ######################
    ##ここから関数ゾーン##
    ######################

    #箱検出関数(紙検出ポリゴン化関数含む)
    def box_detection(self, frame):
        # 1. 緑色の範囲をHSVで定義
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #1.5 ぼかす
        blurred_image = cv2.GaussianBlur(hsv, (5, 5), 0)

        # 緑色の範囲を定義
        # lower_green = np.array([40, 40, 40])   # HSVで緑色の下限
        # upper_green = np.array([80, 255, 255])  # HSVで緑色の上限

        # ”青”色の範囲を定義
        lower_green = np.array([95, 100, 60])   # HSVで青色の下限  広すぎる。
        upper_green = np.array([115, 255, 255])  # HSVで青色の上限
        
        # 緑色部分のマスクを作成
        mask_green = cv2.inRange(blurred_image, lower_green, upper_green)
        
        # 2. マスクを使って緑の箱を検出
        contours_b, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_box = None
        for contour_b in contours_b:
            # 緑色の箱の輪郭を検出
            if cv2.contourArea(contour_b) > 500:  # 面積が小さいものは無視
                green_box = cv2.boundingRect(contour_b)  # 緑色の箱のバウンディングボックスを取得
                cv2.rectangle(frame, (green_box[0], green_box[1]), 
                            (green_box[0] + green_box[2], green_box[1] + green_box[3]), 
                            (255, 0, 0), 2)  # 緑色の箱に矩形を描画青


def main(args=None):
    rclpy.init(args=args)
    node = SearchForBoxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()