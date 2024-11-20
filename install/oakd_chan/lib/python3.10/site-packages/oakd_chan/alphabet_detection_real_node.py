import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pyrealsense2 as rs
from cv_bridge import CvBridge
import cv2
import pytesseract
import numpy as np

class AlphabetDetectionRealNode(Node):
    def __init__(self):
        super().__init__('alphabet_detection_real_node')

        # RealSenseパイプラインの初期化
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGBストリームを設定
        self.pipeline.start(config)

        # ROS2のImageメッセージを送信するパブリッシャーの作成
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 3)
        self.bridge = CvBridge()

        # 30FPSでタイマーを設定し、コールバックを実行
        self.timer = self.create_timer(0.03, self.timer_callback)

    def timer_callback(self):
        # RealSenseからフレームを取得
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        # OpenCV形式に変換
        frame = np.asanyarray(color_frame.get_data())

        # OCR処理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 輪郭を抽出
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 面積が最大の輪郭を取得
            largest_contour = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest_contour)

            # 最大輪郭の外接矩形で画像を切り出す
            roi = frame[cy:cy+ch, cx:cx+cw]

            # OCRで文字を認識
            text = pytesseract.image_to_string(roi, lang='eng').strip()

            if text:
                self.get_logger().info(f'Recognized text: {text}')

            # 最大輪郭を描画
            cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)

        # 画像をROS2トピックに送信
        image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(image_message)

        # 画像を表示
        cv2.imshow("Image of Detected Lines", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        # RealSenseのパイプラインを停止
        self.pipeline.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = AlphabetDetectionRealNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
