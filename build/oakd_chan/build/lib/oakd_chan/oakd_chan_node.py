import rclpy  # ROS2ライブラリ
from rclpy.node import Node  # ノードクラス
from sensor_msgs.msg import Image  # ROSの画像メッセージ型
import depthai as dai  # DepthAIライブラリ
import cv2  # OpenCVライブラリ
import numpy as np  # Numpyライブラリ

class OakDChanNode(Node):
    def __init__(self):
        super().__init__('oakd_chan')  # ROS2ノード名を設定
        

        # DepthAIパイプラインの作成
        pipeline = dai.Pipeline()

        # カラーカメラノードの作成と設定
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(320, 240)  # 解像度を設定
        cam_rgb.setInterleaved(False)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)  # RGBカメラを使用
        cam_rgb.setFps(30)  # フレームレートを30FPSに設定

        # XLinkOutノードの作成（ホストへのデータ出力用）
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")  # ストリーム名を設定
        cam_rgb.preview.link(xout_rgb.input)  # カメラ出力をXLinkOutにリンク

        # デバイス初期化とパイプラインの開始
        self.device = dai.Device(pipeline)

        # ROS2のImageメッセージを送信するパブリッシャーの作成
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', 3)

        # 30FPSでタイマーを設定し、コールバックを実行
        self.timer = self.create_timer(0.03, self.timer_callback)

    def timer_callback(self):
        # デバイスからフレームを取得
        in_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False).get()

        # OpenCV形式のフレームに変換
        frame = in_rgb.getCvFrame()

        # ROS2 Imageメッセージの作成と設定
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()  # 現在時刻をタイムスタンプに設定
        msg.height = frame.shape[0]  # 画像の高さ
        msg.width = frame.shape[1]  # 画像の幅
        msg.encoding = 'bgr8'  # 画像のエンコーディング
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3  # 1行あたりのバイト数（RGBの3チャネル）
        msg.data = np.array(frame).tobytes()  # OpenCVフレームをバイト列に変換

        # トピックにメッセージを配信
        self.image_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)  # ROS2初期化
    oakd_chan_node = OakDChanNode()  # ノードのインスタンス化

    try:
        rclpy.spin(oakd_chan_node)  # ノードをスピン（実行）
    except KeyboardInterrupt:
        pass

    oakd_chan_node.destroy_node()  # ノードの破棄
    rclpy.shutdown()  # ROS2終了

if __name__ == '__main__':
    main()  # メイン関数の実行