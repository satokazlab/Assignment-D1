import rclpy #文字を認識するコード
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import pytesseract

class AlphabetDetectionNode(Node):
    def __init__(self):
        super().__init__('alphabet_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        text = pytesseract.image_to_string(cv_image)
        self.get_logger().info(f'Recognized text: {text}')

def main(args=None):
    rclpy.init(args=args)
    node = AlphabetDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()