import rclpy #文字を認識するコード
from rclpy.node import Node
from sensor_msgs.msg import Image
import depthai as dai  # DepthAIライブラリ
from cv_bridge import CvBridge
import cv2
import pytesseract
import numpy as np
import math

class AlphabetDetectionNode(Node):
    def __init__(self):
        super().__init__('alphabet_detection_node')

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

        in_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False).get()
        # OpenCV形式のフレームに変換
        frame = in_rgb.getCvFrame() #表示用
        frame1 = in_rgb.getCvFrame()#認識用
        frame2 = in_rgb.getCvFrame()#歪み補正チェック用

        # 1. 緑色の範囲をHSVで定義
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #1.5 ぼかす
        blurred_image = cv2.GaussianBlur(hsv, (5, 5), 0)

        
        # 緑色の範囲を定義
        # lower_green = np.array([40, 40, 40])   # HSVで緑色の下限
        # upper_green = np.array([80, 255, 255])  # HSVで緑色の上限

        # ”青”色の範囲を定義
        lower_green = np.array([90, 60, 60])   # HSVで緑色の下限
        upper_green = np.array([150, 255, 255])  # HSVで緑色の上限
        
        # 緑色部分のマスクを作成
        mask_green = cv2.inRange(blurred_image, lower_green, upper_green)
        
        # 2. マスクを使って緑の箱を検出
        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_box = None
        for contour in contours:
            # 緑色の箱の輪郭を検出
            if cv2.contourArea(contour) > 1000:  # 面積が小さいものは無視
                green_box = cv2.boundingRect(contour)  # 緑色の箱のバウンディングボックスを取得
                cv2.rectangle(frame, (green_box[0], green_box[1]), 
                            (green_box[0] + green_box[2], green_box[1] + green_box[3]), 
                            (0, 255, 0), 2)  # 緑色の箱に矩形を描画緑

        if green_box is not None:
            wx, wy, ww, wh = green_box
            # 3. 緑色の箱の内部から白い紙を検出
            roi = frame1[wy:wy+wh, wx:wx+ww]
            blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 白色の範囲をHSVで定義
            lower_white = np.array([0, 0, 150])
            upper_white = np.array([180, 60, 255])#彩度を40から60に上げるとつくばチャレンジの文字の影響が小さくなった。
            
            # # 白い部分のマスクを作成
            # mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
            
            # # 4. 白い部分を検出し、白い紙を強調
            # contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # for contour in contours_white:
            #     if cv2.contourArea(contour) > 300:  # 白い紙の面積のしきい値
            #         # ROIの相対座標を元のフレームの絶対座標に変換
            #         contour_global = contour + np.array([[wx, wy]])  # ROIの座標オフセットを追加
            #         # 元のフレームに輪郭を描画
            #         cv2.drawContours(frame, [contour_global], -1, (0, 0, 255), 2)  # 赤い輪郭を描画


    ###ここからコピペ
                    
            
            # 5. hsvに変換した画像を　白(1)or白以外(0)　の２値画像に変換
            white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
            cv2.imshow("white_mask", white_mask)

            cv2.moveWindow("white_mask", 0, 0)                # 3 グレーエッジ
            # 5.5 2値画像の白色部分を拡張 
            # dilated_image = cv2.dilate(white_mask, np.ones((2, 2), np.uint8), iterations=1)

            # 7. 2値画像の輪郭を検出
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 8. 輪郭or輪郭以外の2値画像を定義
            gray = cv2.cvtColor(hsv_roi, cv2.COLOR_BGR2GRAY)
            binary_mask = np.zeros_like(gray)  # グレースケール画像と同じサイズでゼロ初期化

            # 9. 閾値以上の面積の輪郭だけ白で塗りつぶす。forで輪郭の数だけ繰り返す
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 面積が小さい輪郭はノイズとして除外
                    # ROIの相対座標を元のフレームの絶対座標に変換
                    contour_global = contour + np.array([[wx, wy]])  # ROIの座標オフセットを追加
                    # 元のフレームに輪郭を描画
                    cv2.drawContours(binary_mask, [contour_global], -1, (255), thickness=cv2.FILLED)  # 輪郭を白で塗りつぶす

            cv2.imshow("binary_mask", binary_mask)
            cv2.moveWindow("binary_mask", 0, 500) 
            #コピペゾーン https://qiita.com/sitar-harmonics/items/ac584f99043574670cf3
            for i, cnt in enumerate(contours):
                    # 輪郭の周囲に比例する精度で輪郭を近似する
                    cnt_global = cnt + np.array([[wx, wy]])  # ROIの座標オフセットを追加
                    arclen = cv2.arcLength(cnt_global, True)
                    approx = cv2.approxPolyDP(cnt_global, arclen*0.1, True) #0.02だと低すぎる。0.1だとかなり余裕。

                    #四角形の輪郭は、近似後に4つの頂点があります。
                    #比較的広い領域が凸状になります。

                    # 凸性の確認 
                    area = abs(cv2.contourArea(approx))
                    if approx.shape[0] == 4 and area > 500 and cv2.isContourConvex(approx) :
                            # 四角形のバウンディングボックスを取得
                        # x, y, w, h = cv2.boundingRect(approx)#

                        # 横長判定 (例えば、幅が高さの1.5倍以上であれば横長と判定)
                        maxCosine = 0

                        for j in range(2, 5):
                            # 辺間の角度の最大コサインを算出
                            cosine = abs(angle(approx[j%4], approx[j-2], approx[j-1]))
                            maxCosine = max(maxCosine, cosine)

                        # すべての角度の余弦定理が小さい場合（すべての角度は約90度です）次に、quandrangeを書き込みます
                        # 結果のシーケンスへの頂点
                        if maxCosine < 0.3 :
                            # 四角判定!!
                            rcnt = approx.reshape(-1,2)
                            # cv2.polylines(frame, [rcnt], True, (0,0,255), thickness=2, lineType=cv2.LINE_8)
                            cv2.fillPoly(frame, [rcnt], (0,0,255), lineType=cv2.LINE_8)

                            ##以降歪み補正
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
                            (tl, tr, br, bl) = rect

                            # A5サイズの縦横比（横長）
                            width = 210  # mm
                            height = 148  # mm
                            aspect_ratio = width / height

                            # 横長A5を基準とした新しい座標を計算
                            maxWidth = int(1000)  # 横方向のサイズを任意に設定
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

                            # 結果を保存・表示
                            cv2.imwrite("warped_a5.jpg", warped)
                            cv2.imshow("Warped A5", warped)
                            cv2.moveWindow("Waroed A5", 0, 0)   

    
###ここまでコピペ


                            # cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                            # グレースケールに変換
                            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                            # ノイズ除去 (ガウシアンブラー)
                            # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                            blurred = cv2.medianBlur(gray, 5)
                            # 二値化 (Otsuの手法)
                            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                                    # 輪郭を抽出
                            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # 上だけブラー＆文字認識直前に上40%をトリミング
                            height_b, width_b = binary.shape[:2]

                            # 上部10%の範囲を取得
                            top_blur = int(height_b * 0.5)
                            top_area = binary[:top_blur, :]  # 上部10%

                            # ぼかし処理を適用 (GaussianBlurを使用)
                            blurred_top = cv2.GaussianBlur(top_area, (61, 61), 0)

                            # 元画像にぼかし部分を反映
                            binary[:top_blur, :] = blurred_top
                            # 紙の上40%をトリミング
                            top_trim = int(height_b * 0.42)
                            bottom_trim = int(width_b * 0.05)  # 左右5%をトリミング（必要に応じて調整）
                            bottom_trim = int(height_b * 0.01)  # 下部5%をトリミング

                            # トリミング画像の作成
                            trimmed_image = binary[top_trim:height_b - bottom_trim, bottom_trim:width_b - bottom_trim]

                            cv2.imshow("moji", trimmed_image)
                            cv2.moveWindow("moji", 500, 0) 

                            if contours:
                                # # 面積が最大の輪郭を取得
                                # largest_contour = max(contours, key=cv2.contourArea)
                                # cx, cy, cw, ch = cv2.boundingRect(largest_contour) #文字のボックス座標 "c"hara

                                # # 最大輪郭の外接矩形で画像を切り出す
                                # roi = frame[cy:cy+ch, cx:cx+cw]
                                custom_config = r'--psm 10'
                                # 切り出した部分にOCRを適用
                                text = pytesseract.image_to_string(trimmed_image, lang='eng', config = custom_config).strip()

                            # text = pytesseract.image_to_string(binary)
                            # OCRで文字を認識
                                # self.get_logger().info(f'文字はどこだ')

                                if text:
                                    self.get_logger().info(f'Recognized text: {text}')

                                # 最大文字の外接矩形を描画（デバッグ用）
                                # cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)
        # 画像を表示
        cv2.imshow("Image of Detected Lines", frame)

        cv2.waitKey(1)

# pt0-> pt1およびpt0-> pt2からの
# ベクトル間の角度の余弦(コサイン)を算出
def angle(pt1, pt2, pt0) -> float:
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