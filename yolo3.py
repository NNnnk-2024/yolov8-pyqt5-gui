import sys  # 导入系统模块
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QDesktopWidget, QHBoxLayout  # 导入PyQt5中的相关组件
from PyQt5.QtCore import QTimer, Qt  # 导入PyQt5中的定时器和Qt模块
from PyQt5.QtGui import QImage, QPixmap  # 导入PyQt5中的图像和像素映射模块
import cv2  # 导入OpenCV模块
from ultralytics import YOLO  # 从Ultralytics模块中导入YOLO对象

class YOLOv8Detector(QMainWindow):
    def __init__(self):  # 初始化方法
        super().__init__()  # 调用父类的初始化方法

        self.initUI()  # 初始化用户界面

        #self.model = YOLO('yolov8n.pt')
        self.model = YOLO('best.pt')# 加载YOLOv8模型

        self.timer = QTimer()  # 创建一个定时器对象
        self.timer.timeout.connect(self.detect_camera)  # 将定时器的超时信号连接到detect_camera方法

    def initUI(self):  # 初始化用户界面的方法
        self.setWindowTitle('YOLOv8 道路缺陷检测')  # 设置窗口标题
        self.setGeometry(100, 100, 1200, 800)  # 设置窗口大小和位置

        central_widget = QWidget(self)  # 创建一个中心窗口部件
        self.setCentralWidget(central_widget)  # 将中心窗口部件设置为主窗口的中心部件

        layout = QVBoxLayout()  # 创建一个垂直布局
        central_widget.setLayout(layout)  # 设置中心窗口部件的布局为垂直布局

        self.image_label = QLabel(self)  # 创建一个标签用于显示图像
        self.image_label.setAlignment(Qt.AlignCenter)  # 设置标签的对齐方式为居中
        layout.addWidget(self.image_label)  # 将标签添加到布局中

        button_layout = QHBoxLayout()  # 创建一个水平布局
        layout.addLayout(button_layout)  # 将水平布局添加到垂直布局中

        self.load_image_button = QPushButton('选择图片', self)  # 创建一个按钮用于加载图片
        self.load_image_button.setFixedSize(200, 80)  # 设置按钮的固定大小
        self.load_image_button.clicked.connect(self.load_image)  # 将按钮的点击信号连接到load_image方法
        button_layout.addWidget(self.load_image_button)  # 将按钮添加到水平布局中

        self.load_video_button = QPushButton('选择视频', self)  # 创建一个按钮用于加载视频
        self.load_video_button.setFixedSize(200, 80)  # 设置按钮的固定大小
        self.load_video_button.clicked.connect(self.load_video)  # 将按钮的点击信号连接到load_video方法
        button_layout.addWidget(self.load_video_button)  # 将按钮添加到水平布局中

        self.camera_button = QPushButton('打开摄像头', self)  # 创建一个按钮用于打开摄像头
        self.camera_button.setFixedSize(200, 80)  # 设置按钮的固定大小
        self.camera_button.clicked.connect(self.start_camera)  # 将按钮的点击信号连接到start_camera方法
        button_layout.addWidget(self.camera_button)  # 将按钮添加到水平布局中

        self.stop_camera_button = QPushButton('停止', self)  # 创建一个按钮用于停止
        self.stop_camera_button.setFixedSize(200, 80)  # 设置按钮的固定大小
        self.stop_camera_button.clicked.connect(self.stop_camera)  # 将按钮的点击信号连接到stop_camera方法
        button_layout.addWidget(self.stop_camera_button)  # 将按钮添加到水平布局中

        self.video_capture = None  # 初始化视频捕获对象为空

    def load_image(self):  # 加载图像的方法
        options = QFileDialog.Options()  # 创建文件对话框选项
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg *.jpeg)", options=options)  # 打开图像文件对话框并获取选择的文件名
        if file_name:  # 如果文件名不为空
            self.process_image(file_name)  # 调用process_image方法处理图像

    def load_video(self):  # 加载视频的方法
        options = QFileDialog.Options()  # 创建文件对话框选项
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)", options=options)  # 打开视频文件对话框并获取选择的文件名
        if file_name:  # 如果文件名不为空
            self.video_capture = cv2.VideoCapture(file_name)  # 创建视频捕获对象
            self.timer.start(30)  # 启动定时器，每30毫秒触发一次

    def start_camera(self):  # 启动摄像头的方法
        self.video_capture = cv2.VideoCapture(0)  # 创建视频捕获对象，参数为0表示打开默认摄像头
        self.timer.start(30)  # 启动定时器，每30毫秒触发一次

    def stop_camera(self):  # 停止摄像头的方法
        self.timer.stop()  # 停止定时器
        if self.video_capture:  # 如果视频捕获对象存在
            self.video_capture.release()  # 释放视频捕获对象
        self.image_label.clear()  # 清空图像标签

    def detect_camera(self):  # 处理摄像头捕获的帧的方法
        ret, frame = self.video_capture.read()  # 读取摄像头捕获的帧
        if ret:  # 如果成功读取到帧
            self.process_frame(frame)  # 调用process_frame方法处理帧

    def process_image(self, file_name):  # 处理图像的方法
        image = cv2.imread(file_name)  # 读取图像文件
        self.process_frame(image)  # 调用process_frame方法处理图像帧

    def process_frame(self, frame):
        results = self.model(frame)  # 使用YOLOv8模型处理帧，得到检测结果
        annotated_frame = results[0].plot()  # 绘制标注的帧

        # 将BGR图像转换为RGB并显示
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
        h, w, ch = rgb_image.shape  # 获取图像的高度、宽度和通道数
        bytes_per_line = ch * w  # 计算每行的字节数
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)  # 创建Qt图像对象
        pixmap = QPixmap.fromImage(qt_image)  # 使用Qt图像创建像素映射对象
        self.image_label.setPixmap(pixmap)  # 在标签上设置像素映射
        self.image_label.setScaledContents(True)  # 将图像自适应标签大小

if __name__ == '__main__':  # 主程序入口
    app = QApplication(sys.argv)  # 创建应用程序对象
    detector = YOLOv8Detector()  # 创建YOLOv8Detector对象
    detector.show()  # 显示主窗口
    sys.exit(app.exec_())  # 运行应用程序事件循环，等待退出命令
