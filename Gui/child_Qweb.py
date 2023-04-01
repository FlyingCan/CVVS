import sys
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout,QDialog,QMainWindow

class child_qweb_Qdialog(QDialog,QWebEngineView):
    def __init__(self):
        super(child_qweb_Qdialog, self).__init__()
        self.resize(900, 700)

        self.back_btn = QPushButton(self)
        self.forward_btn = QPushButton(self)
        self.refresh_btn = QPushButton(self)
        self.zoom_in_btn = QPushButton(self)
        self.zoom_out_btn = QPushButton(self)
        self.url_le = QLineEdit(self)

        self.browser = QWebEngineView()

        self.h_layout = QHBoxLayout()
        self.v_layout = QVBoxLayout()

        self.layout_init()
        self.btn_init()
        self.le_init()
        self.browser_init()
        self.setWindowModality(Qt.NonModal)
        self.show()
    def layout_init(self):
        self.h_layout.setSpacing(0)
        self.h_layout.addWidget(self.back_btn)
        self.h_layout.addWidget(self.forward_btn)
        self.h_layout.addWidget(self.refresh_btn)
        self.h_layout.addStretch(2)
        self.h_layout.addWidget(self.url_le)
        self.h_layout.addStretch(2)
        self.h_layout.addWidget(self.zoom_in_btn)
        self.h_layout.addWidget(self.zoom_out_btn)
        self.v_layout.addLayout(self.h_layout)
        self.v_layout.addWidget(self.browser)
        self.setLayout(self.v_layout)
        self.setAcceptDrops(True)
        self.count =1
    def dragEnterEvent(self, evn):
        self.url_le.clear()
        print('web 鼠标拖入窗口了',self.count)
        self.count = self.count+1
        # 鼠标放开函数事件
        evn.accept()

    # 鼠标放开执行
    def dropEvent(self, evn):
        # 判断文件类型
        print('web页面:',evn.mimeData().urls())
        if evn.mimeData().hasImage():
            print('有图片')
        if evn.mimeData().hasText():
            print('鼠标放开了有文字')

    def dragMoveEvent(self, evn):
        pass
        # print('鼠标移入')
    def browser_init(self):
        self.browser.load(QUrl('https://www.bing.com/'))
        self.browser.urlChanged.connect(lambda: self.url_le.setText(self.browser.url().toDisplayString()))

    def btn_init(self):
        self.back_btn.setIcon(QIcon('./icons/back.png'))
        self.forward_btn.setIcon(QIcon('./icons/forward.png'))
        self.refresh_btn.setIcon(QIcon('./icons/refresh.png'))
        self.zoom_in_btn.setIcon(QIcon('./icons/zoomIn.png'))
        self.zoom_out_btn.setIcon(QIcon('./icons/zoomOut.png'))

        self.back_btn.clicked.connect(self.browser.back)
        self.forward_btn.clicked.connect(self.browser.forward)
        self.refresh_btn.clicked.connect(self.browser.reload)
        self.zoom_in_btn.clicked.connect(self.zoom_in_func)
        self.zoom_out_btn.clicked.connect(self.zoom_out_func)

    def le_init(self):
        self.url_le.setFixedWidth(700)
        self.url_le.setPlaceholderText('Search or enter website name')

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == Qt.Key_Return or QKeyEvent.key() == Qt.Key_Enter:
            if self.url_le.hasFocus():
                if self.url_le.text().startswith('https://') or self.url_le.text().startswith('http://'):
                    self.browser.load(QUrl(self.url_le.text()))
                else:
                    self.browser.load(QUrl('https://'+self.url_le.text()))

    def zoom_in_func(self):
        self.browser.setZoomFactor(self.browser.zoomFactor()+0.1)

    def zoom_out_func(self):
        self.browser.setZoomFactor(self.browser.zoomFactor()-0.1)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = child_qweb_Qdialog()
    demo.show()
    sys.exit(app.exec_())