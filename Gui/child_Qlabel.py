from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QDialog, QMainWindow, QMessageBox
import sys
from PyQt5.QtWidgets import QApplication  ,QWidget, QLabel
from PyQt5.QtGui import QPainter, QPixmap, QIcon, QBrush, QColor, QPen, QCursor
from PyQt5.QtCore import Qt , QPoint
class Qlabel_lineEdit_child(QDialog):
    def __init__(self):
        super(Qlabel_lineEdit_child, self).__init__()
        self.resize(120,30)
        self.lineEdit = QtWidgets.QLineEdit(self)  # 提示文本框
        self.lineEdit.setObjectName("Change_name")
        self.lineEdit.setText('')
        self.lineEdit.setGeometry(QtCore.QRect(0, 0, 119, 29))
        self.exec_()
    def getLineEdit_text(self):
        return self.lineEdit.text()
class Qlabel_Paint_child(QDialog):
    def __init__(self,qlabel:QLabel,width:int,height:int):
        super(Qlabel_Paint_child, self).__init__()
        self.drawAble_width = width
        self.drawAble_height = height
        self.resize(width+110,height)
        self.label = qlabel

        self.setWindowTitle("Draw")
        self.setWindowModality(Qt.NonModal)
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.save = False
        self.penSize_lineEdit = QtWidgets.QLineEdit(self)  #提示文本框
        self.penSize_lineEdit.setObjectName("lineEdit")
        self.penSize_lineEdit.setText('Pensize:')
        self.penSize_lineEdit.setReadOnly(True)
        self.penSize_lineEdit.setGeometry(QtCore.QRect(self.width()-100, 0, 71, 20))
        self.spinBox_penSize_spinbox = QtWidgets.QSpinBox(self)
        self.spinBox_penSize_spinbox.setGeometry(QtCore.QRect(self.width()-70, 20, 71, 20))
        self.spinBox_penSize_spinbox.setMinimum(1)
        self.spinBox_penSize_spinbox.setMaximum(50)
        self.spinBox_penSize_spinbox.setProperty("value", 2)
        self.spinBox_penSize_spinbox.setObjectName("spinBox_penSize")
        #颜色选择
        self.PenColor = QtWidgets.QLineEdit(self)  #提示文本框
        self.PenColor.setObjectName("colorEdit")
        self.PenColor.setText('ColorSelect:')
        self.PenColor.setReadOnly(True)
        self.PenColor.setGeometry(QtCore.QRect(self.width() - 105, 40, 105, 20))
        self.color_box = QtWidgets.QComboBox(self)
        self.color_box.setGeometry(QtCore.QRect(self.width()-70, 60, 71, 20))
        self.color_box.setObjectName("color_box")
        self.color_box.addItem("")
        self.color_box.addItem("")
        self.color_box.addItem("")
        self.color_box.addItem("")
        self.color_box.addItem("")
        self.color_box.setItemText(0,'white')
        self.color_box.setItemText(1, 'black')
        self.color_box.setItemText(2, 'red')
        self.color_box.setItemText(3, 'green')
        self.color_box.setItemText(4, 'blue')


        self.setMouseTracking(True)  #跟踪鼠标
        self.exec_()
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'close warning', "save the drawn picture in parent label?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.save = True
            event.accept()

    def paintEvent(self, event):
        pixmap = self.label.pixmap()
        qp = QPainter(pixmap)
        # 根据鼠标指针前后两个位置绘制直线
        #### 设置笔参
        pen = QPen()
        pen.setStyle(Qt.DashDotLine)
        pen.setWidth(self.spinBox_penSize_spinbox.value())
        if self.color_box.currentText() == 'white':
            color = QColor(255,255,255)
        elif self.color_box.currentText() == 'black':
            color = QColor(0, 0, 0)
        elif self.color_box.currentText() == 'red':
            color = QColor(255, 0, 0)
        elif self.color_box.currentText() == 'green':
            color = QColor(0, 255, 0)
        elif self.color_box.currentText() == 'blue':
            color = QColor(0, 0, 255)
        pen.setBrush(color)
        qp.setPen(pen)
        qp.drawLine(self.lastPoint, self.endPoint)
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.label.pixmap())

    def mousePressEvent(self, event):
        # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):
        # 鼠标左键按下的同时移动鼠标
        x = event.localPos().x()
        y = event.localPos().y()
        if x < self.drawAble_width and y <self.drawAble_height:
            # self.setCursor(QCursor(QPixmap('icons/pen.png').scaled(5, 5)))  # 设置鼠标样式
            pass
        else:
            self.setCursor(QCursor())
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    def mouseReleaseEvent(self, event):
        # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

