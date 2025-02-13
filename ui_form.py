# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFormLayout,
    QFrame, QGraphicsView, QHBoxLayout, QLabel,
    QMainWindow, QMenu, QMenuBar, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 800)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayoutWidget_2 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(140, 200, 521, 551))
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.graphicsView = QGraphicsView(self.verticalLayoutWidget_2)
        self.graphicsView.setObjectName(u"graphicsView")

        self.verticalLayout_3.addWidget(self.graphicsView)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.classifyButton = QPushButton(self.verticalLayoutWidget_2)
        self.classifyButton.setObjectName(u"classifyButton")

        self.horizontalLayout_3.addWidget(self.classifyButton)

        self.extractButton = QPushButton(self.verticalLayoutWidget_2)
        self.extractButton.setObjectName(u"extractButton")

        self.horizontalLayout_3.addWidget(self.extractButton)


        self.verticalLayout_3.addLayout(self.horizontalLayout_3)

        self.verticalLayoutWidget_3 = QWidget(self.centralwidget)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(140, 19, 521, 161))
        self.verticalLayout_4 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.pushButtonFile = QPushButton(self.verticalLayoutWidget_3)
        self.pushButtonFile.setObjectName(u"pushButtonFile")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.pushButtonFile.sizePolicy().hasHeightForWidth())
        self.pushButtonFile.setSizePolicy(sizePolicy)

        self.verticalLayout_4.addWidget(self.pushButtonFile)

        self.label = QLabel(self.verticalLayoutWidget_3)
        self.label.setObjectName(u"label")
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)

        self.verticalLayout_4.addWidget(self.label)

        self.formLayout_3 = QFormLayout()
        self.formLayout_3.setObjectName(u"formLayout_3")
        self.comboBox = QComboBox(self.verticalLayoutWidget_3)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy1)

        self.formLayout_3.setWidget(0, QFormLayout.LabelRole, self.comboBox)

        self.doubleSpinBox = QDoubleSpinBox(self.verticalLayoutWidget_3)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        sizePolicy1.setHeightForWidth(self.doubleSpinBox.sizePolicy().hasHeightForWidth())
        self.doubleSpinBox.setSizePolicy(sizePolicy1)
        self.doubleSpinBox.setDecimals(0)
        self.doubleSpinBox.setMaximum(100.000000000000000)

        self.formLayout_3.setWidget(0, QFormLayout.FieldRole, self.doubleSpinBox)

        self.pushButton = QPushButton(self.verticalLayoutWidget_3)
        self.pushButton.setObjectName(u"pushButton")

        self.formLayout_3.setWidget(1, QFormLayout.SpanningRole, self.pushButton)


        self.verticalLayout_4.addLayout(self.formLayout_3)

        self.verticalLayoutWidget = QWidget(self.centralwidget)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(140, 180, 521, 20))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.line = QFrame(self.verticalLayoutWidget)
        self.line.setObjectName(u"line")
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)

        self.verticalLayout.addWidget(self.line)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QMenuBar(MainWindow)
        self.menuBar.setObjectName(u"menuBar")
        self.menuBar.setGeometry(QRect(0, 0, 800, 21))
        self.menuMain = QMenu(self.menuBar)
        self.menuMain.setObjectName(u"menuMain")
        self.menuNext = QMenu(self.menuBar)
        self.menuNext.setObjectName(u"menuNext")
        MainWindow.setMenuBar(self.menuBar)

        self.menuBar.addAction(self.menuMain.menuAction())
        self.menuBar.addAction(self.menuNext.menuAction())

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.classifyButton.setText(QCoreApplication.translate("MainWindow", u"Classify Slice", None))
        self.extractButton.setText(QCoreApplication.translate("MainWindow", u"Extract Slice", None))
        self.pushButtonFile.setText(QCoreApplication.translate("MainWindow", u"Select DCM", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"File Name:", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("MainWindow", u"LCC", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("MainWindow", u"RCC", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("MainWindow", u"LMLO", None))
        self.comboBox.setItemText(3, QCoreApplication.translate("MainWindow", u"RMLO", None))

        self.comboBox.setCurrentText("")
        self.comboBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Select View", None))
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"Load Slice", None))
        self.menuMain.setTitle(QCoreApplication.translate("MainWindow", u"Main", None))
        self.menuNext.setTitle(QCoreApplication.translate("MainWindow", u"Next", None))
    # retranslateUi

