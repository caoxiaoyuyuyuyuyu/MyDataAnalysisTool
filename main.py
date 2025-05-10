import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    # 创建应用实例
    app = QApplication(sys.argv)

    # 设置全局样式（可选）
    app.setStyle('Fusion')

    # 创建并显示主窗口
    window = MainWindow()
    window.show()

    # 运行应用主循环
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()