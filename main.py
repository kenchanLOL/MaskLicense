import inspect
import os
import sys
from glob import glob
import typing

from PySide2.QtCore import QDirIterator, QSettings
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QListView, QAbstractItemView, QTreeView
from PySide2.QtWidgets import QSpinBox, QDoubleSpinBox, QLineEdit, QRadioButton, QMessageBox, QComboBox

from src.blurrer import VideoBlurrer
from src.ui_mainwindow import Ui_MainWindow


class MainWindow(QMainWindow):

    def __init__(self):
        """
        Constructor
        """
        self.receive_attempts = 0
        self.settings = QSettings("gui.ini", QSettings.IniFormat)
        self.blurrer = None
        self.source_list=[]
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.restore()
        self.load_weights_options()
        self.ui.button_source.clicked.connect(self.button_source_clicked)
        self.ui.button_start.clicked.connect(self.button_start_clicked)
        self.ui.button_target.clicked.connect(self.button_target_clicked)
        self.ui.button_abort.clicked.connect(self.button_abort_clicked)
        self.ui.combo_box_weights.currentIndexChanged.connect(self.setup_blurrer)

    def load_weights_options(self):
        self.ui.combo_box_weights.clear()
        cur_dir=os.path.split(__file__)[0]
        weight_path_iter=QDirIterator(cur_dir+'/weights',['*.pt'],flags=QDirIterator.Subdirectories)
        while weight_path_iter.hasNext():
            net_path=weight_path_iter.next()
            clean_name = os.path.splitext(os.path.basename(net_path))[0]
            self.ui.combo_box_weights.addItem(clean_name)
        self.setup_blurrer()

    def setup_blurrer(self):
        """
        Create and connect a blurrer thread
        """
        weights_name = self.ui.combo_box_weights.currentText()
        self.blurrer = VideoBlurrer(weights_name)
        self.blurrer.setMaximum.connect(self.setMaximumValue)
        self.blurrer.updateProgress.connect(self.setProgress)
        self.blurrer.finished.connect(self.blurrer_finished)
        # msg_box = QMessageBox()
        # msg_box.setText(f"Successfully loaded {weights_name}.pt")
        # msg_box.exec_()

    def button_abort_clicked(self):
        """
        Callback for button_abort
        """
        self.force_blurrer_quit()
        self.ui.progress.setValue(0)
        self.ui.button_start.setEnabled(True)
        self.ui.button_abort.setEnabled(False)
        self.setup_blurrer()

    def setProgress(self, value: int):
        """
        Set progress bar's current progress
        :param value: progress to be set
        """
        self.ui.progress.setValue(value)

    def setMaximumValue(self, value: int):
        """
        Set progress bar's maximum value
        :param value: value to be set
        """
        self.ui.progress.setMaximum(value)

    def button_start_clicked(self):
        """
        Callback for button_start
        """

        self.ui.button_abort.setEnabled(True)
        self.ui.button_start.setEnabled(False)

        # read inference size
        inference_size = int(self.ui.combo_box_scale.currentText()[:-1]) * 16 / 9 # ouch again

        # set up parameters
        parameters = {
            "input_path_iter_list": self.source_list,
            "input_path_Cur":'',
            "output_path": self.target_path,
            "blur_size": self.ui.spin_blur.value(),
            "blur_memory": self.ui.spin_memory.value(),
            "threshold": self.ui.double_spin_threshold.value(),
            "roi_multi": self.ui.double_spin_roimulti.value(),
            "inference_size": inference_size
        }
        if self.blurrer:
            self.blurrer.parameters = parameters
            self.blurrer.start()
        else:
            print("No blurrer object!")
        print("Blurrer started!")

    def button_source_clicked(self):
        """
        Callback for button_source
        """
        # dialog=QFileDialog(self)
        # dialog.setFileMode()
        # source_dir_path=QFileDialog.getExistingDirectory(self,"Open File")
        # source_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mkv *.avi *.mov *.mp4)")
        # self.source_paths_iter=QDirIterator(source_dir_path,['*.mkv','*.avi' ,'*.mov' ,'*.mp4'],flags=QDirIterator.Subdirectories)
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.DirectoryOnly)
        file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        file_view = file_dialog.findChild(QListView, 'listView')

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.MultiSelection)
            f_tree_view = file_dialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

        if file_dialog.exec():
            source_dir_paths = file_dialog.selectedFiles()

        self.ui.line_source.setText(' , '.join(source_dir_paths))
        self.source_list=[]
        for path in source_dir_paths:
            self.source_list.append(QDirIterator(path,['*.mkv','*.avi' ,'*.mov' ,'*.mp4'],flags=QDirIterator.Subdirectories))

    def button_target_clicked(self):
        """
        Callback for button_target
        """
        self.target_path= QFileDialog.getExistingDirectory(self, "Save File")
        self.ui.line_target.setText(self.target_path)

    def force_blurrer_quit(self):
        """
        Force blurrer thread to quit
        """
        if self.blurrer.isRunning():
            self.blurrer.terminate()
            self.blurrer.wait()

    def restore(self):
        """
        Restores relevent UI settings from ini file
        """
        for name, obj in inspect.getmembers(self.ui):
            if isinstance(obj, QSpinBox):
                name = obj.objectName()
                value = self.settings.value(name)
                if value:
                    obj.setValue(int(value))

            if isinstance(obj, QDoubleSpinBox):
                name = obj.objectName()
                value = self.settings.value(name)
                if value:
                    obj.setValue(float(value))

            if isinstance(obj, QLineEdit):
                name = obj.objectName()
                value = self.settings.value(name)
                if name=='line_source':
                    temp_list=value.split(' , ')
                    for path in temp_list:
                        self.source_list.append(QDirIterator(path,['*.mkv','*.avi' ,'*.mov' ,'*.mp4'],flags=QDirIterator.Subdirectories))

                    obj.setText(value)
                elif name=='line_target':
                    self.target_path=value
                    obj.setText(value)

            if isinstance(obj, QRadioButton):
                name = obj.objectName()
                value = self.settings.value(name)
                if value and value == "true":  # ouch...
                    obj.setChecked(True)

            if isinstance(obj, QComboBox):
                name = obj.objectName()
                value = self.settings.value(name)
                if value:
                    index = obj.findText(value)
                    if index == -1:
                        obj.insertItems(0, [value])
                        index = obj.findText(value)
                        obj.setCurrentIndex(index)
                    else:
                        obj.setCurrentIndex(index)

    def blurrer_finished(self):
        """
        Create a new blurrer, setup UI and notify the user
        """
        msg_box = QMessageBox()
        if self.blurrer and self.blurrer.result["success"]:
            minutes = int(self.blurrer.result["elapsed_time"] // 60)
            seconds = round(self.blurrer.result["elapsed_time"] % 60)
            msg_box.setText(f"Video blurred successfully in {minutes} minutes and {seconds} seconds.")
        else:
            msg_box.setText("Blurring resulted in errors.")
        msg_box.exec_()
        if not self.blurrer:
            self.setup_blurrer()
        self.ui.button_start.setEnabled(True)
        self.ui.button_abort.setEnabled(False)
        self.ui.progress.setValue(0)

    def save(self):
        """
        Save all relevant UI parameters
        """
        for name, obj in inspect.getmembers(self.ui):
            if isinstance(obj, QSpinBox):
                name = obj.objectName()
                value = obj.value()
                self.settings.setValue(name, value)

            if isinstance(obj, QDoubleSpinBox):
                name = obj.objectName()
                value = obj.value()
                self.settings.setValue(name, value)

            if isinstance(obj, QLineEdit):
                name = obj.objectName()
                value = obj.text()
                self.settings.setValue(name, value)

            if isinstance(obj, QRadioButton):
                name = obj.objectName()
                value = obj.isChecked()
                self.settings.setValue(name, value)

            if isinstance(obj, QComboBox):
                index = obj.currentIndex()  # get current index from combobox
                value = obj.itemText(index)
                self.settings.setValue(name, value)

    def closeEvent(self, event):
        """
        Overload closeEvent to shut down blurrer and save UI settings
        :param event:
        """
        self.force_blurrer_quit()
        self.save()
        print("saved settings")
        QMainWindow.closeEvent(self, event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())