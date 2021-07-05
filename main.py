import inspect
import os
import sys
import time
from glob import glob

from PySide2.QtCore import QSettings
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog
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
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.restore()
        self.load_weights_options()
        self.ui.button_source.clicked.connect(self.button_source_clicked)
        self.ui.button_start.clicked.connect(self.button_start_clicked)
        self.ui.button_target.clicked.connect(self.button_target_clicked)
        self.ui.button_abort.clicked.connect(self.button_abort_clicked)
        self.ui.combo_box_weights.currentIndexChanged.connect(
            self.setup_blurrer)

    def load_weights_options(self):
        self.ui.combo_box_weights.clear()
        for net_path in glob('./weights/*.pt'):
            clean_name = os.path.splitext(os.path.basename(net_path))[0]
            self.ui.combo_box_weights.addItem(clean_name)
        self.setup_blurrer()

    def setup_blurrer(self):
        """
        Create and connect a blurrer thread
        """
        weights_name = self.ui.combo_box_weights.currentText()
        # print(self.ui.combo_box_weights.currentText())
        self.blurrer = VideoBlurrer(weights_name)
        self.blurrer.setMaximum.connect(self.setMaximumValue)
        self.blurrer.updateProgress.connect(self.setProgress)
        self.blurrer.finished.connect(self.blurrer_finished)
        # msg_box = QMessageBox()
        # msg_box.setText(f'Successfully loaded {weights_name}.pt')
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
        print(value)
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
        inference_size = int(self.ui.combo_box_scale.currentText()[
                             :-1]) * 16 / 9  # ouch again
        batch_size=60
        # set up parameters 
        # add a for loop and determine the batch size
        parameters = {
            # "input_path":
            # "output_path": self.ui.line_target.text(),
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
        self.start = time.perf_counter()
        print(self.start)
        print("Blurrer started!")

    def button_source_clicked(self):
        """
        Callback for button_source
        """
        source_path, _ = QFileDialog.getOpenFileNames(
            self, "Open Video", "", "Video Files (*.mkv *.avi *.mov *.mp4)")
        # store full path
        self.source_paths=source_path
        # only store video file name
        self.newPaths=[]
        for path in source_path:
            self.newPaths.append(path.split('/')[-1])
        pathSting=' , '.join(self.newPaths)
        self.ui.line_source.setText(pathSting)

    def button_target_clicked(self):
        """
        Callback for button_target
        """
        target_path_pattern, _ = QFileDialog.getSaveFileName(
            self, "Save Video Naming style(input ? in place that use the same name as input file)", "", "Video Files (*.avi)")
        dir_name,file_patten=target_path_pattern.rsplit('/',1)
        self.target_paths=[]
        for path in self.newPaths:
            file_name=file_patten.replace('$',path.split('.')[0])
            self.target_paths.append('/'.join([dir_name,file_name]))
        self.ui.line_target.setText('/'.join([dir_name,file_patten.replace('$','{'+'Name'+'}')]))

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
        # switch case statement can be implemented
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
                if value:
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
            msg_box.setText(
                f"Video blurred successfully in {minutes} minutes and {seconds} seconds.")
        else:
            msg_box.setText("Blurring resulted in errors.")
        msg_box.exec_()
        self.end = time.perf_counter()
        dur = (self.end-self.start)
        dur_mins = dur//60
        dur_sec = dur % 60
        print(f'Use {dur_mins} Mins and {dur_sec} Secs')
        if not self.blurrer:
            self.setup_blurrer()
        self.ui.button_start.setEnabled(True)
        self.ui.button_abort.setEnabled(False)
        self.ui.progress.setValue(0)

    def save(self):
        """
        Save all relevant UI parameters
        """
        # switch case statement can be implemented
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
