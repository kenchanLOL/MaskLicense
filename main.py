
import inspect
import os
import sys
from PySide2.QtCore import QDirIterator, QSettings
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QListView, QAbstractItemView, QTreeView
from PySide2.QtWidgets import QSpinBox, QDoubleSpinBox, QLineEdit, QRadioButton, QMessageBox, QComboBox

from src.blurrer import VideoBlurrer
from src.ui_mainwindow import Ui_MainWindow
'''
    credits:
    Anonymizer
    https://github.com/understand-ai/anonymizer
    Dashcam cleaner
    https://github.com/tfaehse/DashcamCleaner​
    Pytorch-yolov4
    https://github.com/Tianxiaomo/pytorch-YOLOv4
    deepSORT-yolov4(tensorflow)
    https://github.com/theAIGuysCode/yolov4-deepsort
    deepSORT(pytorch) --not implemented, need further study
    https://github.com/ZQPei/deep_sort_pytorch
    
    Author & contact:
    Ken Chan (APAS 2021 suumer intern)
    chankwankin3@gmail.com

    General workflow:
        __init__ -> button_start_clicked -> ***Blurrer***

    Current bugs in this file:
    -some of the parameter in existing GUI is no longer used and need to be updated
        -Remove:
            -frame memory (deactivate it when deepSORT is activated)
            -Detection threshold 
            -ROI ratio
            -inference size
        -Add:
            -deepSORT activation (Yes/No)
            -threshold of size of plate need to be masked (float)
            -max_age and n_init of deepSORT tracker (int)
            -multiple files selection (Yes/No)
            - show the name of currently processing file
            -new process bar to show the process of all folders
    -For save and load setting data, the choice of network model is not saved. User need to manually select network every time 
    (maybe logic problem of load_weights_options, save and restore function)
    
'''

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
        weight_path_iter=QDirIterator(cur_dir+'/weights/raw_weight',['*.weights'],flags=QDirIterator.Subdirectories)
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
        
        the paramenters dictionary need to be updated
        only one blurrer is created for the whole task
        Dont blurrer for each video(initialization of blurrer takes long time and large memory to load 3 AI network)
        """

        self.ui.button_abort.setEnabled(True)
        self.ui.button_start.setEnabled(False)

        # read inference size (not useful now)
        inference_size = int(self.ui.combo_box_scale.currentText()[:-1]) * 16 / 9

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
        print("Blurring started!")

    def button_source_clicked(self):
        """
        Callback for button_source
        """
        # For single folder selection purpose

        # dialog=QFileDialog(self)
        # dialog.setFileMode()
        # source_dir_path=QFileDialog.getExistingDirectory(self,"Open File")
        # source_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mkv *.avi *.mov *.mp4)")
        # self.source_paths_iter=QDirIterator(source_dir_path,['*.mkv','*.avi' ,'*.mov' ,'*.mp4'],flags=QDirIterator.Subdirectories)

        # For multiple folders selection
        '''
            BUG:
            Single click folder may choose both folder selected and its parent folder
        '''
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
                    if value != None:
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
                if value and value == "true":
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
            hours=int(self.blurrer.result["elapsed_time"]//3600)
            minutes = int((self.blurrer.result["elapsed_time"]//60)%60)
            seconds = round(self.blurrer.result["elapsed_time"] % 60)
            msg_box.setText(f"Video blurred successfully in{hours}hours and {minutes} minutes {seconds} seconds.")
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
                index = obj.currentIndex()
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
