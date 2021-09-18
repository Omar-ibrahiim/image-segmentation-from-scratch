from GUI import Ui_MainWindow
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import Thresholding as TH
import kmean as km
import regionGrowing as rg
import agglo as ag
import MEANSHIFT as ms

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.actionLoad_Image.triggered.connect(self.Load_img)

        self.ui.threshold_comboBox.activated.connect(lambda current_text:
                            self.ThresholdType(self.ui.threshold_comboBox.currentText()))
        self.ui.seg_comboBox.activated.connect(lambda current_text:
                            self.SegmentationType(self.ui.seg_comboBox.currentText()))


    def ThresholdType(self, Th_type ):
        img = self.img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if Th_type == 'Otsu global':
            processed_img = TH.Thresholding_types(Th_type, img)
            self.display_processed_img(processed_img)
        elif Th_type == 'Otsu local':
            processed_img = TH.Thresholding_types(Th_type, img)
            self.display_processed_img(processed_img)
        elif Th_type == 'Optimal global':
            processed_img = TH.Thresholding_types(Th_type, img)
            self.display_processed_img(processed_img)
        else:  # Optimal local
            processed_img = TH.Thresholding_types(Th_type, img)
            self.display_processed_img(processed_img)


    def SegmentationType(self, seg_algorithm):
        seg_img = self.img
        if seg_algorithm == 'k-means':
            if (seg_img.shape[-1] == 3):
                seg_img = km.clust_rgb(seg_img)
            else:
                seg_img = km.clust_gray(seg_img)
                
            self.display_processed_img(seg_img)
        elif seg_algorithm == 'region growing':
            seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
            seeds = [rg.getxy(10,10),rg.getxy(30,100),rg.getxy(50,180)]
            seg_img = rg.regionGrow(seg_img,seeds,8)
            seg_img = np.array(seg_img).astype(np.uint8)*255
            self.display_processed_img(seg_img)
        elif seg_algorithm == 'agglomerative':
            pixels = seg_img.reshape((-1,3))
            nclust = 2
            agglo = ag.Agglomerative(k=nclust, initial_k=30)
            agglo.fit(pixels)
            new_img = [[agglo.predict_center(list(pixel)) for pixel in row] for row in seg_img]
            new_img = np.array(new_img, np.uint8)
                        
            self.display_processed_img(new_img)
        else: 
            # seg_img = # put your function
            seg_img = cv2.cvtColor( seg_img, cv2.COLOR_RGB2LUV )
            meanshift = ms.meanShiftSeg( seg_img, 7 )
            seg_img = meanshift.applyMeanShift()
            
            self.display_processed_img(seg_img)


    def display_processed_img(self, img):
        cv2.imwrite("result.jpg", img)
        noisy_img = QPixmap("result.jpg").scaled(500, 500)
        self.ui.filtered_image.setPixmap(QPixmap(noisy_img))


    def Load_img(self):
        filename, _ = QFileDialog.getOpenFileNames(self, "Select file(s)", " ", "Images (*.png *.xpm *.jpg)")
        self.img = cv2.imread(f'{filename[0]}')

        self.Display_Original_img()


    def Display_Original_img(self):

        cv2.imwrite("Original_gray_img.jpg", self.img)
        gry_result = QPixmap("Original_gray_img.jpg").scaled(500, 500)

        self.ui.original_image.setPixmap(QPixmap(gry_result))


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()

