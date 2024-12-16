from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from reportlab.pdfgen import canvas
import PIL.Image
import matplotlib.pyplot as plt
import cv2
import os
import pyodbc
import datetime
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import tkinter as tk
import math
import numpy as np
import pytesseract
import Preprocess

pdfmetrics.registerFont(TTFont('TimesNewRoman', 'Times New Roman.ttf'))
conx = pyodbc.connect(
    'DRIVER={ODBC Driver 18 for SQL Server};'
    'SERVER=localhost;'
    'Database=qlttn;'
    'UID=sa;'
    'PWD=1405_Hung;'
    'TrustServerCertificate=yes;'
)

cursor = conx.cursor()


class Tab1(Frame):
    def image_process(self, img):
        pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
        imgread = cv2.imread(img)
        img = cv2.resize(imgread, dsize=(0, 0), fx=2, fy=2.215)
        knn_model = self.load_knn_data()
        imgGrayscaleplate, imgThreshplate = self.preprocess_image(img)
        contours = self.find_contours(imgGrayscaleplate, imgThreshplate)

        if not contours:
            cropped_img = self.crop_and_preprocess_image(img)
            number_water = self.preprocess_and_segment_characters(cropped_img, knn_model)
        else:
            number_water = self.preprocess_and_segment_characters(img, knn_model, contours)

        if number_water:
            self.displayNumWater.delete(0, END)
            self.displayNumWater.insert(0, int(number_water))

    def load_knn_data(self):
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
        kNearest = cv2.ml.KNearest_create()
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        return kNearest

    def preprocess_image(self, img):
        return Preprocess.preprocess(img)

    def find_contours(self, imgGrayscaleplate, imgThreshplate):
        canny_image = cv2.Canny(imgThreshplate, 250, 255)
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    def crop_and_preprocess_image(self, img):
        height, width = img.shape[:2]
        left = int(width * 0.8 / 5)
        right = int(width * 4.5 / 5)
        bottom = int(height * 2.5 / 5)
        top = int(height * 1.2 / 5)
        cropped_img = img[top:bottom, left:right]
        gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
        imgBlurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        imgThresh = self.noise_removal(imgThresh)
        imgThresh = self.thin_font(imgThresh)
        cropped_img = cv2.resize(cropped_img, (0, 0), fx=2, fy=2)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=2, fy=2)
        return cropped_img, imgThresh

    def noise_removal(self, image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    def thin_font(self, image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    def preprocess_and_segment_characters(self, img, knn_model, contours=None):
        RESIZED_IMAGE_WIDTH = 20
        RESIZED_IMAGE_HEIGHT = 30
        number_water = ""
        if contours:
            for screenCnt in contours:
                roi, imgThresh = self.crop_and_align_image(img, screenCnt)
                cont, _ = self.find_characters(imgThresh)
                number_water = self.recognize_characters(roi, cont, knn_model, RESIZED_IMAGE_WIDTH,
                                                         RESIZED_IMAGE_HEIGHT)
                if number_water:
                    break
        else:
            cropped_img, imgThresh = img
            cont, _ = self.find_characters(imgThresh)
            number_water = self.recognize_characters(cropped_img, cont, knn_model, RESIZED_IMAGE_WIDTH,
                                                     RESIZED_IMAGE_HEIGHT)
        return number_water

    def crop_and_align_image(self, img, screenCnt):
        (x1, y1) = screenCnt[0, 0]
        (x2, y2) = screenCnt[1, 0]
        (x3, y3) = screenCnt[2, 0]
        (x4, y4) = screenCnt[3, 0]
        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        array.sort(reverse=True, key=lambda x: x[1])
        (x1, y1) = array[0]
        (x2, y2) = array[1]
        doi = abs(y1 - y2)
        ke = abs(x1 - x2)
        angle = math.atan(doi / ke) * (180.0 / math.pi)
        mask = np.zeros(img.shape[:2], np.uint8)
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        roi = img[topx:bottomx, topy:bottomy]
        imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2
        if x1 < x2:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
        else:
            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)
        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)
        return roi, imgThresh

    def find_characters(self, imgThresh):
        kerel1 = np.ones((1, 1), np.uint8)
        img_dilate = cv2.dilate(imgThresh, kerel1, iterations=1)
        kerel1 = np.ones((1, 1), np.uint8)
        img_erode = cv2.erode(img_dilate, kerel1, iterations=1)
        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        thre_mor = cv2.morphologyEx(img_erode, cv2.MORPH_DILATE, kerel3)
        img_blur = cv2.medianBlur(thre_mor, 1)
        cont, _ = cv2.findContours(img_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cont, _

    def recognize_characters(self, roi, cont, knn_model, RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT):
        char_x_ind = {}
        char_x = []
        height, width, _ = roi.shape
        roiarea = height * width
        Min_char = 0.01
        Max_char = 0.09
        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            perheight = h / height
            char_area = w * h
            if (Min_char * roiarea < char_area < Max_char * roiarea) and (
                    0.26 < ratiochar < 1.5) and w < h and 0.55 > perheight > 0.3:
                if x in char_x:
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind
        char_x = sorted(char_x)
        strFinalString = ""
        for i in char_x:
            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
            imgROI = thre_mor[y:y + h, x:x + w]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            npaROIResized = np.float32(npaROIResized)
            _, npaResults, _, _ = knn_model.findNearest(npaROIResized, k=3)
            strCurrentChar = str(chr(int(npaResults[0][0])))
            strFinalString += strCurrentChar
        number_water = int(strFinalString)
        if len(strFinalString) > 4:
            number_water = int(strFinalString[:-2])
        return str(number_water)

    def open_image(self):
        self.file_path = filedialog.askopenfilename()
        self.openPhoto = Image.open(self.file_path)
        self.resize = self.openPhoto.resize((400, 400), Image.LANCZOS)
        self.getPhoto = ImageTk.PhotoImage(self.resize)
        self.displayPhoto = Label(self, image=self.getPhoto)
        self.displayPhoto.image = self.getPhoto
        self.displayPhoto.place(relx=.62, rely=.2, relheight=.35, relwidth=.35)
        self.image_process(self.file_path)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.count = 0
        while True:
            self.ret, self.frame = self.cap.read()
            cv2.imshow("Camera", self.frame)
            k = cv2.waitKey(1)
            if k == ord('s'):
                self.count += 1
                img_name = "image_{}.jpg".format(self.count)
                cv2.imwrite('/home/nguyenthanhhung/Downloads/PYTHON/Python/QLTTN/QLTTN/data/image/' + img_name,
                            self.frame)
                messagebox.showinfo("Thông báo", "Ảnh đã được chụp")
                self.img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.img = PIL.Image.fromarray(self.img)
                self.img = self.img.resize((400, 400), PIL.Image.LANCZOS)
                self.img = PIL.ImageTk.PhotoImage(image=self.img)
                self.displayPhoto = Label(self, image=self.img)
                self.displayPhoto.image = self.img
                self.displayPhoto.place(relx=.62, rely=.2, relheight=.35, relwidth=.35)
                self.image_process('/home/nguyenthanhhung/Downloads/PYTHON/Python/QLTTN/QLTTN/data/image/' + img_name)
            elif k == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
