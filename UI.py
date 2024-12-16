import subprocess
from sys import platform
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

cursor=conx.cursor()

""" 

Process an image to extract and recognize numbers from water meter readings.
            Processing steps:
            1. Image Pre-processing:
                - Setup Tesseract OCR path
                - Resize input image
                - Convert to grayscale
                - Apply adaptive thresholding
                - Apply Canny edge detection
                - Apply dilation to connect edges
            2. License Plate Detection:
                - Find contours in the image
                - Filter contours based on area and shape
                - Extract potential number plate regions
            3. If no plate detected (count = 0):
                - Crop image manually based on fixed ratios
                - Apply noise removal and font thinning
                - Perform character segmentation
            4. If plate detected:
                - Draw contours around the plate
                - Calculate plate angle
                - Rotate and align the plate
                - Resize plate region
            5. Character Recognition:
                - Load KNN model data
                - Segment individual characters
                - Filter characters based on size and ratio
                - Apply KNN classification
                - Convert ASCII results to text
                - Display results in UI
            Parameters:
            -----------
            img : str
                Path to the input image file
            Important Parameters:
            -------------------
            ADAPTIVE_THRESH_BLOCK_SIZE : int
                Block size for adaptive thresholding (19)
            ADAPTIVE_THRESH_WEIGHT : int
                Weight for adaptive thresholding (9)
            Min_char : float
                Minimum character size ratio (0.01)
            Max_char : float
                Maximum character size ratio (0.09)
            RESIZED_IMAGE_WIDTH : int
                Width for character recognition (20)
            RESIZED_IMAGE_HEIGHT : int
                Height for character recognition (30)
            Returns:
            --------
            None
                Updates the UI display with recognized number 

"""


class Tab1 (Frame):
    def image_process(self,img):
        pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
        """ 
         
        Dòng mã này thiết lập đường dẫn đến tệp thực thi của Tesseract OCR. 
        Tesseract OCR là một công cụ nhận dạng ký tự quang học (OCR) mã nguồn mở, 
        được sử dụng để chuyển đổi hình ảnh chứa văn bản thành văn bản có thể chỉnh sửa.
          
            """
        ADAPTIVE_THRESH_BLOCK_SIZE = 19
        ADAPTIVE_THRESH_WEIGHT = 9

        """ ADAPTIVE_THRESH_BLOCK_SIZE:

Đây là kích thước của khối (block size) được sử dụng để tính toán ngưỡng cho mỗi pixel. Kích thước này thường là một số lẻ (ví dụ: 3, 5, 7, 9, 11, ...). Trong trường hợp này, giá trị là 19.
Kích thước khối càng lớn thì ngưỡng tính toán sẽ càng mượt mà, nhưng có thể làm mất chi tiết nhỏ trong ảnh.
ADAPTIVE_THRESH_WEIGHT:

Đây là giá trị trọng số (weight) được trừ đi từ giá trị trung bình của khối để xác định ngưỡng. Trong trường hợp này, giá trị là 9.
Trọng số này giúp điều chỉnh ngưỡng để phù hợp hơn với các điều kiện ánh sáng khác nhau trong ảnh. 

Ngưỡng tính toán (thresholding) là một kỹ thuật trong xử lý ảnh để phân đoạn ảnh thành các vùng khác nhau 
dựa trên mức độ sáng của các pixel. Mục tiêu của ngưỡng tính toán là 
chuyển đổi ảnh xám thành ảnh nhị phân, trong đó các pixel có giá trị sáng hơn 
ngưỡng sẽ được đặt thành một giá trị (thường là trắng), và các pixel tối hơn ngưỡng sẽ được đặt thành giá trị khác (thường là đen).


"""

        n = 1
        Min_char = 0.01
        Max_char = 0.09
        RESIZED_IMAGE_WIDTH = 20
        RESIZED_IMAGE_HEIGHT = 30
        imgread = cv2.imread(img)
        img = cv2.resize(imgread, dsize=(0, 0),fx=2,fy=2.215)
        ###############################################################
        # load dữ liệu của KNN
        ######## Upload KNN model ######################
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
        """ 
         
        Đoạn mã này thay đổi hình dạng của mảng npaClassifications từ một mảng 1 chiều thành một 
        mảng 2 chiều với một cột duy nhất. Điều này thường được thực hiện để chuẩn bị 
        dữ liệu cho các thuật toán học máy hoặc các thao tác xử lý dữ liệu khác yêu cầu 
        dữ liệu đầu vào có hình dạng cụ thể.
          
            """
        npaClassifications = npaClassifications.reshape(
            (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
        kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        #########################
        # sử dụng file preprocess để sử lý ảnh
        ################ Image Preprocessing #################
        imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
        """ 
         lấy ra ảnh xám và ảnh nhị phân từ hàm preprocess trong file Preprocess.py
           """
        canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
        """ 
         dùng thuật toán Canny để tìm cạnh của ảnh, giúp tìm ra các đường viền trong ảnh
            Phân đoạn đối tượng:

Đường viền giúp phân đoạn các đối tượng trong ảnh, chẳng hạn như các ký tự trên biển số xe. Điều này giúp tách biệt các ký tự khỏi nền và các đối tượng khác trong ảnh.
Xác định vùng quan tâm (ROI):

Đường viền giúp xác định vùng quan tâm trong ảnh, chẳng hạn như vùng chứa biển số xe. Điều này giúp tập trung vào các vùng quan trọng và bỏ qua các vùng không liên quan.
           """
        kernel = np.ones((3, 3), np.uint8)
        """ 
         
          uses NumPy to create this matrix, and np.uint8 specifies that the data type of the matrix elements is 8-bit unsigned integers.
This kernel is used in the dilation process to determine the shape and size of the neighborhood for the dilation operation.
Dilation là một phép toán hình thái học (morphological operation) được sử dụng để mở rộng các vùng sáng (foreground) trong ảnh nhị phân. Phép toán này làm cho các đối tượng sáng trong ảnh trở nên lớn hơn và có thể kết nối các vùng sáng gần nhau.

           
             """
        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # Dilation
        # cv2.imshow("imgThreshplate",imgThreshplate)
        ###########################################
        count=0
        # vẽ đường viền chứa khung số
        ###### Draw contour and filter out the license plate  #############
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Lấy 10 contours có diện tích lớn nhất
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        # biến lưu giá trị đường viền
        screenCnt = []
        for c in contours:
            peri = cv2.arcLength(c, True)  # Tính chu vi
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h
            if (len(approx) == 4 and (w-h)>150 and w>30  and h>20 and 20<x and y>20):                         #and <450 and 200<w<600 and 10<h<300 and 20<y<700) điều kiện để lọc đường viền chỉ còn lại khung số
                screenCnt.append(approx)
                [x, y, w, h] = cv2.boundingRect(approx.copy())
                count+=1


        # count=0 có nghĩa là không tìm thấy khung số phải cắt tay
        if count==0:

            detected = 0
            height, width = img.shape[:2]
            # Tính toán giá trị tọa độ cần thiết để cắt ảnh
            left = int(width * 0.8/5)
            right = int(width * 4.5/5)
            bottom = int(height * 2.5/5)
            top = int(height * 1.2/5)
            # Cắt ảnh theo tỷ lệ đã tính toán được
            cropped_img = img[top:bottom, left:right]
            # cv2.imshow("cropped_img",cropped_img)
            # sử lý ảnh#######################################################
            gray_image=cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
            imgBlurred = cv2.GaussianBlur(gray_image, (5,5), 0)
            imgThresh = cv2.adaptiveThreshold(imgBlurred,
                                            255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV,
                                            11,
                                            2)
            def noise_removal(image):
                kernel = np.ones((1, 1), np.uint8)
                image = cv2.dilate(image, kernel, iterations=1) #làm đậm các vùng sáng, loại bỏ điểm nhiễu nhỏ
                kernel = np.ones((1, 1), np.uint8)
                image = cv2.erode(image, kernel, iterations=1) #làm nhạt các vùng sáng, loại bỏ điểm nhiễu nhỏ còn sót lại
                image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) #Phép đóng là sự kết hợp của giãn nở và xói mòn. Nó giúp lấp đầy các lỗ nhỏ trong các vùng sáng của ảnh.
                image = cv2.medianBlur(image, 3) #làm mờ ảnh bằng cách sử dụng bộ lọc trung vị. Đây là một kỹ thuật xử lý ảnh phổ biến để giảm nhiễu trong ảnh.
                return (image)
            no_noise = noise_removal(imgThresh)
            def thin_font(image):
                import numpy as np
                image = cv2.bitwise_not(image) #Đảo ngược màu hình ảnh
                kernel = np.ones((2,2),np.uint8)
                image = cv2.erode(image, kernel, iterations=1) # Quá trình erode sẽ làm giảm kích thước của các vùng màu trắng trong hình ảnh.
                image = cv2.bitwise_not(image) #Đảo ngược màu hình ảnh
                return (image)
            eroded_image = thin_font(no_noise)

            roi=cropped_img
            roi = cv2.resize(roi, (0, 0),fx=2,fy=2) #tăng kích thước hình ảnh lên 2 lần
            imgThresh = cv2.resize(imgThresh, (0, 0),fx=2,fy=2) #tăng kích thước hình ảnh binary lên 2 lần

            #################### Prepocessing and Character segmentation ####################
            kerel1=np.ones((1,1),np.uint8)
            img_dilate=cv2.dilate(imgThresh,kerel1,iterations=1)
            kerel1=np.ones((1,1),np.uint8)
            img_erode=cv2.erode(img_dilate,kerel1,iterations=1)
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)) #tạo kernel hình chữ nhật 1x1
            thre_mor = cv2.morphologyEx(img_erode, cv2.MORPH_DILATE, kerel3)
            img_blur=cv2.medianBlur(thre_mor,3) #mờ ảnh, giảm nhiễu
            canny = cv2.Canny(img_blur, 100, 255) #thực hiện phát hiện biên cạnh (edge detection) trên ảnh img_blur với ngưỡng dưới là 100 và ngưỡng trên là 255.
            # vẽ đường viền của từng số
            cont, hier = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cont = sorted(cont, key=cv2.contourArea, reverse=True)[:20]  # Lấy 10 contours có diện tích lớn nhất
            cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)
            ##################### Filter out characters #################
            char_x_ind = {} #lưu vị trí của từng số
            char_x = [] #Lưu giá trị x của từng số
            height, width, _ = roi.shape
            roiarea = height * width
            # vòng lặp chạy đường viền từng số
            for ind, cnt in enumerate(cont):
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                char_area = w * h

                if (0.0039 * roiarea < char_area < 0.008 * roiarea)and w<h and (0.25 < ratiochar < 1.5):# and 50<h<300 and 30<w<150
                    if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind
            ############ Character recognition ##########################
            char_x = sorted(char_x)
            strFinalString = ""
            number_water = ""
            array_number_water=[]
            # chạy tungdwf khung số đã cắt và so sánh vs KNN
            for i in char_x:
                """ 
                 
                  Hàm này dùng để tìm tọa độ và kích thước của hình chữ nhật bao quanh contour
Trả về 4 giá trị: x, y là tọa độ điểm góc trên bên trái, w, h là chiều rộng và chiều cao
Cần thực hiện để biết được vị trí chính xác của ký tự để có thể cắt và xử lý
                   
                     """
                (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                """ 
                 
                    Vẽ hình chữ nhật xanh lá cây (0,255,0) lên ảnh roi
x,y: tọa độ góc trên bên trái
w,h: chiều rộng và cao của hình chữ nhật
thickness=2: độ dày viền 2 pixel
Cần thực hiện để đánh dấu trực quan vị trí của ký tự được nhận diện      
                    
                      """
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                """ 
                 
                  Cắt từng ký tự riêng biệt để xử lý
Resize về kích thước chuẩn để đưa vào model nhận dạng
Cần thực hiện để chuẩn bị dữ liệu cho bước nhận dạng ký tự
                   
                     """
                imgROI = thre_mor[y:y + h, x:x + w]  # Crop the characters

                imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image

                """ 
                 
                  Chuyển ma trận 2D thành vector 1D
Đổi kiểu dữ liệu sang float32
Cần thực hiện để phù hợp với định dạng đầu vào của model KNN
                    
                      """
                npaROIResized = imgROIResized.reshape(
                    (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

                """ 
                 
                  
                Sử dụng model KNN để dự đoán ký tự
Chuyển kết quả số thành ký tự ASCII
Ghép các ký tự vào chuỗi kết quả
Cần thực hiện để hoàn tất quá trình nhận dạng và lưu kết quả
                   
                    
                      """
                npaROIResized = np.float32(npaROIResized)
                _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)
                strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
                number_water = number_water + strCurrentChar



            number_water=int(number_water)
            number_water=str(number_water)
            if(len(number_water)>=4):
                number_water=number_water[:-2]
            if number_water!="":
                self.displayNumWater.delete(0, END)
                self.displayNumWater.insert(0,int(number_water))
            # roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
            # cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        else:
            detected = 1
        # điều kiện khi cắt dc khung và có thể khong cắt được với mấy ảnh khác
        if detected == 1:
            # vòng lặp chạy từng khung chứa số mà nó nhận được
            for screenCnt in screenCnt:
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
                # cv2.imshow("screenCnt: ",img)
                ############## tìm góc của biển số ##################### vị trí các giá trị biến x.y    2   1
                (x1, y1) = screenCnt[0, 0]                              #                               3   4
                (x2, y2) = screenCnt[1, 0]
                (x3, y3) = screenCnt[2, 0]
                (x4, y4) = screenCnt[3, 0]

                array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                # print(array)
                sorted_array = array.sort(reverse=True, key=lambda x: x[1])
                (x1, y1) = array[0]
                (x2, y2) = array[1]
                doi = abs(y1 - y2)
                ke = abs(x1 - x2)
                angle = math.atan(doi / ke) * (180.0 / math.pi) #biến lưu góc nghiên của biển số
                ####################################

                ########## Crop out the license plate and align it to the right angle ################

                mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
                new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))

                roi = img[topx:bottomx, topy:bottomy]
                # cv2.imshow("imgThresh",roi)
                imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
                # cv2.imshow("imgThresh",imgThresh)
                ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

                if x1 < x2:
                    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
                else:
                    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

                roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
                imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
                # tăng kích thước khung đẫ cắt lên 3 lần
                roi = cv2.resize(roi, (0, 0),fx=3,fy=3)
                imgThresh = cv2.resize(imgThresh, (0, 0),fx=3,fy=3)

                ####################################

                #################### Prepocessing and Character segmentation ####################
                kerel1=np.ones((1,1),np.uint8)
                img_dilate=cv2.dilate(imgThresh,kerel1,iterations=1)
                kerel1=np.ones((1,1),np.uint8)
                img_erode=cv2.erode(img_dilate,kerel1,iterations=1)
                kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                thre_mor = cv2.morphologyEx(img_erode, cv2.MORPH_DILATE, kerel3)
                img_blur=cv2.medianBlur(thre_mor,1)
                # tìm viền từng số
                cont, hier = cv2.findContours(img_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số
                # cv2.imshow("imgThresh",roi)
                ##################### Filter out characters #################
                char_x_ind = {}
                char_x = []
                height, width, _ = roi.shape
                roiarea = height * width
                for ind, cnt in enumerate(cont):
                    (x, y, w, h) = cv2.boundingRect(cont[ind])
                    ratiochar = w / h
                    perheight=h/height

                    char_area = w * h
                    if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.26 < ratiochar < 1.5) and w<h and 0.55>perheight>0.3: #and perheight>0.5 and 70<h<260 and 30<w<150
                        if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                            x = x + 1
                        char_x.append(x)
                        char_x_ind[x] = ind

                ############ Character recognition ##########################
                char_x = sorted(char_x)
                strFinalString = ""
                number_water = ""
                array_number_water =[]
                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    imgROI = thre_mor[y:y + h, x:x + w]  # Crop the characters

                    imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize image
                    npaROIResized = imgROIResized.reshape(
                        (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

                    npaROIResized = np.float32(npaROIResized)
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,k=3)  # call KNN function find_nearest;
                    strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)
                    # cv2.imshow("imgThresh",roi)
                    number_water = number_water + strCurrentChar
                #n là số khung chứa số
                n = n + 1
                number_water=int(number_water)
                number_water=str(number_water)
                if(len(number_water)>4):
                    number_water=number_water[:-2]
                if number_water!="":
                    self.displayNumWater.delete(0, END)
                    self.displayNumWater.insert(0,int(number_water))
                    break
                roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.waitKey(0)
    def open_image(self):
        # Hiển thị file dialog để người dùng chọn một file ảnh
        self.file_path = filedialog.askopenfilename()

        self.openPhoto = Image.open(self.file_path)
        self.resize=self.openPhoto.resize((400,400),Image.LANCZOS)
        self.getPhoto = ImageTk.PhotoImage(self.resize)
        # Hiển thị ảnh
        self.displayPhoto = Label(self, image=self.getPhoto)
        self.displayPhoto.image = self.getPhoto
        self.displayPhoto.place(relx=.62,rely=.2,relheight=.35,relwidth=.35)
        self.image_process(self.file_path)
    def open_camera(self):

        self.cap = cv2.VideoCapture(0)
        self.count = 0
        while True:
            # Chụp ảnh
            self.ret, self.frame = self.cap.read()
            # Hiển thị hình ảnh lên màn hình
            cv2.imshow("Camera", self.frame)
            # Xử lý sự kiện ấn phím
            k = cv2.waitKey(1)
            # ấn phím s để chụp ảnh
            if k == ord('s'):
                self.count+=1
                # Lưu ảnh
                img_name = "image_{}.jpg".format(self.count)
                cv2.imwrite('data/image'+img_name, self.frame)
                # Hiện thông báo ảnh đã được chụp
                messagebox.showinfo("Thông báo", "Ảnh đã được chụp")
                # Chuyển ảnh từ OpenCV sang PIL.Image
                self.img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.img = PIL.Image.fromarray(self.img)
                # Thay đổi kích thước và chuyển định dạng ảnh
                self.img = self.img.resize((400,400), PIL.Image.LANCZOS)
                self.img = PIL.ImageTk.PhotoImage(image=self.img)
                # Hiển thị ảnh
                self.displayPhoto = Label(self, image=self.img)
                self.displayPhoto.image = self.img
                self.displayPhoto.place(relx=.62,rely=.2,relheight=.35,relwidth=.35)
            # ấn q để đóng cửa sổ camera
            elif k == ord('q'):
                break
        # Đóng camera và tất cả cửa sổ hiển thị 
        self.cap.release()
        cv2.destroyAllWindows()

    def cleanEntry(self):

        self.nameUserText.delete(0,self.nameUserText.index(END))
        self.phoneUserText.delete(0,self.phoneUserText.index(END))
        self.addressUserText.delete(0,self.addressUserText.index(END))
        self.Number_water_af_Text.config(state="normal")
        self.Number_water_af_Text.delete(0,self.Number_water_af_Text.index(END))
        self.displayNumWater.config(state="normal")
        self.displayNumWater.delete(0,self.displayNumWater.index(END))
        self.priceWaterText.config(state="normal")
        self.priceWaterText.delete(0,self.priceWaterText.index(END))
        self.openPhoto = Image.open("./default.png")
        self.resize=self.openPhoto.resize((400,400),Image.LANCZOS)
        self.getPhoto = ImageTk.PhotoImage(self.resize)
        self.displayPhoto = Label(self, image=self.getPhoto)
        self.displayPhoto.image = self.getPhoto
        self.displayPhoto.place(relx=.62,rely=.2,relheight=.35,relwidth=.35)

    def clickBut(self):
        if self.nameUserText.get()== "" or self.phoneUserText.get()=="" or self.addressUserText.get()=="" \
                                        or self.priceWaterText.get()=="" or self.displayNumWater.get()=="":
            messagebox.showinfo("Thông tin", "Chưa nhập đủ thông tin")
        else :
            try:
                countSdt=cursor.execute("select SoDienThoai from QLKH_TN").fetchall()
                lenght=len(countSdt)
                #print(lenght)
                messagebox.showinfo("Thông tin", "Đã lưu")
                #lưu vào cơ sở dữ liệu
                cursor.execute("insert QLKH_TN values (?,?,?,?,?,?,?,?,?)",lenght+1,self.nameUserText.get(),\
                                     self.addressUserText.get(), self.phoneUserText.get(),datetime.date.today(),self.Number_water_af_Text.get(),\
                                         self.displayNumWater.get(),self.priceWaterText.get(),1)
                conx.commit()
                self.cleanEntry()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi lưu thông tin khách hàng: {e}")

    def calWater(self):
        bN=self.Number_water_af_Text.get()
        dN=self.displayNumWater.get()
        # Check if either value is null or empty
        if not bN or not dN:  # This checks if bN or dN is None or an empty string
            messagebox.showinfo("Lỗi", "Chưa nhập số nước trước đó!")
            return
        pWT=self.priceWaterText
        numWater=int(dN)-int(bN)
        if bN =="" or dN=="":
            messagebox.showinfo("Thông tin", "Vui lòng nhập đủ thông tin")
        else:
            if numWater <= 10:
                S= 5.973 * numWater
            elif numWater>10 and numWater<=20:
                S=5.973 * 10 + 7.052*(numWater-10)
            elif numWater>20 and numWater<=30:
                S=5.973 *10 + 7.052*10 + 8.669*(numWater-20)
            else:
                S=5.973 *10 + 7.052*10 + 8.669*10 + 15.929*(numWater-30)

        self.Number_water_af_Text.config(state="disabled")
        self.displayNumWater.config(state="disabled")
        pWT.insert(0,str(round((S+S*(15/100)),0))+"00")
        pWT.config(state="disabled",justify=CENTER)

    def xuat_hoa_don(self):
        if self.nameUserText.get()== "" or self.phoneUserText.get()=="" or self.addressUserText.get()=="" \
                                        or self.priceWaterText.get()=="" or self.displayNumWater.get()=="":
            messagebox.showinfo("Thông tin", "Chưa nhập đủ thông tin")
        else:
            try:
                self.now_date = datetime.datetime.now()
                self.formatt_date = self.now_date.strftime("%d/%m/%Y")
                countSdt=cursor.execute("select SoDienThoai from QLKH_TN").fetchall()
                lenght=len(countSdt)
                info=cursor.execute("select id,Ngay from QLKH_TN where SoDienThoai =?",self.phoneUserText.get()).fetchall()
                self.pdf_name=f'HD_{self.phoneUserText.get()}.pdf'
                self.hoa_don = canvas.Canvas(self.pdf_name)
                self.hoa_don.setFont('TimesNewRoman', 16)
                self.hoa_don.setTitle("Water Bill Manager")
                self.hoa_don.drawString(250, 750, "Hóa đơn tiền nước")
                self.hoa_don.drawString(350, 710, "Đến ngày: " + self.formatt_date)
                if info:
                    time=info[len(info)-1].Ngay.strftime("%d/%m/%Y")
                    self.hoa_don.drawString(150, 710, "Từ ngày: " + time)
                    self.hoa_don.drawString(100, 670, "Mã số khách hàng: KH"+str(info[0].id))
                else:
                    self.hoa_don.drawString(150, 710, "Từ ngày: " )
                    self.hoa_don.drawString(100, 670, "Mã số khách hàng: KH"+str(lenght+1))
                self.hoa_don.drawString(100, 620, "Họ và tên: " + self.nameUserText.get())
                self.hoa_don.drawString(100, 570, "Số điện thoại: " + self.phoneUserText.get())
                self.hoa_don.drawString(100, 520, "Địa chỉ: " + self.addressUserText.get())
                data = [
                        ['Số Nước Tháng Này','Số Nước Tháng Trước','Thành Tiền'],
                        [self.displayNumWater.get()+" m3",self.Number_water_af_Text.get()+ " m3",f'{self.priceWaterText.get()} VNĐ']
                        ]
                table = Table(data)

                table.setStyle(TableStyle([
                                        ('BACKGROUND', (0,0), (-1,0), colors.green),
                                        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                                        ('ALIGN',(0,0),(-1,-1),'CENTER'),
                                        ('FONTNAME', (0, 0), (-1, -1), 'TimesNewRoman'),
                                        ('FONTSIZE', (0,0), (-1,0), 14),
                                        ('BOTTOMPADDING', (0,0), (-1,0), 12),
                                        ('GRID',(0,0),(-1,-1),1,colors.black)]))
                table.wrapOn(self.hoa_don, 0, 0)
                table.drawOn(self.hoa_don, 150,450)

                pdf_dir = os.path.join(os.getcwd(), "HoaDon") # Tạo đường dẫn mới
                # pdf_dir = "HoaDon" # Tạo đường dẫn mới
                if not os.path.exists(pdf_dir): # Nếu thư mục không tồn tại, tạo thư mục mới
                    os.makedirs(pdf_dir)
                os.chdir(pdf_dir) # Thay đổi đường dẫn hiện tại
                # self.pdf_name = f'HD_{self.phoneUserText.get()}.pdf'
                self.hoa_don.save()
                pdf_path = os.path.join(pdf_dir, self.pdf_name)

                if platform == 'Darwin':  # macOS
                    subprocess.call(('open', pdf_path))
                elif platform == 'Windows':  # Windows
                    os.startfile(pdf_path)
                else:  # Linux variants
                    subprocess.call(('xdg-open', pdf_path))
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi xuất hóa đơn: {e}")

    def placeTab1(self):
        self.update()

        self.labTab1.place(relx=.1,y=0,relwidth=.9)
        self.nameUser.place(relx=.06,rely=.17,relwidth=.19)
        self.nameUserText.place(relx=.095,rely=.225,relwidth=.4,relheight=.06)
        self.phoneUser.place(relx=.07,rely=.315,relwidth=.19)
        self.phoneUserText.place(relx=.095,rely=.375,relwidth=.4,relheight=.06)
        self.addressUser.place(relx=.07,rely=.439,relwidth=.15)
        self.addressUserText.place(relx=.095,rely=.495,relwidth=.4,relheight=.06)
        self.priceWater.place(relx=.07,rely=.700,relwidth=.3)
        self.priceWaterText.place(relx=.095,rely=.750,relwidth=.4,relheight=.06)
        self.imagePhoto.place(relx=.6,rely=.15,relwidth=.2)
        self.displayPhoto.place(relx=.62,rely=.2,relheight=.35,relwidth=.35)
        self.displayNumWater.place(relx=.635,rely=.560,relwidth=.3,relheight=.06)
        self.butChImage.place(relx=.66,rely=.660,relwidth=.25,relheight=.06)
        # self.butTakeImage.place(relx=.79,rely=.660,relwidth=.15,relheight=.06)
        self.butSave.place(relx=.32,rely=0.9)
        self.butPrint.place(relx=.55,rely=0.9)
        self.Number_water_af.place(relx=.07,rely=.560,relwidth=.3)
        self.Number_water_af_Text.place(relx=.095,rely=.620,relwidth=.4,relheight=.06)
        self.butCalPrice.place(relx=.66,rely=.740,relwidth=.25,relheight=.06)

    def validate_entry_chuoi(self, new_value):
        if all(char.isalpha() or char.isspace() for char in new_value) or new_value == "":
            return True
        else:
            return False

    def validate_entry(self,new_value):
        if new_value.isdigit() or new_value == "":
            return True
        else:
            return False

    def __init__(self,master):
        super().__init__(master)
        self.vcmd = (master.register(self.validate_entry), '%P')
        self.vcmdchuoi = (master.register(self.validate_entry_chuoi), '%P')

        self.labTab1=Label(self, text="Quản lý tính tiền nước\t",padx=325,pady=30, font = ("Helvetica Bold",30))
        self.nameUser=Label(self,text="Họ và tên:", font=("Helvetica ",20))
        self.nameUserText= Entry(self, width= 30,font=("Helvetica ",20),)
        self.nameUserText.configure(validate="key",validatecommand=self.vcmdchuoi, invalidcommand=master.bell)
        self.nameUserText.focus()
        self.phoneUser=Label(self,text="Số điện thoại:",font=("Helvetica ",20))
        self.phoneUserText=Entry(self,textvariable=NUMERIC, width=30,font=("Helvetica ",20))
        self.phoneUserText.configure(validate="key",validatecommand=self.vcmd, invalidcommand=master.bell)
        self.addressUser=Label(self,text="Địa chỉ:",font=("Helvetica",20))
        self.addressUserText=Entry(self,width=30,font=("Helvetica ",20))
        self.priceWater=Label(self,text="Số tiền nước ( VNĐ ):",font=("Helvetica",20))
        self.priceWaterText=Entry(self,width=30,font=("Helvetica ",20))
        self.imagePhoto=Label(self,text="Ảnh chụp:",font=("Helvetica",20))
        self.openPhoto = Image.open("./default.png")
        self.resize=self.openPhoto.resize((400,400),Image.LANCZOS)
        self.getPhoto = ImageTk.PhotoImage(self.resize)
        self.displayPhoto = Label(self, image=self.getPhoto)
        self.displayPhoto.image = self.getPhoto
        self.displayNumWater=Entry(self, font=("Helvetica",19),justify=CENTER)
        self.displayNumWater.configure(validate="key",validatecommand=self.vcmd, invalidcommand=master.bell)
        self.Number_water_af=Label(self,text="Số nước kỳ trước ( m3 ):", font=("Helvetica ",20))
        self.Number_water_af_Text= Entry(self, width= 30,font=("Helvetica ",20),)
        self.Number_water_af_Text.configure(validate="key",validatecommand=self.vcmd, invalidcommand=master.bell)
        self.butChImage=Button(self, text="Chọn ảnh", font=("Helvetica",20,'bold'),border=1.5,relief="solid",background='#333333',foreground='white',command=self.open_image)
        # self.butTakeImage=Button(self, text="Chụp ảnh", font=("Helvetica",20,'bold'),border=1.5,relief="solid",background='#333333',foreground='white',command=self.open_camera)
        self.butSave=Button(self, text="Lưu thông tin", font=("Helvetica",20,'bold'),border=1.5,relief="solid",background='#33FF66',foreground='#333333',command=self.clickBut)
        self.butPrint=Button(self, text="Xuất hóa đơn", font=("Helvetica",20,'bold'),border=1.5,relief="solid",background='#FF9900',foreground='#333333',command=self.xuat_hoa_don)
        self.butCalPrice=Button(self,text="Tính tiền ",font=("Helvetica",20,'bold'),background='#CCCCCC',foreground='#333333',command=self.calWater)

        master.bind("<Configure>",self.placeTab1())


class Tab2 (Frame):
    list_KhachHang=[]

    def findUser(self):
        info=cursor.execute("select * from QLKH_TN where SoDienThoai = ? and TrangThai = 1 ",self.findInfo.get()).fetchall()
        if len(info)==0:
            messagebox.showinfo("Thông tin", "Không có khách hàng nào")
        else :
            for i in range(len(info)) :
                self.list_KhachHang.append((info[i].HoTen,info[i].DiaChi,info[i].SoDienThoai,\
                                            info[i].Ngay.strftime("%d/%m/%Y"),str(int(info[i].SoNuocThangTruoc)),\
                                            str(int(info[i].SoNuocThangNay)),str(round(float(info[i].TienNuoc),0))+"00\n VNĐ"))

        self.find_info()
        self.list_KhachHang.clear()

    def find_restore_user(self):
        cursor.execute("update QLKH_TN set TrangThai = 1 where TrangThai=0 and SoDienThoai = ?",self.findInfo.get())
        conx.commit()
        info=cursor.execute("select * from QLKH_TN where SoDienThoai = ? and TrangThai = 1 ",self.findInfo.get()).fetchall()
        if info:
            messagebox.showinfo("Thông tin", "Khôi phục thành công")
            info.clear()
        else:
            messagebox.showinfo("Thông tin", "Khôi phục thất bại")

    def find_del_user(self):
        cursor.execute("update QLKH_TN set TrangThai = 0 where TrangThai=1 and SoDienThoai = ?",self.findInfo.get())
        conx.commit()
        info=cursor.execute("select * from QLKH_TN where SoDienThoai = ? and TrangThai = 0 ",self.findInfo.get()).fetchall()
        if info:
            messagebox.showinfo("Thông tin", "Xóa thành công")
            info.clear()
        else:
            messagebox.showinfo("Thông tin", "Xóa thất bại")

    def find_del_history(self):
        info=cursor.execute("select * from QLKH_TN where SoDienThoai = ? and TrangThai = 1 ",self.findInfo.get()).fetchall()
        for i in range(len(info)) :
                self.list_KhachHang.append(info[i].id)
        if self.list_KhachHang:
            a=len(self.list_KhachHang)-len(self.list_KhachHang)
            h=self.list_KhachHang[a]
            cursor.execute("update QLKH_TN set TrangThai = 0 where TrangThai=1 and id = ?",h)
            conx.commit()
            messagebox.showinfo("Thông tin", "Đã xóa")
            self.list_KhachHang.clear()
        else : messagebox.showinfo("Thông tin", "Không có lịch sử để xóa")

    def find_fix(self):
        self.t=Tabs(self.master)
        self.t.tab1.cleanEntry()
        info=cursor.execute("select * from QLKH_TN where SoDienThoai = ? and TrangThai = 1",self.findInfo.get()).fetchall()
        for i in range(len(info)) :
                self.list_KhachHang.append((info[i].HoTen,info[i].DiaChi,info[i].SoDienThoai,\
                                            info[i].Ngay.strftime("%d/%m/%Y"),str(int(info[i].SoNuocThangTruoc)),\
                                            str(int(info[i].SoNuocThangNay)),str(round(float(info[i].TienNuoc),))+"00\n VNĐ"))
        h=len(self.list_KhachHang)-1

        self.t.tab1.nameUserText.insert(0,str(self.list_KhachHang[h][0]))
        self.t.tab1.addressUserText.insert(0,str(self.list_KhachHang[h][1]))
        self.t.tab1.phoneUserText.insert(0,"0"+str(int(self.list_KhachHang[h][2])))
        self.t.tab1.Number_water_af_Text.insert(0,str(self.list_KhachHang[h][5]))
        self.list_KhachHang.clear()

    def find_info(self):
        value=self.findInfo.get()
        if value != '':
            self.big_frame = tk.Frame(self.Frame_info,border=1.3,relief="solid")
            self.big_frame.place(relx=0,rely=0,relwidth=.975,relheight=1)
            if self.show_find==True:
                # tạo vùng canvas
                self.canvas = tk.Canvas(self.big_frame)
                self.canvas.pack(fill="both",expand=True)
                # canvas.place(relheight=1,relwidth=1,relx=0,rely=0)
                # tạo thanh cuộn
                self.scrollbar = tk.Scrollbar(self.Frame_info, orient="vertical", command=self.canvas.yview)
                self.scrollbar.pack(side="right",fill="y")
                # liên kết thanh cuộn với vùng canvas
                self.canvas.configure(yscrollcommand=self.scrollbar.set)

                # độ rộng frame = độ rộng của canvas
                frame_width = int(self.canvas.winfo_width())
                frame_height = int(self.canvas.winfo_height())

                # tạo frame con để chứa các Label
                self.frame = tk.Frame(self.canvas,height=frame_height,width=frame_width)
                self.frame.pack(fill="both",expand=True)
                # tỉ lệ khung hình
                ratio = 1.5
                self.canvas.create_window((0, 0), window=self.frame, anchor=NW)
                # hàm reset độ dài khi thay đổi
                def on_configure(event):
                    # Tính toán kích thước mới của frame
                    new_width = event.width
                    new_height = int(new_width * ratio)
                    # Cập nhật kích thước của frame
                    self.frame.config(width=new_width, height=new_height)

                # Thêm callback cho sự kiện Configure
                self.canvas.bind('<Configure>', on_configure)
                vitri_y=0

                for kh in range (len(self.list_KhachHang)):

                    name_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][0]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    diachi_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][1]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    sdt_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][2]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    ngay_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][3]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    sotienkytruoc_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][4]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    sotienkynay_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][5]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    tiennuoc_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][6]),font=("Helvetica",12,"bold"),border=1,relief="solid")

                    # print(name_label)
                    name_label.place(relwidth=.18,height=100,relx=0,y=vitri_y)
                    diachi_label.place(relwidth=.18,height=100,relx=.18,y=vitri_y)
                    sdt_label.place(relwidth=.14,height=100,relx=.36,y=vitri_y)
                    ngay_label.place(relwidth=.12,height=100,relx=.50,y=vitri_y)
                    sotienkytruoc_label.place(relwidth=.13,height=100,relx=.62,y=vitri_y)
                    sotienkynay_label.place(relwidth=.13,height=100,relx=.75,y=vitri_y)
                    tiennuoc_label.place(relwidth=.12,height=100,relx=.88,y=vitri_y)
                    vitri_y=vitri_y+100

                # lệnh bắt đầu cuộn
                self.frame.update_idletasks()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                self.show_find=False
            else:
                self.big_frame.destroy()
                self.canvas.destroy()
                self.scrollbar.pack_forget()
                self.big_frame = tk.Frame(self.Frame_info,border=1.3,relief="solid")
                self.big_frame.place(relx=0,rely=0,relwidth=.975,relheight=1)
                self.canvas = tk.Canvas(self.big_frame)
                self.canvas.pack(fill="both",expand=True)
                # canvas.place(relheight=1,relwidth=1,relx=0,rely=0)
                # tạo thanh cuộn
                self.scrollbar = tk.Scrollbar(self.Frame_info, orient="vertical", command=self.canvas.yview)
                self.scrollbar.pack(side="right",fill="y")
                # liên kết thanh cuộn với vùng canvas
                self.canvas.configure(yscrollcommand=self.scrollbar.set)

                # độ rộng frame = độ rộng của canvas
                frame_width = int(self.canvas.winfo_width())
                frame_height = int(self.canvas.winfo_height())

                # tạo frame con để chứa các Label
                self.frame = tk.Frame(self.canvas,height=frame_height,width=frame_width)
                self.frame.pack(fill="both",expand=True)
                # tỉ lệ khung hình
                ratio = 1.5

                self.canvas.create_window((0, 0), window=self.frame, anchor=NW)
                # hàm reset độ dài khi thay đổi
                def on_configure(event):
                    # Tính toán kích thước mới của frame
                    new_width = event.width
                    new_height = int(new_width * ratio)
                    # Cập nhật kích thước của frame
                    self.frame.config(width=new_width, height=new_height)

                # Thêm callback cho sự kiện Configure
                self.canvas.bind('<Configure>', on_configure)
                vitri_y=0
                for kh in range (len(self.list_KhachHang)):

                    name_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][0]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    diachi_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][1]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    sdt_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][2]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    ngay_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][3]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    sotienkytruoc_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][4]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    sotienkynay_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][5]),font=("Helvetica",12,"bold"),border=1,relief="solid")
                    tiennuoc_label=tk.Label(self.frame, text="{}".format(self.list_KhachHang[kh][6]),font=("Helvetica",12,"bold"),border=1,relief="solid")

                    name_label.place(relwidth=.18,height=100,relx=0,y=vitri_y)
                    diachi_label.place(relwidth=.18,height=100,relx=.18,y=vitri_y)
                    sdt_label.place(relwidth=.14,height=100,relx=.36,y=vitri_y)
                    ngay_label.place(relwidth=.12,height=100,relx=.50,y=vitri_y)
                    sotienkytruoc_label.place(relwidth=.13,height=100,relx=.62,y=vitri_y)
                    sotienkynay_label.place(relwidth=.13,height=100,relx=.75,y=vitri_y)
                    tiennuoc_label.place(relwidth=.12,height=100,relx=.88,y=vitri_y)
                    vitri_y=vitri_y+100

                # lệnh bắt đầu cuộn
                self.frame.update_idletasks()
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        else:
            if self.show_find==True:
                pass
            else:
                self.big_frame = tk.Frame(self.Frame_info,border=1.3,relief="solid")
                self.big_frame.destroy()
                self.canvas.destroy()
                self.scrollbar.pack_forget()

    def placeTab2(self):
        self.update()
        self.labTab2.place(relwidth=1,relheight=.15,x=0,y=0)
        self.findInfo.place(relx=.05,rely=.15,relwidth=.3,relheight=.04)
        self.butFindInfo.place(relx=.35,rely=.14,relheight=.06)
        self.Frame_info.place(relx=0.04,rely=.35,relwidth=.945,relheight=0.59)
        # self.Frame_info.pack(fill="both",expand=True)
        self.Frame_info_label.place(relwidth=.92,relheight=0.10,relx=0.04,rely=.25)
        self.big_frame.place(relx=0,rely=0,relwidth=.975,relheight=1)
        self.Frame_info_name.place(relwidth=.18,relheight=1,relx=.0,rely=.0)
        self.Frame_info_address.place(relwidth=.18,relheight=1,relx=.18,rely=.0)
        self.Frame_info_SDT.place(relwidth=.14,relheight=1,relx=.36,rely=.0)
        self.Frame_info_date.place(relwidth=.12,relheight=1,relx=.50,rely=.0)
        self.Frame_info_NbWater_before.place(relwidth=.13,relheight=1,relx=.62,rely=.0)
        self.Frame_info_NbWater_now.place(relwidth=.13,relheight=1,relx=.75,rely=.0)
        self.Frame_info_Price_Water.place(relwidth=.12,relheight=1,relx=.88,rely=.0)
        self.but_del_user.place(relwidth=.10,relheight=.06,relx=.715,rely=.14)
        self.but_restore_user.place(relwidth=.10,relheight=.06,relx=.61,rely=.14)
        self.but_del_history_user.place(relwidth=.14,relx=.82,rely=.14,relheight=.06)
        self.but_update_user.place(relwidth=.12,relx=.485,rely=.14,relheight=.06)

    def validate_entry(self,new_value):
        if new_value.isdigit() or new_value == "":
            return True
        else:
            return False

    def __init__(self,master):
        super().__init__(master)

        self.labTab2=Label(self, text="Tra cứu thông tin ", font=("Helvetica Bold",30))


        self.findInfo=Entry(self,text ="Tra cứu thông tin",font=("Helvetica",20))
        self.vcmd = (master.register(self.validate_entry), '%P')
        self.findInfo.configure(validate="key",validatecommand=self.vcmd, invalidcommand=master.bell)
        self.findInfo.focus()
        self.butFindInfo=Button(self, text="Tìm kiếm", font=("Helvetica",15,'bold'),border=1.5,relief="solid",background='#FFCCFF',foreground='white',command=self.findUser)
        self.but_del_user=Button(self, text="Xóa", font=("Helvetica",15,'bold'),border=1.5,relief="solid",background='#AA0000',foreground='white',command=self.find_del_user)
        self.but_restore_user=Button(self, text="Khôi Phục", font=("Helvetica",15,'bold'),border=1.5,relief="solid",background='#6666FF',foreground='white',command=self.find_restore_user)
        self.but_update_user=Button(self, text="Cập Nhật", font=("Helvetica",15,'bold'),border=1.5,relief="solid",background='#009999',foreground='white',command=self.find_fix)
        self.but_del_history_user=Button(self, text="Xóa Lịch Sử", font=("Helvetica",15,'bold'),border=1.5,relief="solid",background='#770000',foreground='white',command=self.find_del_history)
        self.Frame_info=Label(self)

        self.big_frame = tk.Frame(self.Frame_info,border=1.3,relief="solid")
        # biến xác định xem đã tìm lần đầu chưa
        self.show_find=True
        self.Frame_info_label=Label(self,border=0.001)
        self.Frame_info_name=Label(self.Frame_info_label,text="Họ và Tên",font=("Helvetica",12,"bold"),border=1,relief="solid")
        self.Frame_info_address=Label(self.Frame_info_label,text="Địa Chỉ",font=("Helvetica",12,"bold"),border=1,relief="solid")
        self.Frame_info_SDT=Label(self.Frame_info_label,text="Số Điện\nThoại",font=("Helvetica",12,"bold"),border=1,relief="solid")
        self.Frame_info_date=Label(self.Frame_info_label,text="Ngày",font=("Helvetica",12,"bold"),border=1,relief="solid")
        self.Frame_info_NbWater_before=Label(self.Frame_info_label,text="Số Nước\nKỳ Trước",font=("Helvetica",12,"bold"),border=1,relief="solid")
        self.Frame_info_NbWater_now=Label(self.Frame_info_label,text="Số Nước\nKỳ Này",font=("Helvetica",12,"bold"),border=1,relief="solid")
        self.Frame_info_Price_Water=Label(self.Frame_info_label,text="Tiền Nước",font=("Helvetica",12,"bold"),border=1,relief="solid")
        master.bind("<Configure>",self.placeTab2())

class Tab3 (Frame):
    def count(self):
        count=[]
        slUser=cursor.execute("select SoDienThoai from QLKH_TN ").fetchall()
        for i in range (len(slUser)):
            count.append(str(slUser[i].SoDienThoai))
        count=list(set(count))
        self.countUser=Label(self,text="Hệ thống có: 0 người", font=("Helvetica Bold",20),borderwidth=2, relief="ridge",background='#333333',foreground='white')
        self.countUser.config(text="Hệ thống có: {} người".format(len(count)),font=("Helvetica Bold",20))
    def countUs(self):
        count=[]
        slUser1=cursor.execute("select SoDienThoai from QLKH_TN where TrangThai=1 ").fetchall()
        for i in range (len(slUser1)):
            count.append(str(slUser1[i].SoDienThoai))
        count=list(set(count))
        self.countUser1=Label(self,text="Còn hoạt động: 0 người", font=("Helvetica Bold",20),borderwidth=2, relief="ridge",background='#333333',foreground='white')
        self.countUser1.config(text="Còn hoạt động: {} người".format(len(count)),font=("Helvetica Bold",20))
    def avg(self):

        self.now_date = datetime.datetime.now()
        self.formatt_date = self.now_date.strftime("%m")

        SoNuocThangTruoc=cursor.execute("select SoNuocThangTruoc from QLKH_TN where Month(Ngay)= ?",self.formatt_date).fetchall()

        SoNuocThangNay=cursor.execute("select SoNuocThangNay from QLKH_TN where Month(Ngay)= ?",self.formatt_date).fetchall()


        soNuocSuDung=[]
        for x in range(len(SoNuocThangTruoc)):
            soNuocSuDung.append(int(SoNuocThangNay[x][0])-int(SoNuocThangTruoc[x][0]))


        suma=sum(soNuocSuDung)
        leng=len(soNuocSuDung)
        self.avgm3=Label(self,text="Trung bình tháng 0: 0 m3", font=("Helvetica Bold",20),borderwidth=2, relief="ridge",background='#333333',foreground='white')
        #ràng buộc
        if leng==0:
            self.avgm3.config(text="Trung bình tháng {}: 0 m3".format(self.formatt_date),font=("Helvetica Bold",20))
        else:
            self.avgm3.config(text="Trung bình tháng {}: {} m3".format(self.formatt_date,round(suma/leng)),font=("Helvetica Bold",20))


    def placeTab3(self):
        self.update()

        self.labTab3.place(relwidth=1,relheight=.15,x=0,y=0)
        self.canvas.get_tk_widget().place(relx=.05,rely=.17,relheight=.6,relwidth=.6)
        self.countUser.place(relx=.68,rely=.2)
        self.countUser1.place(relx=.68,rely=.35)
        self.avgm3.place(relx=.68,rely=.5)


    def __init__(self,master):
        super().__init__(master)
        SoNuocThangTruoc=cursor.execute("select SoNuocThangTruoc,Ngay from QLKH_TN ").fetchall()
        SoNuocThangNay=cursor.execute("select SoNuocThangNay,Ngay from QLKH_TN ").fetchall()
        thang=[1,2,3,4,5,6,7,8,9,10,11,12]
        soNuocSuDung=[]
        for x in range(len(SoNuocThangTruoc)):
            soNuocSuDung.append((int(SoNuocThangNay[x][0])-int(SoNuocThangTruoc[x][0]),SoNuocThangNay[x][1]))
        s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12=0,0,0,0,0,0,0,0,0,0,0,0
        for x in range(len(soNuocSuDung)):
            month=int(soNuocSuDung[x][1].strftime("%m"))
            so=int(soNuocSuDung[x][0])
            if month==1:
                s1=s1+so
            elif month==2:
                s2=s2+so
            elif month==3:
                s3=s3+so
            elif month==4:
                s4=s4+so
            elif month==5:
                s5=s5+so
            elif month==6:
                s6=s6+so
            elif month==7:
                s7=s7+so
            elif month==8:
                s8=s8+so
            elif month==9:
                s9=s9+so
            elif month==10:
                s10=s10+so
            elif month==11:
                s11=s11+so
            elif month==12:
                s12=s12+so
            #print(data)
        data=[s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12]
        self.labTab3=Label(self, text="Thống kê ", font=("Helvetica Bold",30))
        self.fig,self.ax=plt.subplots()
        self.ax.bar(thang,data)
        self.ax.set_xlabel('Tháng',loc=RIGHT)
        self.ax.set_ylabel('m3',loc="top")
        self.ax.set_title('Biểu đồ nước sử dụng trong từng tháng của năm 2023')
        self.ax.legend(['Số nước sử dụng'])
        self.canvas=FigureCanvasTkAgg(self.fig,master=self)
        self.canvas.draw()
        self.count()
        self.countUs()
        self.avg()
        master.bind("<Configure>",self.placeTab3())


class Tabs(Frame):
    def __init__(self,master):
        super().__init__(master)
        self.master=master
        self.s = ttk.Style()
        self.s.configure('TNotebook.Tab', font=('Helvetica Bold','20'),foreground='#6666FF')
        self.tabControl=ttk.Notebook(self.master)
        self.tab1=Tab1(self.tabControl)
        self.tab2=Tab2(self.tabControl)
        self.tab3=Tab3(self.tabControl)
        self.tabControl.add(self.tab1, text='Thông tin người dùng')
        self.tabControl.add(self.tab2, text='Tra cứu thông tin')
        self.tabControl.add(self.tab3, text='Thống kê')
        self.tabControl.pack(expand=True,fill="both",padx=5, pady=5)


if __name__ == '__main__':
    window= Tk()
    screenWidth=window.winfo_screenwidth()
    screenHeight=window.winfo_screenheight()
    window.geometry('1024x768+%d+%d' % (screenWidth/2 -512, screenHeight/2 -384))
    window.title('Ứng dụng quản lý tính tiền nước hàng tháng')
    Tabs(window)
    window.mainloop()



