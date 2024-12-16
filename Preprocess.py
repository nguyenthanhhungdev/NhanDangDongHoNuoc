# Preprocess.py

import cv2
import numpy as np
import math

# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5) #kích cỡ càng to thì càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19 
ADAPTIVE_THRESH_WEIGHT = 9  

###################################################################################################
def preprocess(imgOriginal):

    # imgGrayscale = extractValue(imgOriginal)
    imgGray = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY) #nên dùng hệ màu HSV
    # Trả về giá trị cường độ sáng ==> ảnh gray
    imgMaxContrastGray = maximizeContrast(imgGray) #để làm nổi bật biển số hơn, dễ tách khỏi nền
    #cv2.imwrite("imgGrayscalePlusTopHatMinusBlackHat.jpg",imgMaxContrastGrayscale)
    height, width = imgGray.shape

    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGray, GAUSSIAN_SMOOTH_FILTER_SIZE, 1)
    #cv2.imwrite("gauss.jpg",imgBlurred)
    #Làm mịn ảnh bằng bộ lọc Gauss 5x5, sigma = 0

    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    # cv2.imshow("imgThresh",imgThresh)
    #Tạo ảnh nhị phân
    return imgGray, imgThresh
#Trả về ảnh xám và ảnh nhị phân
# end function

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    
    #màu sắc, độ bão hòa, giá trị cường độ sáng
    #Không chọn màu RBG vì vd ảnh màu đỏ sẽ còn lẫn các màu khác nữa nên khó xđ ra "một màu" 
    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgGray):
    #Làm cho độ tương phản lớn nhất 
    height, width = imgGray.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(imgGray, cv2.MORPH_TOPHAT, structuringElement, iterations = 10) #nổi bật chi tiết sáng trong nền tối
    #cv2.imwrite("tophat.jpg",imgTopHat)
    imgBlackHat = cv2.morphologyEx(imgGray, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10) #Nổi bật chi tiết tối trong nền sáng
    #cv2.imwrite("blackhat.jpg",imgBlackHat)
    imgGrayPlusTopHat = cv2.add(imgGray, imgTopHat) 
    imgGrayPlusTopHatMinusBlackHat = cv2.subtract(imgGrayPlusTopHat, imgBlackHat)

    #cv2.imshow("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)
    #Kết quả cuối là ảnh đã tăng độ tương phản 
    return imgGrayPlusTopHatMinusBlackHat
# end function










