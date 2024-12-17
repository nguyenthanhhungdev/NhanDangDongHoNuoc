def image_process(self, img):
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
    img = cv2.resize(imgread, dsize=(0, 0), fx=2, fy=2.215)
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
        (npaClassifications.size, 1)
    )  # reshape numpy array to 1d, necessary to pass to call to train
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
    count = 0
    # vẽ đường viền chứa khung số
    ###### Draw contour and filter out the license plate  #############
    contours, hierarchy = cv2.findContours(
        dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # Lấy 10 contours có diện tích lớn nhất
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    # biến lưu giá trị đường viền
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tính chu vi
        approx = cv2.approxPolyDP(
            c, 0.06 * peri, True
        )  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        ratio = w / h
        if (
            len(approx) == 4
            and (w - h) > 150
            and w > 30
            and h > 20
            and 20 < x
            and y > 20
        ):  # and <450 and 200<w<600 and 10<h<300 and 20<y<700) điều kiện để lọc đường viền chỉ còn lại khung số
            screenCnt.append(approx)
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            count += 1

    # count=0 có nghĩa là không tìm thấy khung số phải cắt tay
    if count == 0:

        detected = 0
        height, width = img.shape[:2]
        # Tính toán giá trị tọa độ cần thiết để cắt ảnh
        left = int(width * 0.8 / 5)
        right = int(width * 4.5 / 5)
        bottom = int(height * 2.5 / 5)
        top = int(height * 1.2 / 5)
        # Cắt ảnh theo tỷ lệ đã tính toán được
        cropped_img = img[top:bottom, left:right]
        # cv2.imshow("cropped_img",cropped_img)
        # sử lý ảnh#######################################################
        gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
        imgBlurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        imgThresh = cv2.adaptiveThreshold(
            imgBlurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        def noise_removal(image):
            kernel = np.ones((1, 1), np.uint8)
            image = cv2.dilate(
                image, kernel, iterations=1
            )  # làm đậm các vùng sáng, loại bỏ điểm nhiễu nhỏ
            kernel = np.ones((1, 1), np.uint8)
            image = cv2.erode(
                image, kernel, iterations=1
            )  # làm nhạt các vùng sáng, loại bỏ điểm nhiễu nhỏ còn sót lại
            image = cv2.morphologyEx(
                image, cv2.MORPH_CLOSE, kernel
            )  # Phép đóng là sự kết hợp của giãn nở và xói mòn. Nó giúp lấp đầy các lỗ nhỏ trong các vùng sáng của ảnh.
            image = cv2.medianBlur(
                image, 3
            )  # làm mờ ảnh bằng cách sử dụng bộ lọc trung vị. Đây là một kỹ thuật xử lý ảnh phổ biến để giảm nhiễu trong ảnh.
            return image

        no_noise = noise_removal(imgThresh)

        def thin_font(image):
            import numpy as np

            image = cv2.bitwise_not(image)  # Đảo ngược màu hình ảnh
            kernel = np.ones((2, 2), np.uint8)
            image = cv2.erode(
                image, kernel, iterations=1
            )  # Quá trình erode sẽ làm giảm kích thước của các vùng màu trắng trong hình ảnh.
            image = cv2.bitwise_not(image)  # Đảo ngược màu hình ảnh
            return image

        eroded_image = thin_font(no_noise)

        roi = cropped_img
        roi = cv2.resize(roi, (0, 0), fx=2, fy=2)  # tăng kích thước hình ảnh lên 2 lần
        imgThresh = cv2.resize(
            imgThresh, (0, 0), fx=2, fy=2
        )  # tăng kích thước hình ảnh binary lên 2 lần

        #################### Prepocessing and Character segmentation ####################
        kerel1 = np.ones((1, 1), np.uint8)
        img_dilate = cv2.dilate(imgThresh, kerel1, iterations=1)
        kerel1 = np.ones((1, 1), np.uint8)
        img_erode = cv2.erode(img_dilate, kerel1, iterations=1)
        kerel3 = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 1)
        )  # tạo kernel hình chữ nhật 1x1
        thre_mor = cv2.morphologyEx(img_erode, cv2.MORPH_DILATE, kerel3)
        img_blur = cv2.medianBlur(thre_mor, 3)  # mờ ảnh, giảm nhiễu
        canny = cv2.Canny(
            img_blur, 100, 255
        )  # thực hiện phát hiện biên cạnh (edge detection) trên ảnh img_blur với ngưỡng dưới là 100 và ngưỡng trên là 255.
        # vẽ đường viền của từng số
        cont, hier = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont = sorted(cont, key=cv2.contourArea, reverse=True)[
            :20
        ]  # Lấy 10 contours có diện tích lớn nhất
        cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)
        ##################### Filter out characters #################
        char_x_ind = {}  # lưu vị trí của từng số
        char_x = []  # Lưu giá trị x của từng số
        height, width, _ = roi.shape
        roiarea = height * width
        # vòng lặp chạy đường viền từng số
        for ind, cnt in enumerate(cont):
            (x, y, w, h) = cv2.boundingRect(cont[ind])
            ratiochar = w / h
            char_area = w * h

            if (
                (0.0039 * roiarea < char_area < 0.008 * roiarea)
                and w < h
                and (0.25 < ratiochar < 1.5)
            ):  # and 50<h<300 and 30<w<150
                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                    x = x + 1
                char_x.append(x)
                char_x_ind[x] = ind
        ############ Character recognition ##########################
        char_x = sorted(char_x)
        strFinalString = ""
        number_water = ""
        array_number_water = []
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
            imgROI = thre_mor[y : y + h, x : x + w]  # Crop the characters

            imgROIResized = cv2.resize(
                imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)
            )  # resize image

            """ 

              Chuyển ma trận 2D thành vector 1D
Đổi kiểu dữ liệu sang float32
Cần thực hiện để phù hợp với định dạng đầu vào của model KNN

                  """
            npaROIResized = imgROIResized.reshape(
                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)
            )

            """ 


            Sử dụng model KNN để dự đoán ký tự
Chuyển kết quả số thành ký tự ASCII
Ghép các ký tự vào chuỗi kết quả
Cần thực hiện để hoàn tất quá trình nhận dạng và lưu kết quả


                  """
            npaROIResized = np.float32(npaROIResized)
            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=3)
            strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
            number_water = number_water + strCurrentChar

        number_water = int(number_water)
        number_water = str(number_water)
        if len(number_water) >= 4:
            number_water = number_water[:-2]
        if number_water != "":
            self.displayNumWater.delete(0, END)
            self.displayNumWater.insert(0, int(number_water))
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
            (x1, y1) = screenCnt[0, 0]  # 3   4
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
            angle = math.atan(doi / ke) * (
                180.0 / math.pi
            )  # biến lưu góc nghiên của biển số
            ####################################

            ########## Crop out the license plate and align it to the right angle ################

            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(
                mask,
                [screenCnt],
                0,
                255,
                -1,
            )
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
            imgThresh = cv2.warpAffine(
                imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx)
            )
            # tăng kích thước khung đẫ cắt lên 3 lần
            roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
            imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

            ####################################

            #################### Prepocessing and Character segmentation ####################
            kerel1 = np.ones((1, 1), np.uint8)
            img_dilate = cv2.dilate(imgThresh, kerel1, iterations=1)
            kerel1 = np.ones((1, 1), np.uint8)
            img_erode = cv2.erode(img_dilate, kerel1, iterations=1)
            kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            thre_mor = cv2.morphologyEx(img_erode, cv2.MORPH_DILATE, kerel3)
            img_blur = cv2.medianBlur(thre_mor, 1)
            # tìm viền từng số
            cont, hier = cv2.findContours(
                img_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                roi, cont, -1, (100, 255, 255), 2
            )  # Vẽ contour các kí tự trong biển số
            # cv2.imshow("imgThresh",roi)
            ##################### Filter out characters #################
            char_x_ind = {}
            char_x = []
            height, width, _ = roi.shape
            roiarea = height * width
            for ind, cnt in enumerate(cont):
                (x, y, w, h) = cv2.boundingRect(cont[ind])
                ratiochar = w / h
                perheight = h / height

                char_area = w * h
                if (
                    (Min_char * roiarea < char_area < Max_char * roiarea)
                    and (0.26 < ratiochar < 1.5)
                    and w < h
                    and 0.55 > perheight > 0.3
                ):  # and perheight>0.5 and 70<h<260 and 30<w<150
                    if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                        x = x + 1
                    char_x.append(x)
                    char_x_ind[x] = ind

            ############ Character recognition ##########################
            char_x = sorted(char_x)
            strFinalString = ""
            number_water = ""
            array_number_water = []
            for i in char_x:
                (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                imgROI = thre_mor[y : y + h, x : x + w]  # Crop the characters

                imgROIResized = cv2.resize(
                    imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)
                )  # resize image
                npaROIResized = imgROIResized.reshape(
                    (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)
                )

                npaROIResized = np.float32(npaROIResized)
                _, npaResults, neigh_resp, dists = kNearest.findNearest(
                    npaROIResized, k=3
                )  # call KNN function find_nearest;
                strCurrentChar = str(chr(int(npaResults[0][0])))  # ASCII of characters
                cv2.putText(
                    roi,
                    strCurrentChar,
                    (x, y + 50),
                    cv2.FONT_HERSHEY_DUPLEX,
                    2,
                    (255, 255, 0),
                    3,
                )
                # cv2.imshow("imgThresh",roi)
                number_water = number_water + strCurrentChar
            # n là số khung chứa số
            n = n + 1
            number_water = int(number_water)
            number_water = str(number_water)
            if len(number_water) > 4:
                number_water = number_water[:-2]
            if number_water != "":
                self.displayNumWater.delete(0, END)
                self.displayNumWater.insert(0, int(number_water))
                break
            roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.waitKey(0)
