import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import re
import pandas as pd
import seaborn as sns
import xlwt
import imageio
from scipy.spatial import distance as dist
from imutils import perspective
#------------------------------------分割线-----------------------------------------
#中值滤波去除椒盐噪声
def medium_filter(im, x, y, step):
    sum_s = []
    for k in range(-int(step / 2), int(step / 2) + 1):
        for m in range(-int(step / 2), int(step / 2) + 1):
            sum_s.append(im[x + k][y + m])
    sum_s.sort()
    return sum_s[(int(step * step / 2) + 1)]

#求最大连通域的中心点坐标
def centroid(max_contour):
    moment = cv.moments(max_contour)
    if moment['m00'] != 0:
        cx = int(moment['m10'] / moment['m00'])
        cy = int(moment['m01'] / moment['m00'])
        return cx, cy
    else:
        return None

#获取目标图片的信息：边缘、平均灰度值
def Get_targe_value(path):
    #读取图片
    image_filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]
    # 统计图片数量
    count = 0
    #生成三个列表接收全部图像的平均灰度值、边缘值、拍摄时间
    list_name = []
    list_grayscale = []
    list_edge = []
    #生成excel写入数据
    book = xlwt.Workbook() #创建一个Excel
    sheet1 = book.add_sheet('data') #在其中创建一个名为data的sheet
    sheet1.write(0, 0, '图片名称')  # 往sheet里第一行第一列写一个数据
    sheet1.write(0, 1, '拍摄时间')
    sheet1.write(0, 2, '平均灰度值')
    sheet1.write(0, 3, '预测含水率')
    sheet1.write(0, 4, '水平向左运移距离')
    sheet1.write(0, 5, '水平向右运移距离')
    sheet1.write(0, 6, '竖直向上运移距离')
    sheet1.write(0, 7, '竖直向下运移距离')
    sheet1.write(0, 8, '最大运移距离')
    sheet1.write(0, 9, '最小运移距离')
    sheet1.write(0, 10, '湿润锋轮廓长度')
    sheet1.write(0, 11, '湿润面面积')
    for photo_path in image_filenames:
        count = count + 1

        photo_name = Get_filename(photo_path)
        print('Reading photo_MositFront_{}...'.format(photo_name))
        photo_name_int = int(photo_name)
        list_name.append(photo_name_int)

        img_bgr = cv.imread(photo_path)
        b, g, r = cv.split(img_bgr)
        img_rgb = cv.merge([r, g, b])# 变换为rgb    或img[:,:,[2,1,0]]转换bgr通道

        #转换灰度图
        gray_img = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY) #gray_img = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
        dst = 255 - gray_img#二值反转
        # #高斯滤波
        # dst_GaussianBlur = cv.GaussianBlur(dst, (5,5), 0)
        #局部阈值大津算法
        # dst = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,25, 10)
        #全局大津算法
        res, otsu_img = cv.threshold(dst, 0, 255, cv.THRESH_OTSU)

        #形态学操作 原因：腐蚀使大津法处理后图片的连通白色区域断连，便于后续计算最大面积时，使目标区域与最大面积重合，但不能过度腐蚀，这样会使目标区域边缘变得粗糙
        kernel = np.ones((5, 5), np.uint8)#创建核结构
        erosion1 = cv.erode(otsu_img, kernel, 10000)  # 腐蚀
        erosion2 = cv.erode(erosion1, kernel, 10000)  # 腐蚀

        erosion_img = cv.erode(erosion1, kernel, 10000)  # 腐蚀

        #删除面积小的轮廓，保留最大的，重复3次
        dst3 = np.zeros(img_rgb.shape, np.uint8)
        for i in range(3):
            dst3, contours, hierarchy = cv.findContours(erosion2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓检测函数 会影响输入的原图
            # print("消除前contours的个数：", len(contours))
            cv.drawContours(dst3, contours, -1, (255, 255, 0), 1)  # 绘制轮廓
            # 找到最大区域并填充
            area = []
            for i in range(len(contours)):
                area.append(cv.contourArea(contours[i]))
            max_idx = np.argmax(area)
            # print("area的个数：",len(area))
            # print("max_idx：",max_idx)
            for i in range(len(contours)):
                cv.fillConvexPoly(dst3, contours[i], 0)
            cv.fillConvexPoly(dst3, contours[max_idx], 255)

        #形态学操作 平滑边缘
        # kernel = np.ones((5, 5), np.uint8)#创建核结构
        mask = dst3
        # for i in range(15):
        #     mask = cv.dilate(mask, kernel, 10000)#膨胀
        # for i in range(20):
        #     mask = cv.erode(mask, kernel, 10000)  #腐蚀
        edge_img_Blur = cv.blur(mask, (100, 100))  # 均值滤波
        ret, mask = cv.threshold(edge_img_Blur, 100, 255, cv.THRESH_BINARY)

        #获取掩膜的轮廓
        mask, contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓检测函数

        #得到标尺PixelsPerMetric
        scale_photo_path = path + r'\measuring_scale\coin.jpg'
        PixelsPerMetric = Get_PixelsPerMetric(scale_photo_path)

        #轮廓面积检测函数
        area = []
        for i in range(len(contours)):
            area.append(cv.contourArea(contours[i]))
        max_idx = np.argmax(area)#返回最大值中的索引值
        Pixels_square = (1 / PixelsPerMetric) ** 2
        area_measure = '%.1f' % (area[max_idx] * Pixels_square)
        print('面积的测量值:{}cm^2'.format(area_measure))
        # print("max_idx", max_idx)
        # print("消除后contours的个数：", len(contours))


        #求最大连通域的中心坐标
        cnt_centroid = centroid(contours[max_idx])
        cv.circle(contours[max_idx], cnt_centroid, 5, [255, 255, 255], 1)
        # print("Centroid : " + str(cnt_centroid))

        #轮廓长度检测函数
        a = []
        for i in range(0, len(contours)):
            a.append(len(contours[i]))
        max_value = 0
        for i in range(0, len(a)):
            if max_value < a[i]:
                max_value = a[i]
        index = a.index(max_value)
        length_maesure = '%.1f' % (max_value / PixelsPerMetric)
        print('轮廓的测量值:{}cm'.format(length_maesure))

        #目标提取+轮廓+平均灰度图
        image = cv.add(img_rgb, np.zeros(np.shape(img_bgr), dtype=np.uint8), mask=mask)

        aver_histogram_rgb, img_histogram, b_hist, g_hist, r_hist = Read2_value(image)
        # print('aver_histogram_rgb', aver_histogram_rgb)
        aver_histogram_rgb = float(aver_histogram_rgb)
        list_grayscale.append(aver_histogram_rgb)
        moisture = Predict(w, aver_histogram_rgb)
        # print('moisture', moisture)
        # print("rgb通道计算图像平均灰度值= ", aver_histogram_rgb)

        # 显示1
        cv.namedWindow('otsu_img', 0)
        cv.resizeWindow('otsu_img', 1200, 800)  # 自己设定窗口图片的大小
        cv.imshow("otsu_img", otsu_img)
        cv.namedWindow('erosion_img', 0)
        cv.resizeWindow('erosion_img', 1200, 800)  # 自己设定窗口图片的大小
        cv.imshow("erosion_img", erosion_img)
        cv.namedWindow('dst3', 0)
        cv.resizeWindow('dst3', 1200, 800)  # 自己设定窗口图片的大小
        cv.imshow("dst3", dst3)
        cv.namedWindow('edge_img_Blur', 0)
        cv.resizeWindow('edge_img_Blur', 1200, 800)  # 自己设定窗口图片的大小
        cv.imshow("edge_img_Blur", edge_img_Blur)
        cv.namedWindow('mask', 0)
        cv.resizeWindow('mask', 1200, 800)  # 自己设定窗口图片的大小
        cv.imshow("mask", mask)

        b, g, r = cv.split(image)
        image = cv.merge([r, g, b])
        cv.namedWindow('image', 0)
        cv.resizeWindow('image', 1200, 800)  # 自己设定窗口图片的大小
        cv.imshow("image", image)

        cv.waitKey(0)
        cv.destroyAllWindows()

        #在纯黑图片上绘制轮廓：湿润锋、湿润图绘制
        rgb_space = [(120, 2, 124), (0, 47, 129), (23, 93, 163), (35, 116, 207), (61,159,234), (124,206,246), (147,226,243), (190,248,255), (239,255,252)]
        empty_img = np.zeros(mask.shape, np.uint8)
        edge_img = cv.merge((empty_img, empty_img, empty_img))
        cv.drawContours(edge_img, contours, -1, (255, 255, 255), 5)  # 绘制轮廓
        for i in range(len(contours)):
            cv.fillPoly(edge_img, [contours[i]], rgb_space[2])

        # Location = (contours[0][0][0][0], contours[0][0][0][1])

        #找点
        x_axes = []
        y_axes = []
        for i in range(len(contours[index])):
            x_axes.append(contours[index][i][0][0])
            y_axes.append(contours[index][i][0][1])

        # 找最远和最近的点
        farthest_point, nearest_point = 0, 0
        distance_list = []
        for i in range(len(x_axes)):
            # distance = ((cnt_centroid[0] - x_axes[i]) ** 2 + (cnt_centroid[1] - y_axes[i]) ** 2)
            distance = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (x_axes[i], y_axes[i]))
            distance_list.append(distance)
        for index_d, value_d in enumerate(distance_list):
            if value_d == max(distance_list): farthest_point = (x_axes[index_d], y_axes[index_d])
            if value_d == min(distance_list): nearest_point = (x_axes[index_d], y_axes[index_d])
        # print(farthest_point, nearest_point)
        d_max = max(distance_list) / PixelsPerMetric
        d_min = min(distance_list) / PixelsPerMetric

        # 找最上和最下的点
        # min_y = min(y_axes)
        top_point = 0
        bottom_point = 0
        for index_d, value_d in enumerate(y_axes):
            if value_d == max(y_axes): top_point = (x_axes[index_d], y_axes[index_d])
            if value_d == min(y_axes): bottom_point = (x_axes[index_d], y_axes[index_d])
        # print(top_point, bottom_point)
        # d_top = (cnt_centroid[0] - top_point[0]) ** 2 + (cnt_centroid[1] - top_point[1]) ** 2
        # d_bottom = (cnt_centroid[0] - bottom_point[0]) ** 2 + (cnt_centroid[1] - bottom_point[1]) ** 2
        d_top = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (top_point[0], top_point[1]))
        d_top = d_top / PixelsPerMetric
        d_bottom = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (bottom_point[0], bottom_point[1]))
        d_bottom = d_bottom / PixelsPerMetric

        # 找最左和最右的点
        leftmost_point, rightmost_point= 0, 0
        for index_d, value_d in enumerate(x_axes):
            if value_d == max(x_axes): leftmost_point = (x_axes[index_d], y_axes[index_d])
            if value_d == min(x_axes): rightmost_point = (x_axes[index_d], y_axes[index_d])
        # print(leftmost_point, rightmost_point)
        # d_leftmost = (cnt_centroid[0] - leftmost_point[0]) ** 2 + (cnt_centroid[1] - leftmost_point[1]) ** 2
        # d_rigthmost = (cnt_centroid[0] - rightmost_point[0]) ** 2 + (cnt_centroid[1] - rightmost_point[1]) ** 2
        d_leftmost = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (leftmost_point[0], leftmost_point[1]))
        d_leftmost = d_leftmost / PixelsPerMetric
        d_rigthmost = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (rightmost_point[0], rightmost_point[1]))
        d_rigthmost = d_rigthmost / PixelsPerMetric

        # 找竖直向上 竖直向下的两点
        n = 10
        vertical_list = []
        dot_y = []
        for index_x, value_x in enumerate(x_axes):
            if abs(value_x - cnt_centroid[0]) < n:
                vertical = (cnt_centroid[0], y_axes[index_x])
                dot_y.append(y_axes[index_x])
                vertical_list.append(vertical)
        min_up = min(dot_y)
        max_down = max(dot_y)
        up_point = (cnt_centroid[0], min_up)
        down_point = (cnt_centroid[0], max_down)
        # print('up_point', up_point)
        # print('down_point', down_point)
        # d_up = (cnt_centroid[0] - up_point[0]) ** 2 + (cnt_centroid[1] - up_point[1]) ** 2
        # d_down = (cnt_centroid[0] - down_point[0]) ** 2 + (cnt_centroid[1] - down_point[1]) ** 2
        d_up = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (up_point[0], up_point[1]))
        d_up = d_up / PixelsPerMetric
        d_down = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (down_point[0], down_point[1]))
        d_down = d_down / PixelsPerMetric

        # 找水平向左 水平向右两点
        horizon_list = []
        dot_x = []
        for index_y, value_y in enumerate(y_axes):
            if abs(value_y - cnt_centroid[1]) < n:
                horizon = (x_axes[index_y], cnt_centroid[1])
                dot_x.append(x_axes[index_y])
                horizon_list.append(horizon)
        min_left = min(dot_x)
        max_right = max(dot_x)
        left_point = (min_left, cnt_centroid[1])
        right_point = (max_right, cnt_centroid[1])
        # print('left_point', left_point)
        # print('right_point', right_point)
        # d_left = (cnt_centroid[0] - left_point[0]) ** 2 + (cnt_centroid[1] - left_point[1]) ** 2
        # d_right = (cnt_centroid[0] - right_point[0]) ** 2 + (cnt_centroid[1] - right_point[1]) ** 2
        d_left = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (left_point[0], left_point[1]))
        d_left = d_left / PixelsPerMetric
        d_right = dist.euclidean((cnt_centroid[0], cnt_centroid[1]), (right_point[0], right_point[1]))
        d_right = d_right / PixelsPerMetric

        #绘图
        cv.circle(edge_img, cnt_centroid, 10, [255, 255, 255], -1)

        cv.line(edge_img, cnt_centroid, up_point, [255, 255, 255], 5)
        cv.putText(edge_img, '{}cm'.format('%.1f' % d_up), up_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        cv.line(edge_img, cnt_centroid, down_point, [255, 255, 255], 5)
        cv.putText(edge_img, '{}cm'.format('%.1f' % d_down), down_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        cv.line(edge_img, cnt_centroid, left_point, [255, 255, 255], 5)
        cv.putText(edge_img, '{}cm'.format('%.1f' % d_left), left_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        cv.line(edge_img, cnt_centroid, right_point, [255, 255, 255], 5)
        cv.putText(edge_img, '{}cm'.format('%.1f' % d_right), right_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv.LINE_AA)

        cv.line(edge_img, cnt_centroid, farthest_point, [255, 0, 0], 5)
        cv.putText(edge_img, '{}cm'.format('%.1f' % d_max), farthest_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

        cv.line(edge_img, cnt_centroid, nearest_point, [0, 255, 0], 5)
        cv.putText(edge_img, '{}cm'.format('%.1f' % d_min), nearest_point, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)

        cv.circle(edge_img, top_point, 15, [255, 0, 0], 5)
        # cv.putText(edge_img, '{}'.format(d_top), top_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

        cv.circle(edge_img, bottom_point, 15, [255, 0, 0], 5)
        # cv.putText(edge_img, '{}'.format(d_bottom), bottom_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

        cv.circle(edge_img, rightmost_point, 15, [255, 0, 0], 5)
        # cv.putText(edge_img, '{}'.format(d_rigthmost), rightmost_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

        cv.circle(edge_img, leftmost_point, 15, [255, 0, 0], 5)
        # cv.putText(edge_img, '{}'.format(d_leftmost), leftmost_point, cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)

        text = '{}moisture content'.format('%.2f' % float(moisture))
        cv.putText(edge_img, text, cnt_centroid, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv.LINE_AA)

        cv.putText(edge_img, 'Wetted surface all info:', (80, 150), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 2, cv.LINE_AA)

        cv.putText(edge_img, 'migration distance of wetting front:', (100, 250), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(edge_img, 'Maximum: {}cm'.format('%.1f' % d_max), (150, 320), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(edge_img, 'Minimum: {}cm'.format('%.1f' % d_min), (150, 390), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(edge_img, 'straight up: {}cm'.format('%.1f' % d_up), (150, 460), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(edge_img, 'straight down: {}cm'.format('%.1f' % d_down), (150, 530), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(edge_img, 'Horizontal left: {}cm'.format('%.1f' % d_left), (150, 600), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(edge_img, 'Horizontal right: {}cm'.format('%.1f' % d_right), (150, 670), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)

        cv.putText(edge_img, 'Wetting front profile length: {}cm'.format(length_maesure), (100, 770), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(edge_img, 'Wetted surface area: {}cm^2'.format(area_measure), (100, 850), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv.LINE_AA)

        #写excel
        sheet1.write(count, 0, '{}'.format(photo_name))
        # sheet1.write(count, 0, '{}月{}日 {}时{}分'.format())
        sheet1.write(count, 2, '{}'.format('%.3f' % float(aver_histogram_rgb)))
        sheet1.write(count, 3, '{}'.format('%.3f' % float(moisture)))
        sheet1.write(count, 4, '{}'.format('%.3f' % float(d_left)))
        sheet1.write(count, 5, '{}'.format('%.3f' % float(d_right)))
        sheet1.write(count, 6, '{}'.format('%.3f' % float(d_up)))
        sheet1.write(count, 7, '{}'.format('%.3f' % float(d_down)))
        sheet1.write(count, 8, '{}'.format('%.3f' % float(d_max)))
        sheet1.write(count, 9, '{}'.format('%.3f' % float(d_min)))
        sheet1.write(count, 10, '{}'.format('%.3f' % float(length_maesure)))
        sheet1.write(count, 11, '{}'.format('%.3f' % float(area_measure)))

        #保存图片
        b, g, r = cv.split(edge_img)
        edge_img = cv.merge([r, g, b])
        print('Drawing photo_MositFront_{}...'.format(photo_name))
        print()
        picture_path = r'Moist front\{}.png'.format(photo_name)
        save_path = os.path.join(path, picture_path)
        cv.imwrite(save_path, edge_img)

    #保存excel
    file_excel = path + r'\Moist front\info.xls'
    book.save(file_excel) #创建保存文件

    #生成gif
    file_gif = path + r'\Moist front'
    name = r'\Migration.gif'
    create_gif(file_gif, name)

    # 排序
    list_grayscale = np.array(list_grayscale)  # 数据类型转换
    predict_targe_value = Predict(w, list_grayscale)
    key = ["图片名", "平均灰度值", "预测含水率"]
    a = []
    train_dict = []
    for i in range(len(list_name)):
        a.append(list_name[i])
        a.append(list_grayscale[i])
        a.append(predict_targe_value[i])
        dic = dict(zip(key, a))
        train_dict.append(dic)
        a[:] = []
    for i in range(len(train_dict)):
        for j in range(i + 1, len(train_dict)):
            if train_dict[i].get('图片名') > train_dict[j].get('图片名'):
                train_dict[i], train_dict[j] = train_dict[j], train_dict[i]
    print()
    print("3.目标集")
    print("目标图片数量：", count)
    for i in range(len(train_dict)):
        print("目标图片内容：", train_dict[i])
    print()

    return list_grayscale
# return list_name, list_moisture, list_edge

def midpoint(A, B):
    return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

#得到比例系数pixelsPerMetric
def Get_PixelsPerMetric(scale_photo_path):
    pixelsPerMetric = None
    width = 2.5

    img_bgr = cv.imread(scale_photo_path)
    # img_text = cv.imread(r'C:\Users\zhang\Desktop\the moisture content photo\rule\text.jpg')
    gray_img = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)  # gray_img = np.dot(img_rgb[...,:3], [0.299, 0.587, 0.114])
    dst = 255 - gray_img  # 二值反转

    # 全局大津算法
    res, otsu_img = cv.threshold(dst, 0, 255, cv.THRESH_OTSU)
    # canny_img = cv.Canny(dst, 50, 100)

    # dst3 = np.zeros(img_bgr.shape, np.uint8)
    dst3, contours, hierarchy = cv.findContours(otsu_img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓检测函数 会影响输入的原图
    # print("消除前contours的个数：", len(contours))
    cv.drawContours(dst3, contours, -1, (255, 255, 0), 1)  # 绘制轮廓
    # 找到最大区域并填充
    area = []
    for i in range(len(contours)):
        area.append(cv.contourArea(contours[i]))
    max_idx = np.argmax(area)
    # print("area的个数：",len(area))
    # print("max_idx：",max_idx)
    for i in range(len(contours)):
        cv.fillConvexPoly(dst3, contours[i], 0)
    cv.fillConvexPoly(dst3, contours[max_idx], 255)

    # cnts = cv.findContours(dst3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # (cnts, _) = cnt.sort_contours(cnts)
    dst2, contours, hierarchy = cv.findContours(dst3.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # 轮廓检测函数 会影响输入的原图

    # print(len(contours))
    # print(contours)
    # np.asarray(cnts)
    # empty_img1 = np.zeros(dst3.shape, np.uint8)
    # edge_img = cv.merge((empty_img1, empty_img1, empty_img1))
    # cv.drawContours(edge_img, contours[0], -1, (255, 255, 255), 5)
    # cv.fillPoly(edge_img, [contours[0]], (255, 255, 255))

    box = cv.minAreaRect(contours[0])
    box = cv.boxPoints(box)
    box = np.array(box, dtype="int")
    # order the points in the contour such that they appear in top-left, top-right, bottom-right, and bottom-left order,
    # then draw the outline of the rotated bounding
    box = perspective.order_points(box)  # 测量距离
    # print('box', box)
    # cv.drawContours(edge_img, [box.astype("int")], -1, (0, 255, 0), 2)

    # # 显示1
    # cv.namedWindow('dst', 0)
    # cv.resizeWindow('dst', 1200, 800)  # 自己设定窗口图片的大小
    # cv.imshow("dst", dst)
    # cv.namedWindow('otsu_img', 0)
    # cv.resizeWindow('otsu_img', 1200, 800)  # 自己设定窗口图片的大小
    # cv.imshow("otsu_img", otsu_img)
    # cv.namedWindow('dst3', 0)
    # cv.resizeWindow('dst3', 1200, 800)  # 自己设定窗口图片的大小
    # cv.imshow("dst3", dst3)
    # cv.namedWindow('edge_img', 0)
    # cv.resizeWindow('edge_img', 1200, 800)  # 自己设定窗口图片的大小
    # cv.imshow("edge_img", edge_img)
    # # cv.namedWindow('img_text', 0)
    # # cv.resizeWindow('img_text', 1200, 800)  # 自己设定窗口图片的大小
    # # cv.imshow("img_text", img_text)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # loop over the original points and draw them
    # for (x, y) in box:
    #     cv.circle(edge_img, (int(x), int(y)), 5, (0, 0, 255), -1)
    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by the midpoint between bottom-left and bottom-right coordinates
    (tl, tr, br, bl) = box
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)

    # compute the midpoint between the top-left and top-right points, followed by the midpoint between the top-righ and bottom-right
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)

    # draw the midpoints on the image
    # x_list = []
    # y_list = []
    # cv.circle(edge_img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    # cv.circle(edge_img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    # cv.circle(edge_img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    # cv.circle(edge_img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
    # x_list.append(tltrX)
    # x_list.append(blbrX)
    # x_list.append(tlblX)
    # x_list.append(trbrX)
    # y_list.append(tltrY)
    # y_list.append(blbrY)
    # y_list.append(tlblY)
    # y_list.append(trbrY)
    # x_min = int(min(x_list))
    # x_max = int(max(x_list))
    # y_min = int(min(y_list))
    # y_max = int(max(y_list))
    # print((x_min, y_min))
    # print((x_max, y_max))

    # draw lines between the midpoints
    # cv.line(edge_img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
    # cv.line(edge_img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    # if the pixels per metric has not been initialized, then compute it as the ratio of pixels to supplied metric (in this case, inches)
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / width #代表1的真实距离对应多少像素距离

    # compute the size of the object
    # dimA = dA / pixelsPerMetric #真实值
    # dimB = dB / pixelsPerMetric
    # draw the object sizes on the image
    # cv.putText(edge_img, "{:.1f}in".format(dimA),
    #             (int(tltrX - 15), int(tltrY - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    # cv.putText(edge_img, "{:.1f}in".format(dimB),
    #             (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # imageROI = edge_img[y_min: y_max, x_min:x_max]
    # img_text[y_min: y_max, x_min:x_max] = imageROI

    return pixelsPerMetric
# return pixelsPerMetric

#读取文件名
def Get_filename(path):
    regex = "([^<>/\\\|:""\*\?]+)\.\w+$"
    file_name = re.findall(regex, path)
    # print(file_name)
    a = file_name[0]
    # moisture_content = float(a)

    # moisture_content = []
    # for n in file_name:
    #     moisture_content.append(float(n))
    # print("图片文件名:", a)
    return a
#return a

#读取对应图片rgb及grayscale信息的函数
def Read_value(photo):
    # print("Read_value")
    #rgb三通道转换
    b, g, r = cv.split(photo)
    img_rgb = cv.merge([r, g, b])  # 变换为rgb

    #获取图片基本信息
    # print("_________________________________________________________信息起始点_________________________________________________________")
    # height = len(img_rgb)
    # width = len(img_rgb[0])
    # print('图片大小：%d*%d' % (width, height))
    # print('图片size：', img_rgb.size)
    # print('图片dtype：', img_rgb.dtype)

    #绘制图片灰度直方图
    cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
    img_histogram = cv.calcHist([photo], [0], None, [256], [0, 256])  # 中括号一定要加
    # img_histogram_list = img_histogram.ravel()
    # sum = 0
    # sum_i = 0
    # for i, v in enumerate(img_histogram):  # enumerate为枚举，返回对应元素及其索引
    #     # print(v, end='')  # end=‘’，将结尾默认换行改为空格
    #     sum_i = sum_i + i * v
    #     sum = sum + v
    #     # if(i+1)%16==0:print("\tsum= {},\tsum_i= {}".format(int(sum), int(sum_i)))#每16个就换行，一个16*16=256个灰度值
    # aver_histogram = sum_i / sum
    # print()
    # print("灰度图像得到的平均灰度值= ", aver_histogram)
    # print()
    # 绘制图片彩色三通道直方图
    histSize = 256
    histRange = (0, histSize)
    b_hist = cv.calcHist([b], [0], None, [histSize], histRange)
    g_hist = cv.calcHist([g], [0], None, [histSize], histRange)
    r_hist = cv.calcHist([r], [0], None, [histSize], histRange)

    #对b通道操作
    sum = 0
    sum_i = 0
    for i, v in enumerate(b_hist):  # enumerate为枚举，返回对应元素及其索引
        # print(v,end='')#end=‘’，将结尾默认换行改为空格
        sum = sum + v
        sum_i = sum_i + i * v
        # if(i+1)%16==0:print("\tsum= {},\tsum_i= {}".format(int(sum), int(sum_i)))#每16个就换行，一个16*16=256个灰度值
    aver1 = sum_i / sum
    # print("b通道的平均灰度值=%s " % aver1)
    # print()

    #对g通道操作
    sum = 0
    sum_i = 0
    for i, v in enumerate(g_hist):  # enumerate为枚举，返回对应元素及其索引
        # print(v,end='')#end=‘’，将结尾默认换行改为空格
        sum = sum + v
        sum_i = sum_i + i * v
        # if(i+1)%16==0:print("\tsum= {},\tsum_i= {}".format(int(sum), int(sum_i)))#每16个就换行，一个16*16=256个灰度值
    aver2 = sum_i / sum
    # print("g通道的平均灰度值=%s " % aver2)
    # print()

    #对r通道操作
    sum = 0
    sum_i = 0
    for i, v in enumerate(r_hist):  # enumerate为枚举，返回对应元素及其索引
        # print(v,end='')#end=‘’，将结尾默认换行改为空格
        sum = sum + v
        sum_i = sum_i + i * v
        # if(i+1)%16==0:print("\tsum= {},\tsum_i= {}".format(int(sum), int(sum_i)))#每16个就换行，一个16*16=256个灰度值
    aver3 = sum_i / sum
    # print("r通道的平均灰度值=%s " % aver3)
    # print()

    aver_histogram_rgb = (aver1 + aver2 + aver3) / 3
    # print("rgb通道计算图像平均灰度值= ", aver_histogram_rgb)
    # print("_________________________________________________________信息结束点_________________________________________________________\n")

    return aver_histogram_rgb, img_histogram, b_hist, g_hist, r_hist
#return aver_histogram_rgb, img_histogram, b_hist, g_hist, r_hist

#读取对应ROI图片info的函数
def Read2_value(photo):
    # print("Read_value")
    #rgb三通道转换
    b, g, r = cv.split(photo)
    img_rgb = cv.merge([r, g, b])  # 变换为rgb
    #绘制图片灰度直方图
    cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
    img_histogram = cv.calcHist([photo], [0], None, [256], [0, 256])  # 中括号一定要加

    histSize = 256
    histRange = (0, histSize)
    b_hist = cv.calcHist([b], [0], None, [histSize], histRange)
    g_hist = cv.calcHist([g], [0], None, [histSize], histRange)
    r_hist = cv.calcHist([r], [0], None, [histSize], histRange)

    #对b通道操作
    sum = 0
    sum_i = 0
    for i, v in enumerate(b_hist):  # enumerate为枚举，返回对应元素及其索引
        if i == 0 : continue
        # print(v,end='')#end=‘’，将结尾默认换行改为空格
        sum = sum + v
        sum_i = sum_i + i * v
        # if(i+1)%16==0:print("\tsum= {},\tsum_i= {}".format(int(sum), int(sum_i)))#每16个就换行，一个16*16=256个灰度值
    aver1 = sum_i / sum
    # print("b通道的平均灰度值=%s " % aver1)
    # print()

    #对g通道操作
    sum = 0
    sum_i = 0
    for i, v in enumerate(g_hist):  # enumerate为枚举，返回对应元素及其索引
        if i == 0 : continue
        # print(v,end='')#end=‘’，将结尾默认换行改为空格
        sum = sum + v
        sum_i = sum_i + i * v
        # if(i+1)%16==0:print("\tsum= {},\tsum_i= {}".format(int(sum), int(sum_i)))#每16个就换行，一个16*16=256个灰度值
    aver2 = sum_i / sum
    # print("g通道的平均灰度值=%s " % aver2)
    # print()

    #对r通道操作
    sum = 0
    sum_i = 0
    for i, v in enumerate(r_hist):  # enumerate为枚举，返回对应元素及其索引
        if i == 0 : continue
        # print(v,end='')#end=‘’，将结尾默认换行改为空格
        sum = sum + v
        sum_i = sum_i + i * v
        # if(i+1)%16==0:print("\tsum= {},\tsum_i= {}".format(int(sum), int(sum_i)))#每16个就换行，一个16*16=256个灰度值
    aver3 = sum_i / sum
    # print("r通道的平均灰度值=%s " % aver3)
    # print()

    aver_histogram_rgb = (aver1 + aver2 + aver3) / 3
    # print("rgb通道计算图像平均灰度值= ", aver_histogram_rgb)
    # print("_________________________________________________________信息结束点_________________________________________________________\n")

    return aver_histogram_rgb, img_histogram, b_hist, g_hist, r_hist
#return aver_histogram_rgb, img_histogram, b_hist, g_hist, r_hist

# 读取训练集或测试集图片信息的函数
def Get_value(path, flag):
    #传递路径参数+批量处理
    filename_list = os.listdir(path)
    # 给文件名加上路径
    file_list = [os.path.join(path, i) for i in filename_list]
    # print(file_list)
    # print()
    # 统计图片数量
    count = 0
    #生成列表接收图像灰度值、三通道灰度值
    # list_y1 = []
    list_y2 = []
    #生成列表接收图像文件名即含水率
    list_x = []
    list_name = []
    for photo in file_list:
        count = count + 1

        img = cv.imread(photo)
        b,g,r = cv.split(img)
        img_rgb = cv.merge([r,g,b])

        # 剪裁图像
        img_croped = img[1000:3000, 2000:4000]
        b,g,r = cv.split(img_croped)
        img_croped = cv.merge([r,g,b])

        print("图片路径：", photo)
        photo_name = Get_filename(photo)
        list_name.append(photo_name)

        a2, img_histogram, b_hist, g_hist, r_hist = Read_value(img_croped)
        # list_y1.append(float(a1))
        list_y2.append(float(a2))
        moisture_content = float(photo_name)
        list_x.append(moisture_content)

        #绘制当前图片直方图
        # Mapping(img_rgb, img_croped, b_hist, g_hist, r_hist, a2)

    list_x = np.array(list_x)#数据类型转换
    list_y2 = np.array(list_y2)
    # 信息输出
    if flag == 1:
        key = ["图片名", "平均灰度值", "含水率"]
        a = []
        train_dict = []

        for i in range(len(list_name)):
            a.append(list_name[i])
            a.append(list_y2[i])
            a.append(list_x[i])
            dic = dict(zip(key, a))
            train_dict.append(dic)
            a[:] = []
        #排序
        for i in range(len(train_dict)):
            for j in range(i + 1, len(train_dict)):
                if train_dict[i].get('含水率') > train_dict[j].get('含水率'):
                    train_dict[i], train_dict[j] = train_dict[j], train_dict[i]

        print()
        print("1.训练集")
        print("训练图片数量：", count)
        for i in range(len(train_dict)):
            print("训练图片内容：", train_dict[i])

    elif flag == 2:
        # 根据训练模型得到相应含水率，并分析误差
        predict_test_value = Predict(w, list_y2)
        key = ["图片名", "平均灰度值", "真实含水率", "预测含水率", "误差值"]
        a = []
        train_dict = []
        for i in range(len(list_x)):
            a.append(list_name[i])
            a.append(list_y2[i])
            a.append(list_x[i])
            a.append(predict_test_value[i])
            c = predict_test_value[i] - list_x[i]
            a.append(c)
            dic = dict(zip(key, a))
            train_dict.append(dic)
            a[:] = []
        # 排序
        for i in range(len(train_dict)):
            for j in range(i + 1, len(train_dict)):
                if train_dict[i].get('真实含水率') > train_dict[j].get('真实含水率'):
                    train_dict[i], train_dict[j] = train_dict[j], train_dict[i]
        print()
        print("2.测试集")
        print("测试图片数量：", count)
        for i in range(len(train_dict)):
            print("测试图片内容：", train_dict[i])
    print()


    return list_x, list_y2, count
# return  list_x, list_y2

#绘制对应图片直方图函数
def Mapping(img_rgb, img_croped, b_hist, g_hist, r_hist, img_histogram):
    #画图
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=100)
    axes[0, 0].imshow(img_rgb)#cmap为颜色图谱，默认绘制为RGB(A)颜色空间 plt.cm.gray
    axes[0, 0].set_title("original")
    axes[0, 1].imshow(img_croped)
    axes[0, 1].set_title("Croped")
    axes[1, 0].plot(img_histogram, color='k')
    axes[1, 0].grid()#增加网格
    axes[1, 0].set_title("Gray Histogram")
    axes[1, 1].plot(b_hist, color='b')
    axes[1, 1].plot(g_hist, color='g')
    axes[1, 1].plot(r_hist, color='r')
    axes[1, 1].grid()#增加网格
    axes[1, 1].set_title("rgb Histogram")
    plt.show()
    return None

#绘制部分组件函数
def draw_ticks():
    plt.tick_params(labelsize=15)#设置标签大小
    # 设置横纵坐标范围以及步长
    # plt.xticks(np.linspace(0, 1, 2))
    # plt.yticks(np.linspace(-1, 1, 3))
    #设置纵坐标上下限
    # plt.ylim(-1.5, 1.5)
    #设置横纵坐标标签
    font = {'family':'Times New Roman','size':20}
    plt.xlabel('grayscale_value', font)
    plt.ylabel('moisture_content', font, rotation='270', labelpad=30)

#拟合函数（lamda默认为0，即无正则项）
def regress(M, N, x, x_n, t_n, lamda=0): #0, 10, x：范围, x_10, t_10
    print("-----------------------M=%d, N=%d-------------------------" %(M,N))
    order = np.arange(M+1)#返回一个终点为M+1和起点为0的步长为1的数组
    order = order[:, np.newaxis]#将order从shape=(10,)转换为shape=(10,1)
    e = np.tile(order, [1,N])#将order沿着y轴扩展1倍，沿着x轴扩展N倍（复制N个） N代表有几个数  tile：平铺
    XT = np.power(x_n, e)#计算次方：x_n为列表 e为列表，则分布求对应位置的次方，x_n为底数，e为幂
    X = np.transpose(XT)#transpose 作用是改变序列 二维不指定参数为矩阵转置
    a = np.matmul(XT, X) + lamda*np.identity(M+1) #X.T * X
    b = np.matmul(XT, t_n) #X.T * T
    w = np.linalg.solve(a,b) #aW = b => (X.T * X) * W = X.T * T   返回ax=b的解
    print("(从零次到高次)W:")
    print(w)

    e2 = np.tile(order, [1,x.shape[0]])
    XT2 = np.power(x, e2)
    p = np.matmul(w, XT2)
    return p, w
# return p, w

#预测函数
def Predict(w, grayscale_value_test):
    x = grayscale_value_test
    y = w[0] + w[1] * x + w[2] * x**2 + w[3] * x**3
    return y
# return y

#曲线拟合
def Curve_Fit(grayscale_value, moisture_content, n):
    x = np.linspace(50, 110, 6000)#产生从0到22的等差数列，数量2200个
    # print('grayscale_value:', grayscale_value)
    # print('moisture_content:', moisture_content)
    font = {'family': 'Times New Roman', 'size': 20}
    #图像绘制部分：横坐标含水率 纵坐标灰度值

    # 原始数据散点图
    # plt.figure(1, figsize=(8, 5))#plt.figure用于画图，自定义画布大小
    # # plt.plot(x, t, 'g', linewidth=3) #画曲线 plt.plot(x,y,format_string,**kwargs) format_string用于控制曲线风格，linewidth控制曲线宽度
    # plt.scatter(grayscale_value, moisture_content, color='', marker='o', edgecolors='b', s=100, linewidth=3, label="training data")#画散点图
    # draw_ticks()
    # plt.title('Figure 1 : origin_data', font)
    # plt.savefig('1.png', dpi=400)
    # # plt.show()

    # # M=1, N=5
    # p, w = regress(1, 5, x, moisture_content, grayscale_value)
    # # 图像绘制部分
    # plt.figure(2, figsize=(8, 5))
    # plt.plot(x, p, 'r', linewidth=3)
    # plt.scatter(moisture_content, grayscale_value, color='', marker='o', edgecolors='b', s=100, linewidth=3)
    # draw_ticks()
    # a0 = '%.2f'% w[0]
    # a1 = '%+.2f'% w[1]
    # plt.text(5, 100, "function: y = {}{}x".format(a0, a1), size=15,
    #          family="fantasy", color="r", style="italic", weight="light",
    #          bbox=dict(facecolor="r", alpha=0.2))
    # plt.title('Figure 3 : M = 1, N = 10', font)
    # plt.text(0.8, 0.9, 'M = 1', font, style='italic')
    # plt.savefig('2.png', dpi=400)

    # M=3, N=5
    p, w = regress(3, n, x, grayscale_value, moisture_content)
    # 图像绘制部分
    plt.figure(2, figsize=(8, 5))
    plt.plot(x, p, 'r', linewidth=3)
    plt.scatter(grayscale_value, moisture_content, color='', marker='o', edgecolors='b', s=100, linewidth=3)
    draw_ticks()
    a0 = '%.2f'% w[0]
    a1 = '%+.2f'% w[1]
    a2 = '%+.2f'% w[2]
    a3 = '%+.6f'% w[3]
    plt.text(65, 25, "function: y = {}{}x{}x*x{}x*x*x".format(a0, a1, a2, a3), size=15,
             family="fantasy", color="r", style="italic", weight="light",
             bbox=dict(facecolor="r", alpha=0.2))
    plt.title('Fit function diagram: order M = 3, fittings point N = {}'.format(n), font)
    plt.text(0.8, 0.9, 'M = 3', font, style='italic')
    plt.savefig('拟合函数图.png', dpi=400)
    # plt.show()

    print("function: y = {}{}x{}x^2{}x^3".format(a0, a1, a2, a3))
    print()
    return w
# return w

#判断文件是否为图片
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

#读取批量读取目标图片热力图信息
def Read_ROI(path, columns, rows):
    # #读取文件夹中所有图片的名字
    # filename_list = os.listdir(path)
    # # 给文件名加上路径
    # file_list = [os.path.join(path, i) for i in filename_list]

    image_filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]

    count_photo = 0
    for photo_path in image_filenames:
        photo_name = Get_filename(photo_path)
        count_photo = count_photo + 1
        print('Reading photo_ROI_{}...'.format(photo_name))
        lwpImg = cv.imread(photo_path)
        height = len(lwpImg)
        width = len(lwpImg[0])

        box_height = int(height/rows)
        box_width = int(width/columns)

        list_Snum = []
        list_aver_grayscale = []
        list_moisture = []
        count = 0
        for y in range(rows):
            for x in range(columns):
                count = count +1
                list_Snum.append(count)
                h = y * box_height #y=1
                wi = x * box_width #x=2
                h_1 = (y + 1) * box_height
                wi_1 = (x + 1) * box_width
                img_ROI = lwpImg[h:h_1, wi:wi_1] #wi是横坐标，h是纵坐标(wi,h)是左上角的点

                aver_histogram_rgb, img_histogram, b_hist, g_hist, r_hist = Read_value(img_ROI)
                list_aver_grayscale.append("%.3f" % float(aver_histogram_rgb))

                moisture = Predict(w, aver_histogram_rgb)
                list_moisture.append("%.3f" % float(moisture))

        # print('list_Snum', list_Snum)
        # print('list_aver_grayscale =', list_aver_grayscale)
        # print('list_moisture', list_moisture)
        # print('count =', count)

        data = list_aver_grayscale
        #绘制热力图
        Draw_Thermogram(data, photo_name, path)
    #制作热力图gif
    file_gif = path + r'\Thermogram'
    name = r'\Moisture_change.gif'
    create_gif(file_gif, name)

    return

#绘制图片热力图
def Draw_Thermogram(data, photo_name, path):
    print('Drawing thermogram_{}...'.format(photo_name))
    print()
    #绘制热力图
    data_frame = list_of_groups(data, 60)#分行
    data = pd.DataFrame(data_frame)  # 这时候是以raw为标准写入的
    # print(data)
    # print(type(data))
    for i in range(0,60,1):
        data[i] = pd.Series(data[i]).astype(float)
    f, ax1 = plt.subplots(figsize=(12, 6))
    sns.heatmap(data, annot=True, linewidths=0.02, ax=ax1, fmt='.0f', cmap='YlGnBu_r', annot_kws={'size':5})
    ax1.set_title('Grayvalue Thermogram_{}'.format(photo_name))
    ax1.set_xlabel('x-axis')
    # ax1.set_xticklabels([])  # 设置x轴图例为空值
    ax1.set_ylabel('y-axis')
    # plt.savefig('Grayvalue Thermogram_{}.png'.format(Snum), dpi=400)
    # plt.show()
    picture_path = r'Thermogram\{}.png'.format(photo_name)
    save_path = os.path.join(path, picture_path)
    plt.savefig(save_path, dpi=400)#保存图片

#列表分割函数
def list_of_groups(list_info, per_list_len):
    '''
    :param list_info:   列表
    :param per_list_len:  每个小列表的长度
    :return:
    '''
    list_of_group = zip(*(iter(list_info),) *per_list_len)
    end_list = [list(i) for i in list_of_group] # i is a tuple
    count = len(list_info) % per_list_len
    end_list.append(list_info[-count:]) if count !=0 else end_list
    return end_list

#生成gif
def create_gif(path, name):
    '''
    :param image_filenames: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间 数字越小，播放越快
    :return:
    '''
    print('createing gif...')
    image_filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]
    gif_name = path + name
    duration = 0.5
    frames = []
    for image_name in image_filenames:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    print('create finish, save as {}'.format(name))

    return

#------------------------------------分割线-----------------------------------------
if __name__ == '__main__':
    #获取训练集、测试集、目前检测集路径
    train = r"C:\Users\zhang\Desktop\the moisture content photo\train"
    test = r"C:\Users\zhang\Desktop\the moisture content photo\test"
    targe = r"C:\Users\zhang\Desktop\the moisture content photo\targe\0301"

    #读取训练集
    moisture_content_train, grayscale_value_train, count = Get_value(train, 1)
    #曲线拟合函数 并绘制
    w = Curve_Fit(grayscale_value_train, moisture_content_train, count)

    #读取测试集 并进行预测
    moisture_content_test, grayscale_value_test, count = Get_value(test, 2)

    #读取目标集 并进行预测
    grayscale_value_targe = Get_targe_value(targe)

    # 读取目标集热力图并保存
    Read_ROI(targe, 60, 40)

"""
1.拟合函数计算R^2
2.更改灰度值为三通道灰度值 √
3.绘制湿润锋（包括此时刻区域预测含水率）
"""