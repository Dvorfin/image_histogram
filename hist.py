from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np


def calc_mat_expectation(lst):  # на вход подается массив значений
    res = 0
    my_list = lst.flatten()  # преобразовние
    sum_of_pixels = sum(my_list)    # считаем общее кол-во значений тонов серого
    for pixel_num in range(256):    # проходим по каждому тону
        p_i = my_list[pixel_num] / sum_of_pixels    # вычисление вероятности данного тона
        res += p_i * pixel_num      # мат ожидание для i-го тона
    return res



imgs_paths = ['overlaped_1_brightness_0.tif', 'overlaped_2_brightness_128.tif', 'overlaped_3_brightness_64.tif',
                  'overlaped_4_brightness_32.tif', 'overlaped_5_brightness_96.tif']

img = cv.imread(imgs_paths[2])

img_split = cv.split(img)
histSize = 256
histRange = (0, 256) # the upper boundary is exclusive
accumulate = False
b_hist = cv.calcHist(img_split, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv.calcHist(img_split, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv.calcHist(img_split, [2], None, [histSize], histRange, accumulate=accumulate)

r = [b_hist, g_hist, r_hist]
color_names = ['B', 'G', 'R']
for i in range(3):
    print(f'mat of {color_names[i]}: {calc_mat_expectation(r[i])}')

hist_w = 512
hist_h = 400
bin_w = int(round( hist_w/histSize ))
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
for i in range(1, histSize):
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(b_hist[i]) ),
            ( 255, 0, 0), thickness=2)
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(g_hist[i]) ),
            ( 0, 255, 0), thickness=2)
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(r_hist[i]) ),
            ( 0, 0, 255), thickness=2)
# cv.imshow('Source image', img)
cv.imshow('calcHist Demo', histImage)
cv.waitKey()
