from matplotlib import pyplot as plt
import cv2 as cv
from decimal import *
import numpy as np


images = {}
histograms = {}

def calc_hist_old():
    #for image_number in range(1, 17):
    for k in parallel_experiments.keys():
        #image_name = f"2-{image_number}.tif"
        image_name = k
        #image_name.append(f"2-{image_number}.tif")
        images.setdefault(image_name, cv.imread(path + image_name))
        histograms.setdefault(image_name, cv.calcHist([images[image_name]], [0], None, [256], [0, 256]))
        sum_quantity = 0
        for i in histograms[image_name]:
            i[0] = i[0] / parallel_experiments[image_name]
            sum_quantity = sum_quantity + i[0]
        for i in histograms[image_name]:
            i[0] = i[0] * 100 / sum_quantity


def calc_single_hist(image, i, d=6):
    hist = cv.calcHist(image, [i], None, [256], [0, 256])
    sum_quantity = 0
    for i in hist:
        i[0] = i[0] / d
        sum_quantity = sum_quantity + i[0]
    for i in hist:
        i[0] = i[0] * 100 / sum_quantity
    return hist


def calc_hist(img):
    hist = cv.calcHist(img, [0], None, [256], [0, 256])
    return hist


def calc_mat_expectation(lst):  # на вход подается массив значений
    res = 0
    my_list = lst.flatten()  # преобразовние
    sum_of_pixels = sum(my_list)    # считаем общее кол-во значений тонов серого
    for pixel_num in range(256):    # проходим по каждому тону
        p_i = my_list[pixel_num] / sum_of_pixels    # вычисление вероятности данного тона
        res += p_i * pixel_num      # мат ожидание для i-го тона
    return res


def calc_mat_dispertion(lst):  # на вход подается массив значений
    res = 0
    my_list = lst.flatten()  # преобразовние
    sum_of_pixels = sum(my_list)    # считаем общее кол-во значений тонов серого
    for pixel_num in range(256):    # проходим по каждому тону
        p_i = my_list[pixel_num] / sum_of_pixels    # вычисление вероятности данного тона
        res += p_i * (pixel_num ** 2)      # мат ожидание для i-го тона
    return res - calc_mat_expectation(lst) ** 2

def show_plot(img):

    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]

    img_hist = calc_single_hist(img_b)
    print(f'mat of blue: {calc_mat_expectation(img_hist)}')

    # plt.figure(figsize=(1, 3))
    plt.ylabel('value')
    plt.xlabel('tone')
    x = [i for i in range(0, 256)]
    plt.xlim([0, 255])
    plt.plot(x, img_hist, color='blue')

    img_hist = calc_single_hist(img_g)
    print(f'mat of green: {calc_mat_expectation(img_hist)}')

    plt.ylabel('value')
    x = [i for i in range(0, 256)]
    plt.xlim([0, 255])
    plt.plot(x, img_hist, color='green')

    img_hist = calc_single_hist(img_r)
    print(f'mat of red: {calc_mat_expectation(img_hist)}')

    plt.ylabel('value')
    x = [i for i in range(0, 256)]
    plt.xlim([0, 255])
    plt.plot(x, img_hist, color='red')

    plt.show()


# считает математическое ожидание изображений и строит изображение его
def show_and_calc_hist():
    imgs_paths = ['overlaped_1_brightness_0.tif', 'overlaped_2_brightness_128.tif', 'overlaped_3_brightness_64.tif',
                  'overlaped_4_brightness_32.tif', 'overlaped_5_brightness_96.tif']

    img = cv.imread(imgs_paths[0])
    img = cv.imread('calib5p_res_crop.tif')
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

    # image = cv.imread(imgs_paths[0])

    for path in imgs_paths:
        img = cv.imread(path)
        print(path)
        # img =
        # img_b = img[:, :, 0]
        # img_g = img[:, :, 1]
        # img_r = img[:, :, 2]
        # GBR = [img_b, img_g, img_r]
        color_names = ['B', 'G', 'R']
        img_split = cv.split(img)

        for i in range(3):
            img_hist = cv.calcHist(img_split, [i], None, [256], [0, 256])
            mat = Decimal(calc_mat_expectation(img_hist))
            print(f'mat of {color_names[i]}: {mat.quantize(Decimal("1.000"))}')
        print()


if __name__ == '__main__':
    # path = 'overlaped_1_brightness_0.tif'
    # path = 'overlaped_2_brightness_128.tif'
    # path = 'overlaped_3_brightness_64.tif'
    # path = 'overlaped_4_brightness_32.tif'
    # path = 'overlaped_5_brightness_96.tif'

    #show_and_calc_hist()


# -----------------------------------------------------------------
# обрезка изображений под 1 размер
#-----------------------------------------------------------------
    # imgs_paths = ['9', 'overlaped_1_brightness_0', 'overlaped_4_brightness_32', 'overlaped_3_brightness_64',
    #               'overlaped_5_brightness_96', 'overlaped_2_brightness_128', 'calib5p_res_crop',
    #               'calib3p_res_crop', 'calib2p_res_crop']
    #
    # delims = [[1, 0, 11, 11], [0, 0, 11, 11], [15, 16, 2, 2], [14, 14, 0, 0], [16, 16, 2, 2], [15, 15, 0, 1],
    #           [16, 16, 2, 3], [16, 17, 0, 1], [16, 16, 6, 6]]
    # i = 0
    # for path in imgs_paths:
    #     img = cv.imread(path + '.tif')
    #     y, x, _ = img.shape
    #     x_start, x_stop, y_start, y_stop = delims[i]
    #     # print(f'{path} {x} {y}')
    #     img = img[y_start:y - y_stop, x_start:x - x_stop]
    #     y, x, _ = img.shape
    #     print(f'{path} {x} {y}')
    #     i += 1
    #
    #     cv.imwrite('C:/Users/Root/Desktop/crop/' + path + '.tif', img)

# -----------------------------------------------------------------



#----------------------------------------------------------
# средние макc мин значения
# ----------------------------------------------------------

    imgs_paths = ['overlaped_1_brightness_0.tif', 'overlaped_2_brightness_128.tif', 'overlaped_3_brightness_64.tif',
                  'overlaped_4_brightness_32.tif', 'overlaped_5_brightness_96.tif', 'calib3p_res_crop.tif', '1.tif',
                  'overlap_2985x1772.tif', 'calib5p_2985x1772.tif', '9_2985x1772.tif']

    imgs_paths = ['9', 'overlaped_1_brightness_0', 'overlaped_4_brightness_32', 'overlaped_3_brightness_64',
                  'overlaped_5_brightness_96', 'overlaped_2_brightness_128', 'calib5p_res_crop',
                  'calib3p_res_crop', 'calib2p_res_crop']


    for i in range(9):

        path = imgs_paths[i]
        print(path)
        #img = cv.imread(path)
        img = cv.imread('C:/Users/Root/Desktop/crop/' + path + '.tif')

        print(img.shape)
        y, x, _ = img.shape
        img = img[5:y - 5, 5:x - 5]
        print(img.shape)
        img_g = img[:, :, 1]

        # print(imgs_paths[0])
        # print(f' max val from all {img_g.max()}')
        # print(f' avg val from all {np.average(img_g)}')
        # print(f' min val from all {img_g.min()}')

       # img_g = cv.GaussianBlur(img_g, (11, 11), 9)
        img = cv.GaussianBlur(img, (21, 21), 15)

        histr = cv.calcHist([img], [1], None, [256], [0, 256])

        for i in range(len(histr)):
            if histr[i] <= 200:
                histr[i] = 0

        plt.figure(figsize=(8, 5))

        with open('res.txt', 'w', encoding='utf8') as file:
            for val in histr:
                file.write(str(int(val)) + '\n')


        accum = 0
        prev = 0
        min_r = 0
        for i in range(len(histr)):
            if histr[i] > 0:
                accum += histr[i]
                if accum >= 1000:
                    print(f'min = {i}')
                    min_r = i
                    plt.axvline(x=i, ymin=0.05, ymax=0.55, color='red', ls='--')
                    break
            # if histr[i] - prev > 1500:
            #     print(f'min = {i}')
            #     min_r = i
            #     plt.axvline(x=i, ymin=0.05, ymax=0.55, color='red', ls='--')
            #     break
            # else:
            #     prev = histr[i]

        accum = 0
        prev = 0
        max_r = 0
        for i in range(len(histr) - 1, 0, - 1):
            if histr[i] > 0:
                accum += histr[i]
                if accum >= 1000:
                    print(f'max = {i}')
                    max_r = i
                    plt.axvline(x=i, ymin=0.05, ymax=0.55, color='red', ls='--')
                    break
            # if histr[i] - prev > 1500:
            #     print(f'max = {i}')
            #     max_r = i
            #     plt.axvline(x=i, ymin=0.05, ymax=0.55, color='red', ls='--')
            #     break
            # else:
            #     prev = histr[i]

        print(f'range: = {max_r - min_r}')
        print(f'avg = {round(np.average(img_g), 3)}')

        print()

        MAT = round(calc_mat_expectation(histr), 3)
        print(f'M[x] = {MAT}')
        print(f'D[x] = {round(calc_mat_dispertion(histr), 3)}')
        sigma = calc_mat_dispertion(histr) ** 0.5
        print(f'СКО = {round(sigma, 3)}')

     #   plt.axvline(x=MAT, ymin=0.05, ymax=0.66, color='y', ls='--')
     #   plt.axvline(x=MAT - 3 * sigma, ymin=0.05, ymax=0.55, color='b', ls='dotted')
      #  plt.axvline(x=MAT + 3 * sigma, ymin=0.05, ymax=0.55, color='b', ls='dotted')

        print(f'sigma min: {round(MAT - 3 * sigma, 3)}\nsigma max: {round(MAT + 3 * sigma, 3)}')

        plt.plot(histr, color='g')
        plt.xlim([min_r - 10, max_r + 10])

        import matplotlib.ticker as mticker
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        plt.title(f'{path}')
        plt.ylabel('Количество точек')
        plt.xlabel('Диапазон точек')

        #plt.show()

        plt.savefig('C:/Users/Root/Desktop/crop/' + path, bbox_inches='tight')

    # img_g = list(img_g.flatten())
    #
    # import random
    #
    # smp = random.sample(img_g, 1000)
    # print(f' max val from 1000 sample {max(smp)}')
    # print(f' avg val from 1000 sample {sum(smp) / len(smp)}')
    # print(f' min val from 1000 sample {min(smp)}')
    #
    #
    # f = lambda x: 0.3754 * x * x - 104.88 * x + 7481
    #
    # print(f(174))

# ----------------------------------------------------------





#-----------------------------------------
#   отрисовка с колорбаром и колормэпом
#__________________________________________

    # img = cv.imread('Screenshot_1.png')
    # img = cv.imread('calib3p_res_crop.tif')
    # #img = cv.imread('0.tif')
    # y, x, _ = img.shape
    # img = img[20:y - 20, 20:x + 20]
    # img_g = img[:, :, 1]
    # plt.figure()
    # #plt.pcolormesh(img_g, cmap='Greys')
    # plt.pcolormesh(img_g, cmap='viridis')
    # plt.colorbar()
    # plt.show()
# -----------------------------------------








    # 3D график крутое
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.ticker import LinearLocator
    #
    #
    #
    #
    # # generate example data
    # import numpy as np


    # x, y = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
    # z = np.cos(x * np.pi) * np.sin(y * np.pi)
    #
    # # actual plotting example
    # fig = plt.figure()
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis')
    # ax2 = fig.add_subplot(122)
    # cf = ax2.contourf(x, y, z, 51, vmin=-1, vmax=1, cmap='viridis')
    # cbar = fig.colorbar(cf)
    # cbar.locator = LinearLocator(numticks=11)
    # cbar.update_ticks()
    # for ax in {ax1, ax2}:
    #     ax.set_xlabel(r'$x$')
    #     ax.set_ylabel(r'$y$')
    #     ax.set_xlim([-1, 1])
    #     ax.set_ylim([-1, 1])
    #     ax.set_aspect('equal')
    #
    # ax1.set_zlim([-1, 1])
    #ax1.set_zlabel(r'$\cos(\pi x) \sin(\p    i y)$')


   # show_plot(image)