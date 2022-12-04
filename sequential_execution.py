import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageCms

from build_palette import build_palette
from change_colors import image_transfer, lab2rgb


def ByteLAB(LAB):
    return int(LAB[0] / 100 * 255), int(LAB[1] + 128), int(LAB[2] + 128)


def RGBtoXYZ(RGB):
    def f(n):
        return n / 12.92 if n <= 0.04045 else ((n + 0.055) / 1.055) ** 2.4

    R, G, B = [f(x / 255) for x in RGB]
    X = (0.4124 * R + 0.3576 * G + 0.1805 * B) * 100
    Y = (0.2126 * R + 0.7152 * G + 0.0722 * B) * 100
    Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) * 100
    return X, Y, Z


def XYZtoLAB(XYZ):
    def f(n):
        return n ** (1 / 3) if n > (6 / 29) ** 3 else (n / (3 * ((6 / 29) ** 2))) + (4 / 29)

    X, Y, Z = XYZ
    X /= 95.047
    Y /= 100.000
    Z /= 108.883

    L = 116 * f(Y) - 16
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return L, a, b


def RGBtoLAB(RGB):
    return XYZtoLAB(RGBtoXYZ(RGB))


def rgb2lab(image):
    RGB_p = ImageCms.createProfile('sRGB')
    LAB_p = ImageCms.createProfile('LAB')
    return ImageCms.profileToProfile(image, RGB_p, LAB_p, outputMode='LAB')


def wirting_image_of_color(data, name):
    if len(data) > 5:
        fig = plt.figure(figsize=(5, 2))
    else:
        fig = plt.figure(figsize=(5, 1))
    ax = fig.add_subplot(111)
    for data_i in range(0, len(data)):
        rect = plt.Rectangle((0.1 * (data_i % 5), 0.1 * np.floor(data_i / 5)), 0.1, 0.1,
                             color=(data[data_i][0] / 256, data[data_i][1] / 256, data[data_i][2] / 256))
        # print(data[data_i])
        ax.add_patch(rect)
    plt.xlim(0, 0.5)

    if len(data) <= 5:
        plt.ylim(0, 0.1)
    else:
        plt.ylim(0, 0.2)
    plt.axis("off")
    # plt.savefig(savePath + str(name12) + '.png', dpi=600, bbox_inches='tight', pad_inches=-0.1)
    # plt.savefig("static/palette_new/"+str(name) + ".jpg")
    plt.show()


#  [[211, 224, 219], [55, 71, 57], [163, 184, 174], [117, 134, 115], [23, 38, 35]]
#  0.8256831856374061 0.017622583418092197
#  [1.54892812e-03 8.48898743e-01 9.99923578e-01 3.20709332e-01
#  7.09103925e-01 6.02145515e-04 5.95076472e-01 6.08004202e-01
#  5.76137550e-03 5.25745857e-01 7.96860264e-01 2.55169375e-01
#  5.50576472e-01 9.98724144e-01 9.99999888e-01]
#  [[0.39652559872 ,
# 217.318078208 ,
# 255.980435968] ,
# [82.101588992 ,
# 181.5306048 ,
# 0.15414925184] ,
# [152.339576832 ,
# 155.649075712 ,
# 1.474912128] ,
# [134.590939392 ,
# 203.996227584 ,
# 65.32336] ,
# [140.947576832 ,
# 255.673380864 ,
# 255.999971328 ]]
#  [[211, 224, 219], [55, 71, 57], [163, 184, 174], [117, 134, 115], [23, 38, 35]]
#  [[213.090310912015, 225.33931286253582, 220.3695942849993], [155.2660099290232, 177.9286257121156, 166.88501068104242],
#  [95.33711170393202, 108.70522153321535, 89.60880698028473],
#  [36.92578286840363, 53.63532281907892, 45.37666285620789], [15.747649062453394, 32.24504402013099, 31.963836404927196]]
#  1
def change(image_name, new_palette1, new_m_name):
    image_rgb = Image.open(image_name)
    image_lab = rgb2lab(image_rgb)
    palette, coool, need = build_palette(image_lab, 5)
    wirting_image_of_color(new_palette1, "no-1")

    wirting_image_of_color(new_palette1, "no-1")

    new_palette = []
    for xxxx in range(0, len(new_palette1)):
        new_palette.append(ByteLAB(RGBtoLAB(new_palette1[xxxx])))
    # print(new_palette)
    image_lab_m = image_transfer(image_lab, palette, new_palette, sample_level=10, luminance_flag=False)
    image_rgb_m = lab2rgb(image_lab_m)
    image_rgb_m.save("static/photo_center/"+str(new_m_name)+".jpg")


def be_color(image_name):
    image_rgb = Image.open(image_name)
    img = image_rgb
    image_lab = rgb2lab(image_rgb)
    palette, coool, need = build_palette(image_lab, 6)
    zhen = []
    for i_need in range(0, len(need) - 1):
        zhen.append([need[i_need][0], need[i_need][1], need[i_need][2]])
    return coool, zhen
