from PIL import Image, ImageCms
import numpy as np
import os
import random
import matplotlib.pyplot as plt


def chiyun(data1,name):
    for data_1i in range(0, len(data1)):
        data = []
        data1_tmp = []
    for i_1255 in range(0, len(data1)):
        if (i_1255 + 1) % 3 == 0:
            data1_tmp.append(data1[i_1255])
            data.append(data1_tmp)
            data1_tmp = []
        else:
            data1_tmp.append(data1[i_1255])
    print(data)
    data.reverse()
    print(data)
    ans = "C:/Users/xws/PycharmProjects/flaskProject/static/poto_for_woeder/wordle" + str(len(data)) + ".png";
    image = Image.open(
        "C:/Users/xws/PycharmProjects/flaskProject/static/poto_for_woeder/wordle" + str(len(data)) + ".png")
    will_change_color_list = []
    image_data = np.array(image)
    im = Image.new("RGB", (len(image_data[0]), len(image_data)))
    for I_image_data_i in range(0, len(image_data)):
        for I_image_data_i_i in range(0, len(image_data[0])):
            if image_data[I_image_data_i][I_image_data_i_i][0] != 255 and image_data[I_image_data_i][I_image_data_i_i][
                1] != 255 and image_data[I_image_data_i][I_image_data_i_i][2] != 255:
                will_tmp = False
                for will_change_color_list_i in range(0, len(will_change_color_list)):
                    if will_change_color_list[will_change_color_list_i][0] == \
                            image_data[I_image_data_i][I_image_data_i_i][0] and \
                            will_change_color_list[will_change_color_list_i][1] == \
                            image_data[I_image_data_i][I_image_data_i_i][1] and \
                            will_change_color_list[will_change_color_list_i][2] == \
                            image_data[I_image_data_i][I_image_data_i_i][2]:
                        will_change_color_list[will_change_color_list_i][3] = \
                            will_change_color_list[will_change_color_list_i][3] + 1
                        will_tmp = True
                if not will_tmp:
                    will_change_color_list.append([image_data[I_image_data_i][I_image_data_i_i][0],
                                                   image_data[I_image_data_i][I_image_data_i_i][1],
                                                   image_data[I_image_data_i][I_image_data_i_i][2], 1])
    print(will_change_color_list)

    def myFunc(e):
        return e[3]

    will_change_color_list.sort(key=myFunc)
    print(will_change_color_list)

    for I_image_data_i in range(0, len(image_data)):
        for I_image_data_i_i in range(0, len(image_data[0])):
            if image_data[I_image_data_i][I_image_data_i_i][0] != 255 and image_data[I_image_data_i][I_image_data_i_i][
                1] != 255 and image_data[I_image_data_i][I_image_data_i_i][2] != 255:
                for will_change_color_list_i in range(0, len(will_change_color_list)):
                    if will_change_color_list[will_change_color_list_i][0] == \
                            image_data[I_image_data_i][I_image_data_i_i][0] and \
                            will_change_color_list[will_change_color_list_i][1] == \
                            image_data[I_image_data_i][I_image_data_i_i][1] and \
                            will_change_color_list[will_change_color_list_i][2] == \
                            image_data[I_image_data_i][I_image_data_i_i][2]:
                        image_data[I_image_data_i][I_image_data_i_i][0] = data[will_change_color_list_i][0]
                        image_data[I_image_data_i][I_image_data_i_i][1] = data[will_change_color_list_i][1]
                        image_data[I_image_data_i][I_image_data_i_i][2] = data[will_change_color_list_i][2]
                        im.putpixel((I_image_data_i_i, I_image_data_i), (int(data[will_change_color_list_i][0]), int(data[will_change_color_list_i][1]),
                            int(data[will_change_color_list_i][2])))
            else:
                im.putpixel((I_image_data_i_i, I_image_data_i), (255, 255, 255))
    im.save("static/photo_center/" + str(name) + ".jpg")

    print(will_change_color_list)

#[[223, 52, 52, 7910], [245, 107, 65, 14229], [253, 174, 97, 14901], [26, 152, 79, 15100], [254, 88, 64, 18155], [167, 217, 106, 20973]]

