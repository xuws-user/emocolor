import numpy as np
from sko.SA import SA
from weighting import te_zen_be, ex_ey
import math


def using_sa(load, choose_model, feature_list, L, max_stay_counter, ex0, ey0, need_be):
    def ex0_ex1_or_ey0_ey1distance(x, t, load2, need_be):
        def rgb2lab(rgb_color_list):

            lab_color_list = []
            referenceX = 95.047
            referenceY = 100.000
            referenceZ = 108.883

            for rgb_color_list_i in range(0, len(rgb_color_list)):

                R = rgb_color_list[rgb_color_list_i][0]
                G = rgb_color_list[rgb_color_list_i][1]
                B = rgb_color_list[rgb_color_list_i][2]

                if R > 0.04045:
                    R = math.pow(((R + 0.055) / 1.055), 2.4)
                else:
                    R /= 12.92
                if G > 0.04045:
                    G = math.pow(((G + 0.055) / 1.055), 2.4)
                else:
                    G /= 12.92
                if B > 0.04045:
                    B = math.pow(((B + 0.055) / 1.055), 2.4)
                else:
                    B /= 12.92

                R *= 100
                G *= 100
                B *= 100

                X = (R * 0.4124) + (G * 0.3576) + (B * 0.1805)
                Y = (R * 0.2126) + (G * 0.7152) + (B * 0.0722)
                Z = (R * 0.0193) + (G * 0.1192) + (B * 0.9505)

                # XYZ to Lab
                X /= referenceX
                Y /= referenceY
                Z /= referenceZ

                if X > 0.008856:
                    X = math.pow(X, 1 / 3)

                else:
                    X = (7.787 * X) + (16 / 116)

                if Y > 0.008856:
                    Y = math.pow(Y, 1 / 3)

                else:
                    Y = (7.787 * Y) + (16 / 116)

                if Z > 0.008856:
                    Z = math.pow(Z, 1 / 3)

                else:
                    Z = (7.787 * Z) + (16 / 116)

                L = (116 * Y) - 16
                a = 500 * (X - Y)
                b = 200 * (Y - Z)

                lab_color_list.append([round(L / 100, 4), round((a + 128) / 256, 4), round((b + 128) / 256, 4)])

            return lab_color_list

        def rgb2hsv(rgb):
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]
            # R, G, B values are divided by 255
            # to change the range from 0..255 to 0..1:
            # h, s, v = hue, saturation, value

            cmax = max(r, g, b)  # maximum of r, g, b
            cmin = min(r, g, b)  # minimum of r, g, b
            diff = cmax - cmin  # diff of cmax and cmin.
            # if cmax and cmax are equal then h = 0
            if cmax == cmin:
                h = 0
            # if cmax equal r then compute h
            elif cmax == r:
                h = (60 * ((g - b) / diff) + 360) % 360
            # if cmax equal g then compute h
            elif cmax == g:
                h = (60 * ((b - r) / diff) + 120) % 360
            # if cmax equal b then compute h
            elif cmax == b:
                h = (60 * ((r - g) / diff) + 240) % 360
            # if cmax equal zero
            if cmax == 0:
                s = 0
            else:
                s = (diff / cmax)
            # compute v
            v = cmax
            return [round(h / 360, 4), round(s, 4), round(v, 4)]

        def min_lab_distance(rgb_data):
            def distance(color_a, color_b):
                return (sum([(a - b) ** 2 for a, b in zip(color_a, color_b)])) ** 0.5

            lab_data1127 = rgb2lab(rgb_data)
            hsv_h = []
            for rgb_data_i_i in range(0, len(rgb_data)):
                hsv_h.append(rgb2hsv(rgb_data[rgb_data_i_i])[0])
            one_all_lab_data1127 = []
            for lab_data_i1127 in range(0, len(lab_data1127)):
                for lab_data_i11271 in range(lab_data_i1127 + 1, len(lab_data1127)):
                    if 0.10 > lab_data1127[lab_data_i1127][0] or lab_data1127[lab_data_i1127][0] > 0.90:
                        bed = 1
                    if (85 / 360) < hsv_h[lab_data_i1127] < (114 / 360):
                        bed = 1
                    one_all_lab_data1127.append(distance(lab_data1127[lab_data_i1127], lab_data1127[lab_data_i11271]))
                    print(distance(lab_data1127[lab_data_i1127], lab_data1127[lab_data_i11271]))
            return min(one_all_lab_data1127)



        if t == 0:
            need11 = []
            need2 = []
            i_x = 0
            for i_need2 in range(0, len(need_be)):
                if load2[i_need2] == 1:
                    need11.append(need_be[i_need2])
                else:
                    need11.append(x[i_x])
                    i_x = i_x + 1
            len1 = int(len(need11) / 3)
            for len_of_x_i in range(0, len1):
                need2.append([need11[len_of_x_i * 3], need11[len_of_x_i * 3 + 1], need11[len_of_x_i * 3 + 2]])
            ans_min_lab_ = min_lab_distance(need2)
            need2 = te_zen_be(feature_list, need2)
            ex, ey = ex_ey(choose_model, feature_list, need2)
            return np.sqrt((ex - ex0) ** 2 + (ey - ey0) ** 2) - 0.2 * ans_min_lab_
        else:
            need11 = []
            need2 = []
            i_x = 0
            for i_need2 in range(0, len(need_be)):
                if load2[i_need2] == 1:
                    need11.append(need_be[i_need2])
                else:
                    need11.append(x[i_x])
                    i_x = i_x + 1
            len1 = int(len(need11) / 3)
            for len_of_x_i in range(0, len1):
                need2.append([need11[len_of_x_i * 3], need11[len_of_x_i * 3 + 1], need11[len_of_x_i * 3 + 2]])
            need2 = te_zen_be(feature_list, need2)
            ex, ey = ex_ey(choose_model, feature_list, need2)
            return ex, ey, ex0, ey0


    load1 = []
    load2 = []
    need_be1 = []
    for load1_i in range(0, len(load)):
        load1.append(load[load1_i])
        load1.append(load[load1_i])
        load1.append(load[load1_i])
    for load1_i in range(0, len(need_be)):
        load2.append(load1[load1_i])
    for load1_i in range(0, len(need_be)):
        if load2[load1_i] == 0:
            need_be1.append(need_be[load1_i])

    sa = SA(load2=load2, need_be=need_be, func=ex0_ex1_or_ey0_ey1distance, lb=0, ub=1, x0=need_be1, T_max=3000,
            T_min=1e-9, L=L,
            max_stay_counter=max_stay_counter)
    best_x, best_y = sa.run()
    need11 = []
    i_x = 0
    for i_need2 in range(0, len(need_be)):
        if load2[i_need2] == 1:
            need11.append(need_be[i_need2])
        else:
            need11.append(best_x[i_x])
            i_x = i_x + 1
    return need11, best_y
