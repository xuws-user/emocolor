import math

import numpy as np
import pandas as pd
import scipy


def ex_ey(choose, feature_list, te_zen):
    ex_list = [-0.335452814,
               0.346868343,
               0.097997945,
               0.315648773,
               0.419328294,
               0.237411015]
    ey_list = [0.346888169,
               0.300813017,
               0.090772497,
               0.225620258,
               0.002247504,
               0.05485013]
    if choose == 3:
        ex_list = [-0.230969312,
                   0.27622583,
                   0.13554217,
                   0.31153197,
                   0.450465104,
                   0.235845824
                   ]
        ey_list = [0.381424819,
                   0.343273182,
                   0.103641932,
                   0.284487751,
                   -0.014323594,
                   0.007372746]
    if choose == 5:
        ex_list = [-0.337009037,
                   0.333024077,
                   0.104918871,
                   0.333959126,
                   0.432483885,
                   0.213433527]
        ey_list = [0.371121999,
                   0.321825147,
                   0.076694489,
                   0.244253057,
                   0.031715767,
                   0.014348025]
    if choose == 7:
        ex_list = [-0.335452814,
                   0.346868343,
                   0.097997945,
                   0.315648773,
                   0.419328294,
                   0.237411015]
        ey_list = [0.346888169,
                   0.300813017,
                   0.090772497,
                   0.225620258,
                   0.002247504,
                   0.05485013]
        if choose == 10:
            ex_list = [-0.299535094,
                       0.327593552,
                       0.079750827,
                       0.289313708,
                       0.431613158,
                       0.240070518
                       ]
            ey_list = [0.311711733,
                       0.322411197,
                       0.095946948,
                       0.195653721,
                       0.019895577,
                       0.05374676
                       ]

    ex_be = 0
    ey_be = 0

    for te_zen_i in range(0, len(te_zen)):
        if feature_list[te_zen_i] == 1:
            ex_be += ex_list[te_zen_i] * te_zen[te_zen_i]
            ey_be += ey_list[te_zen_i] * te_zen[te_zen_i]

    return ex_be, ey_be


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


def hue_p(hue_p_data):
    data = []

    for hue_p_data_i in range(0, len(hue_p_data)):
        data.append(
            [hue_p_data[hue_p_data_i][0] * 255, hue_p_data[hue_p_data_i][1] * 255, hue_p_data[hue_p_data_i][2] * 255])

    hueProbs_hueJoint = []

    wb = pd.read_excel('hueJoint.xlsx', header=None)
    wb = wb.values

    for wb_i in range(0, len(wb)):

        hueProbs_hueJoint1 = []

        for wb_1_ii in range(0, len(wb[0])):
            hueProbs_hueJoint1.append(wb[wb_i][wb_1_ii])

        hueProbs_hueJoint.append(hueProbs_hueJoint1)

    wb = pd.read_excel('hueAdjacency.xlsx', header=None)
    wb = wb.values

    hueProbs_hueAdjacency = []
    for wb_i in range(0, len(wb)):
        hueProbs_hueAdjacency1 = []
        for wb_ii in range(0, len(wb[0])):
            hueProbs_hueAdjacency1.append(wb[wb_i][wb_ii])
        hueProbs_hueAdjacency.append(hueProbs_hueAdjacency1)
    hueProbs_hueProb = []

    wb = pd.read_excel('hueProb.xlsx', header=None)
    wb = wb.values

    for wb_i in range(0, len(wb)):
        hueProbs_hueProb1 = []
        for wb_ii in range(0, len(wb[0])):
            hueProbs_hueProb1.append(wb[wb_i][wb_ii])
        hueProbs_hueProb.append(hueProbs_hueProb1)
    satValThresh = 0.2
    hsv = []
    hsv1 = []
    for data_len_i in range(0, len(data)):
        data1 = rgb2hsv(data[data_len_i])
        hsv.append([data1[0] / 360, data1[1] / 100, data1[2] / 100])
        hsv1.append(data1)
    selectColors = []
    for hsv_len_i in range(0, len(hsv)):
        minMin = min(hsv[hsv_len_i][2], hsv[hsv_len_i][2])
        if minMin >= satValThresh:
            selectColors.append(1)
        else:
            selectColors.append(0)
    visHues = []
    for hsv1_len_i in range(0, len(hsv1)):
        if selectColors[hsv1_len_i] == 1:
            visHues.append(hsv1[hsv1_len_i][0])

    hueJointList = []

    for h1 in range(0, len(visHues)):
        for h2 in range(h1, len(visHues)):
            hueJointList1 = hueProbs_hueJoint[int(visHues[h1])][int(visHues[h2])]
            hueJointList.append(int(hueJointList1))
    hueAdjList = []
    # print('sssss')
    # print(visHues)
    for h1 in range(0, len(visHues) - 1):
        hueAdjList1 = hueProbs_hueAdjacency[int(visHues[h1])][int(visHues[h1 + 1])]
        hueAdjList.append(hueAdjList1)
    hueProbFeatures1 = []
    for i in range(0, len(visHues)):
        hueProbFeatures1.append(hueProbs_hueProb[int(visHues[i])])
    alpha = []
    for i in range(0, 360):
        alpha.append(2 * math.pi * (i / 360))
    pMix = []

    for j in range(0, len(visHues)):
        circ_vmpdf = 1 / (2 * math.pi * scipy.special.jv(0, 2 * math.pi)) * math.exp(
            2 * math.pi * math.cos(alpha[j] - visHues[j]) * 2 * math.pi)
        pMix.append(circ_vmpdf)
    pMix_all = sum(pMix)
    for i in range(0, len(pMix)):
        pMix[i] = pMix[i] / pMix_all

    entropy = 0  # 要求的
    for i in range(0, len(pMix)):
        entropy = entropy + pMix[i] * np.log(pMix[i])
    # print(float(entropy))
    return entropy


def color_name_difference(a, b):
    f = open("t.txt")
    line = f.readline()
    t_list = []
    while line:
        num = list(map(int, line.split()))
        t_list.append(int(num[0]))
        line = f.readline()
    f.close()

    f = open("lab.txt")
    line = f.readline()
    lab_list = []
    while line:
        num = list(map(float, line.split()))
        lab_list.append([num[0], num[1], num[2]])
        line = f.readline()
    f.close()

    a = rgb2lab([a])

    a = a[0]

    a = [a[0] * 100, a[1] * 100, a[2] * 100]

    b = rgb2lab([b])

    b = b[0]

    b = [b[0] * 100, b[1] * 100, b[2] * 100]

    near_a = []

    near_b = []

    for lab_list_i in range(0, len(lab_list)):

        a_one_distance = 0
        b_one_distance = 0

        for one_color_i in range(0, 3):
            a_one_distance += (a[one_color_i] - lab_list[lab_list_i][one_color_i]) ** 2
            b_one_distance += (b[one_color_i] - lab_list[lab_list_i][one_color_i]) ** 2

        near_a.append(a_one_distance)
        near_b.append(b_one_distance)

    min_a_index = near_a.index(min(near_a))
    min_b_index = near_b.index(min(near_b))

    W = 153

    sa = 0
    sb = 0
    sc = 0

    for w_i in range(0, W):
        if min_a_index * W + w_i == 0:
            ta = 0
            tb = 0

        else:
            ta = t_list[min_a_index * W + w_i]
            tb = t_list[min_b_index * W + w_i]

        sa = sa + ta * ta
        sb = sb + tb * tb
        sc = sc + ta * tb

    Color_Difference = 1 - sc / (math.sqrt(sb * sa))

    return Color_Difference


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


def lab2rgb(Lab, bound=1):
    # l : 0 100
    # a b -50 +50
    # Lab to XYZ
    referenceX = 95.047
    referenceY = 100.000
    referenceZ = 108.883

    Y = (Lab[0] + 16) / 116
    X = Lab[1] / 500 + Y
    Z = Y - Lab[2] / 200

    if math.pow(Y, 3) > 0.008856:
        Y = math.pow(Y, 3)
    else:
        Y = (Y - 16 / 116) / 7.787
    if math.pow(X, 3) > 0.008856:
        X = math.pow(X, 3)
    else:
        X = (X - 16 / 116) / 7.787
    if math.pow(Z, 3) > 0.008856:
        Z = math.pow(Z, 3)
    else:
        Z = (Z - 16 / 116) / 7.787

    X *= referenceX
    Y *= referenceY
    Z *= referenceZ

    # XYZ to sRGB
    X /= 100
    Y /= 100
    Z /= 100

    R = X * 3.2406 + Y * -1.5372 + Z * -0.4986
    G = X * -0.9689 + Y * 1.8758 + Z * 0.0415
    B = X * 0.0557 + Y * -0.2040 + Z * 1.0570

    if R > 0.0031308:
        R = 1.055 * math.pow(R, 1 / 2.4) - 0.055
    else:
        R *= 12.92
    if G > 0.0031308:
        G = 1.055 * math.pow(G, 1 / 2.4) - 0.055
    else:
        G *= 12.92
    if B > 0.0031308:
        B = 1.055 * math.pow(B, 1 / 2.4) - 0.055
    else:
        B *= 12.92

    R *= 255
    G *= 255
    B *= 255

    if bound:
        R = round(min(max(R, 0), 255))
        G = round(min(max(G, 0), 255))
        B = round(min(max(B, 0), 255))

    return [R, G, B]


def distance(piont):
    distance_of = []
    num = 0
    for piont_i in range(0, len(piont) - 1):
        for piont_i_i in range(piont_i + 1, len(piont)):
            num = num + 1
            distance_one = 0

            for piont_i_i_i in range(0, 3):
                distance_one += (piont[piont_i][piont_i_i_i] - piont[piont_i_i][piont_i_i_i]) ** 2

            distance_of.append(np.sqrt(distance_one))

    return distance_of


def te_zen_be(feature_list, need):
    ans = []
    rgb = []

    for i in range(0, len(need)):
        rgb.append([need[i][0], need[i][1], need[i][2]])

    lab = rgb2lab(rgb)
    lab_a = []
    for no_1_lab_a_max_i in range(0, len(lab)):
        lab_a.append(lab[no_1_lab_a_max_i][1])
    no_1_lab_a_max = max(lab_a)
    if feature_list[0] == 1:
        ans.append(no_1_lab_a_max)

    no_2_std = np.std(lab_a)
    if feature_list[1] == 1:
        ans.append(no_2_std)

    hsv = []
    for i_hsv in range(0, len(rgb)):
        hsv.append(rgb2hsv(rgb[i_hsv]))
    hsv_h = []
    hsv_v = []
    for hsv_i in range(0, len(hsv)):
        hsv_h.append(hsv[hsv_i][0])
        hsv_v.append(hsv[hsv_i][2])
    if feature_list[2] == 1:
        no_3_hsv_h_max = max(hsv_h)
        ans.append(no_3_hsv_h_max)
    if feature_list[3] == 1:
        no_4_hsv_v_max = max(hsv_v)
        ans.append(no_4_hsv_v_max)
    if feature_list[4] == 1:
      no_5_rgb_1_g = rgb[0][1]
      ans.append(no_5_rgb_1_g)
    #if feature_list[5] == 1:
        #no_6_rgb_2_g = rgb[1][1]
        #ans.append(no_6_rgb_2_g)
    if feature_list[5] == 1:

        no_6_hsv_2_v = hsv[1][2]
        ans.append(no_6_hsv_2_v)
    max_min = [[0.81687535,	0.204062178,	0.999047619,1	,0.88627451,	1],
               [0.395667896,	0	,0	,0.749019608,	0.058823529,	0.458823529]]
    for I_ans_i in range(0, len(ans)):
        ans[I_ans_i] = (ans[I_ans_i]-max_min[1][I_ans_i])/(max_min[0][I_ans_i]-max_min[1][I_ans_i])
    return ans


if __name__ == '__main__':
    text_data = [[211 / 255, 224 / 255, 219 / 255], [55 / 255, 71 / 255, 57 / 255], [163 / 255, 184 / 255, 174 / 255],
                 [117 / 255, 134 / 255, 115 / 255], [23 / 255, 38 / 255, 35 / 255]]
    features = te_zen_be(text_data)
    ex, ey = ex_ey(features)
    print(ex, ey)
