import json
import os
import csv
import time
from flask import Flask, render_template, request
from flask import send_from_directory
from werkzeug.utils import secure_filename
from imge_change import image_change
from using_sa import using_sa
from weighting import te_zen_be, ex_ey
from sequential_execution import change
import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
import math
from chiyun import chiyun

feature_list = [1, 1, 1, 1, 1, 1, 1]
load = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
L = 100
max_stay_counter = 100
data_Will_change = []
data_Will_change_by3_1 = []
data_Will_change_tmp_1 = []
data_Will_change_by3 = []
data_Will_change_tmp = []
now_image_is = 0
color_list = -1
dijihang = -1
app = Flask(__name__)
ex = 0
ey = 0
choose_model = 7
app.config['UPLOAD_FOLDER'] = 'static/'


def keshihua3(list_data_be1, new_m_name):
    print("xws")
    temp = []
    list_data_be = []
    for list_data_be_i in range(0, len(list_data_be1)):
        if (list_data_be_i + 1) % 3 == 0:
            temp.append(list_data_be1[list_data_be_i])
            list_data_be.append(temp)
            temp = []
        else:
            temp.append(list_data_be1[list_data_be_i])
    for list_data_be_i in range(0, len(list_data_be)):
        list_data_be[list_data_be_i][0] = list_data_be[list_data_be_i][0] / 255
        list_data_be[list_data_be_i][1] = list_data_be[list_data_be_i][1] / 255
        list_data_be[list_data_be_i][2] = list_data_be[list_data_be_i][2] / 255
    print(list_data_be)
    theta1 = np.linspace(0, 2 * np.pi, len(list_data_be) + 1)
    theta = []
    for i_thera in range(0, len(theta1) - 1):
        theta.append(theta1[i_thera])
    theta = np.array(theta)
    data = []
    for data_i in range(0, len(list_data_be)):
        data.append(300 + 100 * data_i)
    data = sorted(data, reverse=True)
    plt.figure(figsize=(100, 100))
    # 设置极坐标系
    ax = plt.axes(polar=True)  # 实例化极坐标系
    ax.set_theta_direction(-1)  # 顺时针为极坐标正方向
    ax.set_theta_zero_location('N')  # 极坐标 0° 方向为 N

    print(list_data_be)
    ax.bar(x=theta,  # 柱体的角度坐标
           height=data,  # 柱体的高度, 半径坐标
           width=6.28 / len(list_data_be),  # 柱体的宽度
           color=list_data_be
           )
    ax.bar(x=theta,  # 柱体的角度坐标
           height=130,  # 柱体的高度, 半径坐标
           width=6.28,  # 柱体的宽度
           color='white'
           )
    ax.set_axis_off()
    print("static/photo_center/" + str(new_m_name) + ".jpg")
    plt.savefig("static/photo_center/" + str(new_m_name) + ".jpg")
    plt.show()


def keshihua0(list_data_be1, new_m_name):
    temp = []
    list_data_be = []
    for list_data_be_i in range(0, len(list_data_be1)):
        if (list_data_be_i + 1) % 3 == 0:
            temp.append(list_data_be1[list_data_be_i])
            list_data_be.append(temp)
            temp = []
        else:
            temp.append(list_data_be1[list_data_be_i])
    print(list_data_be)
    for list_data_be_i in range(0, len(list_data_be)):
        list_data_be[list_data_be_i][0] = list_data_be[list_data_be_i][0] / 255
        list_data_be[list_data_be_i][1] = list_data_be[list_data_be_i][1] / 255
        list_data_be[list_data_be_i][2] = list_data_be[list_data_be_i][2] / 255
    plt.figure()
    ax = plt.gca()  # 获取当前绘图区的坐标系
    bmap = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45,
                   lon_0=-95)
    # 画省
    shp_info = bmap.readshapefile('gadm36_CHN_shp/gadm36_USA_1', 'states', drawbounds=False)
    data_amer = [['Wyoming', 576412], ['Vermont', 626011], ['District of Columbia', 632323], ['North Dakota', 699628],
                 ['Alaska', 731449], ['South Dakota', 833354], ['Delaware', 917092], ['Montana', 1005141],
                 ['Rhode Island', 1050292], ['New Hampshire', 1320718], ['Maine', 1329192], ['Hawaii', 1392313],
                 ['Idaho', 1595728], ['West Virginia', 1855413], ['Nebraska', 1855525], ['New Mexico', 2085538],
                 ['Nevada', 2758931], ['Utah', 2855287], ['Kansas', 2885905], ['Arkansas', 2949131],
                 ['Mississippi', 2984926], ['Iowa', 3074186], ['Connecticut', 3590347], ['Puerto Rico', 3667084],
                 ['Oklahoma', 3814820], ['Oregon', 3899353], ['Kentucky', 4380415], ['Louisiana', 4601893],
                 ['South Carolina', 4723723], ['Alabama', 4822023], ['Colorado', 5187582], ['Minnesota', 5379139],
                 ['Wisconsin', 5726398], ['Maryland', 5884563], ['Missouri', 6021988], ['Tennessee', 6456243],
                 ['Indiana', 6537334], ['Arizona', 6553255], ['Massachusetts', 6646144], ['Washington', 6897012],
                 ['Virginia', 8185867], ['New Jersey', 8864590], ['North Carolina', 9752073], ['Michigan', 9883360],
                 ['Georgia', 9919945], ['Ohio', 11544225], ['Pennsylvania', 12763536], ['Illinois', 12875255],
                 ['Florida', 19317568], ['New York', 19570261], ['Texas', 26059203], ['California', 38041430]]
    data_amer = [
        ["Texas",	695662],
        ["Arizona", 295234],
        ["Alaska",	433967],
        ["California",423967],

        ["New Mexico",314917],
        ["Montana", 380831],
        ["Nevada",286380],

        ["Oregon",254799],

        ["Wyoming",253335],
        ["Michigan",250487],
        ["Kansas", 213100],
        ["Minnesota",225163],
        ["Utah",219882],
        ["Colorado", 269601],
        ["Idaho",216443],

        ["Nebraska",200330],
        ["South Dakota",199729],
        ["Washington",	184661],
        ["North Dakota",	183108],
        ["Oklahoma",	181037],
        ["Missouri",	180540],
        ["Florida",	170312],
        ["Wisconsin",	169635],
        ["Georgia",	153910],
        ["Illinois",	149995],
        ["Iowa",	145746],
        ["New York",	141297],
        ["North Carolina",	139391],
        ["Arkansas",	137732],
        ["Alabama",	135767],
        ["Louisiana",	135659],
        ["Mississippi",	125438],
        ["Pennsylvania",	119280],
        ["Ohio",	116098],
        ["Virginia",	110787],
        ["Tennessee",	109153],
        ["Kentucky",	104656],
        ["Indiana",	94326],
        ["Maine",	91633],
        ["South Carolina",	82933],
        ["West Virginia",	62756],
        ["Maryland",	32131],
        ["Hawaii",	28313],
        ["Massachusetts",	27336],
        ["Vermont",	24906],
        ["New Hampshire",	24214],
        ["New Jersey",	22591],
        ["Connecticut",	14357],
        ["Delaware",	6446],
       ["Rhode Island",4001]

    ]
    number_of_one = math.ceil(50 / len(list_data_be))
    for info, shp in zip(bmap.states_info, bmap.states):
        proid = info['NAME_1']
        for i_ame in range(0, len(data_amer)):
            if proid == data_amer[i_ame][0] and proid == 'Alaska':
                shp1 = []
                for poly_i in range(0, len(shp)):
                    shp1.append((0.35 * shp[poly_i][0] + 900000, 0.35 * shp[poly_i][1] - 1300000))
                poly = Polygon(shp1, facecolor=list_data_be[math.floor(i_ame / number_of_one)])
                ax.add_patch(poly)
            if proid == data_amer[i_ame][0] and proid == 'Hawaii':
                shp1 = []
                for poly_i in range(0, len(shp)):
                    shp1.append((shp[poly_i][0] + 5100000, shp[poly_i][1] - 1400000))
                poly = Polygon(shp1, facecolor=list_data_be[math.floor(i_ame / number_of_one)])
                ax.add_patch(poly)
            if (len(list_data_be) == 9 or len(list_data_be) == 10) and (proid == "South Dakota" or proid == "North Dakota" or proid == "Missouri" or proid == "Iowa" or proid == "Utah" or proid == "South Carolina" or proid == "Ohio" or proid == "Alabama"):
                poly = Polygon(shp, facecolor=list_data_be[0])
                ax.add_patch(poly)
            if (len(list_data_be) == 9 or len(list_data_be) == 10) and proid == "Nebraska":
                poly = Polygon(shp, facecolor=list_data_be[1])
                ax.add_patch(poly)
            else:
                if proid == data_amer[i_ame][0]:
                    poly = Polygon(shp, facecolor=list_data_be[math.floor(i_ame / number_of_one)])
                    ax.add_patch(poly)
    bmap.drawmapboundary()
    plt.savefig("static/photo_center/" + str(new_m_name) + ".jpg")
    plt.show()


def keshihua2(list_data_be1, new_m_name):
    print("xws")
    temp = []
    list_data_be = []
    for list_data_be_i in range(0, len(list_data_be1)):
        if (list_data_be_i + 1) % 3 == 0:
            temp.append(list_data_be1[list_data_be_i])
            list_data_be.append(temp)
            temp = []
        else:
            temp.append(list_data_be1[list_data_be_i])
    print(list_data_be)
    for list_data_be_i in range(0, len(list_data_be)):
        list_data_be[list_data_be_i][0] = list_data_be[list_data_be_i][0] / 255
        list_data_be[list_data_be_i][1] = list_data_be[list_data_be_i][1] / 255
        list_data_be[list_data_be_i][2] = list_data_be[list_data_be_i][2] / 255
    plt.figure()
    data_choose = [[2, 1, 0, 2, 1, 1, 0, 0, 1, 1, 2, 0, 0, 0]
    ,[2, 1, 0, 3, 1, 1, 0, 0, 0, 1, 2, 0, 0, 3]
        , [2, 1, 0, 0, 2, 1, 0, 1, 2, 3, 4, 2, 4, 3]
        , [2, 1, 0, 3, 5, 1, 0, 5, 3, 2, 0, 2, 4, 0]
        , [2, 1, 0, 6, 5, 1, 0, 5, 2, 4, 3, 2, 3, 2]
        , [2, 1, 0, 3, 5, 1, 0, 6, 4, 3, 7, 2, 4, 5]
        , [2, 1, 0, 7, 5, 1, 0, 6, 8, 3, 2, 2, 4, 8]
        , [2, 1, 0, 7, 5, 1, 0, 6, 8, 8, 7, 2, 4, 9]]

    data_data = [875 / 100, 994 / 100, 997 / 100, 202 / 100, 353 / 100, 407 / 100, 812 / 100, 288 / 100, 174 / 100,
                 46 / 100, 109 / 100, 769 / 100, 456 / 100, 38 / 100]
    data_choose1 = data_choose[len(list_data_be) - 3]
    colors_bar = []
    plt.axis('off')
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    for data_choose1_i in range(0, len(data_choose1)):
        colors_bar.append(list_data_be[data_choose1[data_choose1_i]])
    plt.bar(x=x, height=data_data, color=colors_bar)
    print("static/photo_center/" + str(new_m_name) + ".jpg")
    plt.savefig("static/photo_center/" + str(new_m_name) + ".jpg")
    plt.show()


def wirting_image_of_color(data, name):
    fig = plt.figure(figsize=(len(data), 1))
    ax = fig.add_subplot()
    for data_i in range(0, len(data)):
        rect = plt.Rectangle((0.1 * data_i, 0), 0.1, 0.1,
                             color=(data[data_i][0] / 256, data[data_i][1] / 256, data[data_i][2] / 256))
        # print(data[data_i])
        ax.add_patch(rect)
    plt.xlim(0, 0.1 * len(data))
    plt.ylim(0, 0.1)
    plt.axis("off")
    # plt.savefig(savePath + str(name12) + '.png', dpi=600, bbox_inches='tight', pad_inches=-0.1)
    plt.savefig("static/palette_new/" + str(name) + ".jpg")
    plt.show()


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        print(request.files['no1'])
        f = request.files['no1']
        print(request.files)
        new_way = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
        f.save(new_way)
        return render_template('index.html', data=new_way)


@app.route('/')
def index():  #
    return render_template('index.html', data=0)


@app.route("/download/<filename>", methods=['GET'])
def weichat_png(filename):  # put application's code here
    directory = "/static/png"
    return send_from_directory(directory, filename, as_attachment=True)


@app.route('/load1', methods=['GET', 'POST'])
def load1():
    global load
    dlist = json.loads(request.get_data())
    shuo = dlist["shuo"]
    load[int(shuo)] = 1


@app.route('/new_deplete', methods=['GET', 'POST'])
def generate_result():
    global data_Will_change_by3, ex, ey, dijihang, color_list, data_Will_change, data_Will_change_by3_1, data_Will_change_tmp_1, data_Will_change_tmp, load
    load = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    image_name = time.time()
    dlist = json.loads(request.get_data())
    dijihang = dlist['number']  # 第几行
    color_list = dlist['number_of_list']
    print(color_list)
    print(dijihang)
    data_Will_change = []
    with open('palette_data/' + str(color_list) + '.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        num = 0
        for row in reader:
            if num == dijihang:
                for i in range(0, len(row)):
                    data_Will_change.append(int(row[i]))
            num = num + 1
    print(data_Will_change)
    data_Will_change_by3_1 = []
    data_Will_change_tmp_1 = []
    if data_Will_change_by3:
        data_Will_change_by3 = []
    data_Will_change_tmp = []
    for i_255 in range(0, len(data_Will_change)):
        if (i_255 + 1) % 3 == 0:
            data_Will_change_tmp.append(data_Will_change[i_255] / 255)
            data_Will_change_by3.append(data_Will_change_tmp)
            data_Will_change_tmp = []
            data_Will_change_tmp_1.append(data_Will_change[i_255])
            data_Will_change_by3_1.append(data_Will_change_tmp_1)
            data_Will_change_tmp_1 = []
        else:
            data_Will_change_tmp.append(data_Will_change[i_255] / 255)
            data_Will_change_tmp_1.append(data_Will_change[i_255])
    fea_true = te_zen_be(feature_list, data_Will_change_by3)
    ex, ey = ex_ey(choose_model, feature_list, fea_true)
    if now_image_is == 0:  # 现在选的是
        keshihua0(data_Will_change, image_name)
    if now_image_is == 1:  # 现在选的是
        chiyun(data_Will_change, image_name)
    if now_image_is == 2:  # 现在选的是
        keshihua2(data_Will_change, image_name)
    if now_image_is == 3:  # 现在选的是
        keshihua3(data_Will_change, image_name)
        # change("static/png/our_data/Painting/data.jpg", data_Will_change_by3_1, image_name)

    return [ex, ey, image_name, data_Will_change_by3_1]


@app.route('/new_ex_ey', methods=['GET', 'POST'])
def new_ex_ey():
    global ex, ey, L, max_stay_counter, feature_list, choose_model, load
    print("xws")
    image_name = time.time()
    dlist = json.loads(request.get_data())
    ex_new = dlist["ex_new"]
    ey_new = dlist["ey_new"]
    print(ex_new)
    print(ey_new)
    need_be = []
    print(data_Will_change_by3)
    for data_Will_change_by3_i in range(0, len(data_Will_change_by3)):
        need_be.append(data_Will_change_by3[data_Will_change_by3_i][0])
        need_be.append(data_Will_change_by3[data_Will_change_by3_i][1])
        need_be.append(data_Will_change_by3[data_Will_change_by3_i][2])
    best_x, best_y = using_sa(load, choose_model, feature_list, L, max_stay_counter, ex_new, ey_new, need_be)
    best_x_tmp = []
    best_x_ = []
    for i_255 in range(0, len(best_x)):
        if (i_255 + 1) % 3 == 0:
            best_x_tmp.append(best_x[i_255] * 255)
            best_x_.append(best_x_tmp)
            best_x_tmp = []
        else:
            best_x_tmp.append(best_x[i_255] * 255)
    ex = ex_new
    ey = ey_new
    print(best_x_)
    wirting_image_of_color(best_x_, image_name)
    best_xx = []
    for I_best_x_ in range(0, len(best_x_)):
        for I_best_x_x in range(0, len(best_x_[0])):
            best_xx.append(best_x_[I_best_x_][I_best_x_x])

    if now_image_is == 0:  # 现在选的是
        keshihua0(best_xx, image_name)
    if now_image_is == 1:  # 现在选的是
        chiyun(best_xx, image_name)
    if now_image_is == 2:  # 现在选的是
        keshihua2(best_xx, image_name)
    if now_image_is == 3:  # 现在选的是
        keshihua3(best_xx, image_name)
    return [ex, ey, image_name]


@app.route('/change_chan_shu', methods=['GET', 'POST'])
def change_chan_shu():
    global feature_list
    dlist = json.loads(request.get_data())
    print(dlist)
    xxx = dlist["number"]
    xxx = int(xxx)
    print(xxx)
    if feature_list[xxx] == 0:
        feature_list[xxx] = 1
        return [1]
    else:
        feature_list[xxx] = 0
        print(feature_list[xxx])
        return [0]


@app.route('/change_l', methods=['GET', 'POST'])
def change_l():
    global L
    dlist = json.loads(request.get_data())
    print(dlist)
    L = int(dlist["mySelect_L"])


@app.route('/change_model12', methods=['GET', 'POST'])
def change_model12():
    global choose_model
    dlist = json.loads(request.get_data())
    print(dlist)
    choose_model = dlist["change_model"]
    return [choose_model]


@app.route('/change_max_stay_counter', methods=['GET', 'POST'])
def change_max_stay_counter():
    global max_stay_counter
    dlist = json.loads(request.get_data())
    print(dlist)
    max_stay_counter = int(dlist["max_stay_counter"])


@app.route('/now_image_is12', methods=['GET', 'POST'])
def now_image_is12():
    global now_image_is
    dlist = json.loads(request.get_data())
    print(dlist)
    now_image_is = int(dlist["now_image_is1"])


if __name__ == '__main__':
    app.run()

# Important Feature Weight:<br>
#                   Lab_a_max:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<button type="submit"  id="data0" onclick="change_color1(0)" style="background: rgb(220,220,220);width: 20px; height: 20px; border-radius:50%;border: none"></button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                   Lab_a_std:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<button type="submit"  id="data1" onclick="change_color1(1)" style="background: rgb(220,220,220);width: 20px; height: 20px; border-radius:50%;border: none"></button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                   hsv_h_max:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<button type="submit"  id="data2" onclick="change_color1(2)" style="background: rgb(220,220,220);width: 20px; height: 20px; border-radius:50%;border: none"></button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                   hsv_v_max:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<button type="submit"  id="data3" onclick="change_color1(3)" style="background: rgb(220,220,220);width: 20px; height: 20px; border-radius:50%;border: none"></button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                   rgb_1_g&nbsp;&nbsp;:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<button type="submit"  id="data4" onclick="change_color1(4)" style="background: rgb(220,220,220);width: 20px; height: 20px; border-radius:50%;border: none"></button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                   rgb_2_g&nbsp;&nbsp;:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<button type="submit"  id="data5" onclick="change_color1(5)" style="background: rgb(220,220,220);width: 20px; height: 20px; border-radius:50%;border: none"></button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                   hsv_2_v&nbsp;&nbsp;:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<button type="submit"  id="data6" onclick="change_color1(6)" style="background: rgb(220,220,220);width: 20px; height: 20px; border-radius:50%;border: none"></button>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
#                   &nbsp;<br>
#                   &nbsp;<br>
