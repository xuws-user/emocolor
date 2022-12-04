from sequential_execution import change
from using_sa import using_sa

need_be = [99 / 255, 34 / 255, 22 / 255, 63 / 255, 27 / 255, 22 / 255, 147 / 255, 24 / 255, 13 / 255, 130 / 255,
           62 / 255, 17 / 255, 187 / 255, 4 / 255, 4 / 255]


def image_change(now_image_is, ex0, ey0, image_name):
    best_x, best_y = using_sa(ex0, ey0, need_be)
    tempp = []
    data = []
    for i_best_x in range(0, len(best_x)):
        if (i_best_x + 1) % 3 == 0:
            tempp.append(best_x[i_best_x] * 256)
            data.append(tempp)
            tempp = []
        else:
            tempp.append(best_x[i_best_x] * 256)
    if now_image_is == 0:
        change("static/png/our_data/Painting/data.jpg", data, image_name)  # 放一个固定位置

    print('change_over')
