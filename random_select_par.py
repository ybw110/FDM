import random
from datetime import datetime
import time

def generate_random_values(first_run=False):
    # 生成两个参数的随机值
    flowrate = round(random.uniform(50, 160), 2)       # flowrate 在 20 到 190 之间随机生成一个均匀分布的浮点数，并保留两位小数
    feed_rate = round(random.uniform(50, 160), 2)      # feed_rate 在 20 到 190 之间随机生成一个均匀分布的浮点数，并保留两位小数
    return flowrate, feed_rate


def generate_gcode(param1, param2):
    # 这个函数接受两个参数，并根据这两个参数构建对应的 G-code 指令列表。
    # 每个 G-code 指令分别对应 flowrate 和 feed_rate，包括 M221 (设置挤出机百分比) 和 M220 (设置速度因子百分比)。
    gcode_list = []

    # M221: Set extrusion percentage (flowrate)
    gcode_m221 = f"M221 S{param1:.2f}"
    gcode_list.append(gcode_m221)

    # M220: Set speed factor override percentage (feed_rate)
    gcode_m220 = f"M220 S{param2:.2f}"
    gcode_list.append(gcode_m220)

    return gcode_list

def random_select_para(first_run=False):
    # 生成随机参数值
    flowrate, feed_rate = generate_random_values(first_run)

    # 生成G-code指令列表
    gcode_list = generate_gcode(flowrate, feed_rate)

    print(f"Parameters updated: flowrate={flowrate}, feed_rate={feed_rate}")
    print(gcode_list)

    return gcode_list, flowrate, feed_rate


if __name__ == "__main__":
    random_select_para()