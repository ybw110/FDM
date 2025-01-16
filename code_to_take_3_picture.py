import json
import sys
from octorest import OctoRest
import pandas
import numpy
import threading
import requests
import os
import io
import time
from PIL import Image
from io import BytesIO
from datetime import datetime
import csv

"""
我在这里修改了，我没有使用随机选参数那个模块，就只是直接打印获取数据集
其他包括：移速，流速，手动输入——因为好像调不出来
        加入了热床的温度——我觉得对于翘曲，这个是一个因素，如果做侧面监督会有需要，所以我先考量
        要注意的是，之后的jpg，analysis代码也要修改对应的
"""

def getSnapshotStreamMjpeg(path, webcamUrl, image_name, camera_id):
    # path是保存图像文件的目录路径。webcamUrl是OctoPrint网络摄像头的URL地址。
    TIMEOUT = 5
    response = requests.get(webcamUrl, stream=True, timeout=TIMEOUT)
    # 从webcamUrl请求网络摄像头流数据,并将stream=True设置为流式传输模式,同时设置请求超时时间为TIMEOUT
    if response.status_code != 200:
        raise Exception('Webcam Snapshot URL returned HTTP status code ' + str(response.status_code))

    startTime = time.time()   #记录开始时间startTime = time.time()
    try:                      #使用try-except块来捕获可能发生的TimeoutError异常
        while True:
            if time.time() - startTime >= TIMEOUT:
                raise TimeoutError("Copy operation timed out")

            chunk = response.raw.read(1024) # 从响应流response.raw中读取1024字节的数据块chunk
            start = chunk.find(b'\xff\xd8') # 在读取的数据中查找字节串b'\xff\xd8'，这是JPEG图像的开始标志
            if start != -1:                 # 找到了就不是-1
                imageData = chunk[start:]   #如果找到了开始标志,则将数据块从开始标志处截断,赋值给imageData
                while True:
                    chunk = response.raw.read(1024) #继续从响应流中读取1024字节的数据块chunk
                    end = chunk.find(b'\xff\xd9')   # 读取更多的数据，直到找到字节串b'\xff\xd9'，这是JPEG图像的结束标志。
                    if end != -1:                   # 找到了就不是-1
                        imageData += chunk[:end + 2]
                        break                       #将包含结束标志的数据块添加到imageData中,然后使用break退出内层循环
                    else:
                        imageData += chunk
                break
    except TimeoutError:           # 设置了最长等待图像数据的时间，保证实时性
        if os.path.isfile(path):
            os.remove(path)        #检查path参数指定的文件是否存在,如果存在则删除它
        raise Exception('Webcam Stream took too long sending Data')

    imageBytes = io.BytesIO(imageData) # 如果成功从响应流中读取了完整的图像数据imageData。
                                       # 使用io.BytesIO(imageData)将图像数据转换为一个类文件对象imageBytes
    image = Image.open(imageBytes)     # 使用Image.open(imageBytes)打开这个类文件对象,将其作为PIL图像对象image
    # timestape = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # 获取当前时间戳timestape,使用datetime.now().strftime("%Y%m%d_%H%M%S_%f")格式化为字符串
    # outputfilename = os.path.join(path, f"{image_name}.jpg") # 文件名和文件所保存的文件夹是不同的！！！
    outputfilename = os.path.join(path, f"{camera_id}", f"{image_name}.jpg")
    image.save(outputfilename, 'JPEG', quality=100, subsampling=0) # 将PIL图像对象image保存为JPEG格式的文件,文件质量设置为最高(100),不进行子采样

#  函数的作用是从OctoPrint的网络摄像头获取MJPEG流数据,从中提取出完整的JPEG图像数据,并将其保存为文件,文件名包含时间戳和传入的参数信息


def check_condition(client): # 返回True/False,代表当前打印机的打印状态

    try:
        # 获取打印机的当前状态
        printer_status = client.printer()
        is_printing = printer_status['state']['flags']['printing']
        return is_printing
    except Exception as e:
        print(f"Error occurred while checking printer status: {e}")
        return False


def check_and_create_directory(directory_path): # 图像文件所要保存的路径

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def save_to_csv(imgae_name, img_num, timestape,config_Z_offset, actual_Z_offset, Z_offset,
                flowrate,feed_rate,hotend_T, actual_hotend_value,bed_T, actual_bed_value, directory):
    # 函数接受多个参数,包括图像文件名、图像编号、时间戳、Z 偏移量、流量率、进给率、目标热端温度、实际热端温度以及要保存 CSV 文件的目录路径
    directory = directory
    # 函数检查给定的目录是否存在,如果不存在则创建该目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    csv_file = os.path.join(directory, "data.csv")  #构建 CSV 文件的完整路径,将目录路径与文件名 "data.csv" 连接
    fieldnames = ["img_path", "img_num", "timestape", "config_z_offset", "actual_z_offset", "z_offset", "flow_rate",
                  "feed_rate", "target_hotend_T", 'actual_hotend_value', "target_bed_T", 'actual_bed_value']
    # 以追加模式打开 CSV 文件,并创建一个 csv.DictWriter 对象 writer
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if file.tell() == 0:
            writer.writeheader()  #如果文件为空,则使用 writer.writeheader() 写入字段名

        writer.writerow({
            "img_path": str(imgae_name),
            "img_num": str(img_num),
            "timestape": str(timestape),
            "config_z_offset": str(config_Z_offset),
            "actual_z_offset": str(actual_Z_offset),
            "z_offset": str(Z_offset),
            "flow_rate": str(flowrate),
            "feed_rate": str(feed_rate),
            "target_hotend_T": str(hotend_T),
            'actual_hotend_value': str(actual_hotend_value),
            "target_bed_T": str(bed_T),
            'actual_bed_value': str(actual_bed_value)
        })
def capture_image(path, webcamUrl, image_name, camera_id, error_list):
    try:
        getSnapshotStreamMjpeg(path, webcamUrl, image_name, camera_id)
    except Exception as e:
        error_list.append(f"Error capturing image from {camera_id}: {e}")


def main():
    DATE = datetime.now().strftime("%Y%m%d%H%M%S") + "dataset_else=print-02"
    print("octorest has been imported!")

    url = 'http://192.168.124.15'
    apikey = 'FB0DA015FA9847D0BD2A1F3CE6728587'
    client = OctoRest(url=url, apikey=apikey)

    Image_path = os.path.join(r"E:\3D\ybw_data_3cam\my_picture_flow_feed", DATE)
    check_and_create_directory(Image_path)
    check_and_create_directory(os.path.join(Image_path, f"{DATE}-besides"))
    check_and_create_directory(os.path.join(Image_path, f"{DATE}-45"))
    check_and_create_directory(os.path.join(Image_path, f"{DATE}-90"))

    max_photos = 500
    time_interval = 0.5
    webcamUrls = [
        "http://192.168.124.15:8080/?action=stream",
        "http://192.168.124.15:8081/?action=stream",
        "http://192.168.124.15:8082/?action=stream",
    ]

    photo_num = 0
    total_num = 0
    # 手动输入Feed Rate和Flow Rate
    flowrate = float(input("请输入Flow Rate (%): "))
    feed_rate = float(input("请输入Feed Rate (%): "))
    print("等待打印机开始打印...")
    while True:
        if check_condition(client):
            print("打印机已开始打印。开始收集数据...")  # 新增的提示语句
            while photo_num < max_photos:
                try:
                    last_snapshot_time = time.time()

                    photo_num += 1
                    total_num += 1
                    image_name = "image-" + str(total_num)

                    threads = []
                    error_list = []
                    for i, webcamUrl in enumerate(webcamUrls):
                        if i == 0:
                            camera_id = f"{DATE}-besides"
                        elif i == 1:
                            camera_id = f"{DATE}-45"
                        else:
                            camera_id = f"{DATE}-90"
                        thread = threading.Thread(target=capture_image,
                                                  args=(Image_path, webcamUrl, image_name, camera_id, error_list))
                        thread.start()
                        threads.append(thread)

                    for thread in threads:
                        thread.join()

                    if error_list:
                        for error in error_list:
                            print(error)
                        print("Stopping the program due to camera failure.")
                        sys.exit(1)



                    timestape = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    config_Z_offset = -3.9
                    Z_offset = 0  # 假设Z_offset为0，因为我们不再修改参数
                    actual_Z_offset = config_Z_offset + Z_offset

                    hotend_dict = client.tool()                           # 从客户端获取当前的热端温度数据 hotend_dict
                    actual_hotend_value = hotend_dict['tool0']['actual']  # 提取actual_hotend的值
                    target_hotend_value = hotend_dict['tool0']['target']  # 提取target_hotend的值

                    bed_dict = client.bed()                                 # 获取热床温度数据
                    actual_bed_value = bed_dict['bed']['actual']            # 当前热床温度
                    target_bed_value = bed_dict['bed']['target']            # 目标热床温度

                    save_to_csv(f"image-{total_num}", str(total_num), timestape, str(config_Z_offset),
                                str(actual_Z_offset), str(Z_offset), str(flowrate), str(feed_rate),
                                str(target_hotend_value),
                                str(actual_hotend_value), str(target_bed_value), str(actual_bed_value), Image_path)

                    after_snapshot_time = time.time()
                    operation_time = after_snapshot_time - last_snapshot_time
                    actual_wait_time = max(time_interval - operation_time, 0)
                    time.sleep(actual_wait_time)

                    if photo_num == max_photos:
                        print("Data collection completed.")
                        return

                except Exception as e:
                    print(f"Error occurred while taking photo: {e}")
                    print("Stopping the program.")
                    sys.exit(1)
        else:
            print("Waiting for the printer to start printing...")
            time.sleep(5)

if __name__ == "__main__":
    main()