import json
import sys
from octorest import OctoRest
import pandas as pd
import numpy as np
import threading
import requests
import os
import io
import time
from PIL import Image
from io import BytesIO
from datetime import datetime, timedelta
import csv
import torch
from torchvision import transforms
from collections import deque, Counter
from yxx_multi_model_attention import MultiViewNet
import ybw_config as Config
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# 使用的环境是当地环境，python3.10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 统一窗口长度，修改此处即可全局修改 window_size
WINDOW_SIZE = 20  # 统一设置窗口长度

def condition_is_met(gcode_executed=True):
    """检查条件是否满足（如打印机状态）"""
    return gcode_executed

class ParameterTracker:
    def __init__(self):
        self.last_values = {
            'flow_rate': 140,  # 初始默认值
            'feed_rate': 70   # 初始默认值
        }

    def update_value(self, param_name, new_value):
        self.last_values[param_name] = new_value

    def get_last_value(self, param_name):
        return self.last_values[param_name]
class PredictionRecorder:
    def __init__(self, data_file='print_data.csv', defect_file='defect_data.csv'):
        # 使用硬编码的初始参数
        initial_flow_rate = 140  # 固定值 130
        initial_feed_rate = 70   # 固定值 70

        self.timestamps = []
        self.predictions = []
        self.flow_rate_adjustments = []
        self.feed_rate_adjustments = []
        self.flow_rate_values = []  # 记录实时流量率
        self.feed_rate_values = []  # 记录实时进给率
        self.prediction_window = deque(maxlen=WINDOW_SIZE)

        # 初始化流量率和进给率的初始值
        initial_timestamp = datetime.now()
        self.timestamps.append(initial_timestamp)
        self.flow_rate_values.append((initial_timestamp, initial_flow_rate))
        self.feed_rate_values.append((initial_timestamp, initial_feed_rate))

        # 初始化文件路径
        self.data_file = data_file
        self.defect_file = defect_file

        # 创建并初始化 CSV 文件
        self.initialize_data_file()
        self.initialize_defect_file()

    def initialize_data_file(self):
        """初始化流量率和进给率的 CSV 文件，写入表头"""
        with open(self.data_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Flow Rate', 'Feed Rate'])

    def initialize_defect_file(self):
        """初始化缺陷类别的 CSV 文件，写入表头"""
        with open(self.defect_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Prediction', 'Under-extrusion Probability', 'Good-quality Probability', 'Over-extrusion Probability'])

    def save_data_to_file(self, timestamp, flow_rate, feed_rate):
        """将流量率和进给率数据保存到 CSV 文件"""
        with open(self.data_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, flow_rate, feed_rate])

    def save_defect_data_to_file(self, timestamp, prediction, under_prob, good_prob, over_prob):
        """将缺陷预测数据保存到 CSV 文件"""
        with open(self.defect_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, prediction, under_prob, good_prob, over_prob])

    def add_adjustment(self, param_name, adjustment, new_value):
        """记录流量率和进给率的调整"""
        timestamp = self.timestamps[-1] if self.timestamps else datetime.now()
        if param_name == 'flow_rate':
            self.flow_rate_adjustments.append((timestamp, adjustment))
            self.flow_rate_values.append((timestamp, new_value))
        elif param_name == 'feed_rate':
            self.feed_rate_adjustments.append((timestamp, adjustment))
            self.feed_rate_values.append((timestamp, new_value))

        # 保存调整后的流量率和进给率数据到文件
        self.save_data_to_file(timestamp, self.flow_rate_values[-1][1], self.feed_rate_values[-1][1])

    def add_prediction(self, timestamp, prediction):
        """添加预测结果"""
        self.timestamps.append(timestamp)
        self.predictions.append(prediction)
        self.prediction_window.append(prediction)

        # 保存当前的流量率和进给率数据
        self.save_data_to_file(timestamp, self.flow_rate_values[-1][1], self.feed_rate_values[-1][1])

    def calculate_probabilities(self):
        """计算每个时间窗口内各类别的概率"""
        probabilities = []
        for i in range(0, len(self.predictions), WINDOW_SIZE):
            window = self.predictions[i:i + WINDOW_SIZE]
            total = len(window)
            under_extrusion_prob = sum(1 for p in window if p in[0]) / total
            good_quality_prob = sum(1 for p in window if p in [1,2,3]) / total
            over_extrusion_prob = sum(1 for p in window if p in [4]) / total
            probabilities.append((under_extrusion_prob, good_quality_prob, over_extrusion_prob))
            # 打印当前时间窗口内的类别概率
            print(f"Window {i // WINDOW_SIZE + 1}: Under-extrusion: {under_extrusion_prob:.2f}, "
                  f"Good-quality: {good_quality_prob:.2f}, Over-extrusion: {over_extrusion_prob:.2f}")
            # 保存每个窗口的类别概率到文件
            timestamp = self.timestamps[i + WINDOW_SIZE - 1] if i + WINDOW_SIZE - 1 < len(self.timestamps) else self.timestamps[-1]
            prediction = self.predictions[i + WINDOW_SIZE - 1] if i + WINDOW_SIZE - 1 < len(self.predictions) else self.predictions[-1]
            self.save_defect_data_to_file(timestamp, prediction, under_extrusion_prob, good_quality_prob, over_extrusion_prob)

        return probabilities

    def plot_defect_states(self, save_path, initial_sample_count=30):
        """绘制缺陷类别状态图"""
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        # 确保时间段计算是基于实际的样本数量
        if len(self.predictions) >= WINDOW_SIZE:
            total_time = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            time_per_segment = total_time / (len(self.predictions) // WINDOW_SIZE)
        else:
            time_per_segment = 1  # 如果样本数量不足，设置默认时间段长度

        probabilities = self.calculate_probabilities()
        colors = {'under': 'blue', 'good': 'green', 'over': 'red'}
        labels = {'under': 'Under-extrusion', 'good': 'Good-quality', 'over': 'Over-extrusion'}

        # 绘制每段时间内的类别占比
        prev_under, prev_good, prev_over = 0, 0, 0
        for i, (under_prob, good_prob, over_prob) in enumerate(probabilities):
            start_time = i * time_per_segment
            end_time = (i + 1) * time_per_segment

            # 绘制三条横线，分别表示每个类别的占比
            ax1.plot([start_time, end_time], [under_prob] * 2, color=colors['under'], linewidth=3)
            ax1.plot([start_time, end_time], [good_prob] * 2, color=colors['good'], linewidth=3)
            ax1.plot([start_time, end_time], [over_prob] * 2, color=colors['over'], linewidth=3)

            # 绘制竖直虚线，连接上一时间段的类别
            if i > 0:
                ax1.plot([start_time, start_time], [prev_under, under_prob], color=colors['under'], linestyle='--',
                         linewidth=1)
                ax1.plot([start_time, start_time], [prev_good, good_prob], color=colors['good'], linestyle='--',
                         linewidth=1)
                ax1.plot([start_time, start_time], [prev_over, over_prob], color=colors['over'], linestyle='--',
                         linewidth=1)

            prev_under, prev_good, prev_over = under_prob, good_prob, over_prob

        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend([labels['under'], labels['good'], labels['over']], loc='upper right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Defect probabilities image saved to: {save_path}")

    def plot_rate_adjustments(self, save_path):
        """绘制流量率和进给率的实时值"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))  # 只创建一个子图

        start_time = self.timestamps[0]

        # 如果有记录的流量率和进给率值，绘制阶梯图
        if self.flow_rate_values and self.feed_rate_values:
            flow_rate_times, flow_rate_values = zip(*self.flow_rate_values)
            feed_rate_times, feed_rate_values = zip(*self.feed_rate_values)

            flow_rate_time_diffs = [(t - start_time).total_seconds() for t in flow_rate_times]
            feed_rate_time_diffs = [(t - start_time).total_seconds() for t in feed_rate_times]

            ax.step(flow_rate_time_diffs, flow_rate_values, color='#ff7f0e', linewidth=2,
                    label='Flow Rate (Real-time)', where='pre')
            ax.step(feed_rate_time_diffs, feed_rate_values, color='#9467bd', linewidth=2,
                    label='Feed Rate (Real-time)', where='pre')

        # 设置图形的其他属性
        ax.set_ylabel('Rate Value', fontsize=12)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        # 调整布局并保存图形
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Rate adjustments image saved to: {save_path}")


def process_images(model, image_queues, prediction_queue, client, recorder, parameter_tracker):
    """处理图像并根据预测进行参数调整"""
    prediction_lists = {'flow_rate': deque(maxlen=WINDOW_SIZE), 'feed_rate': deque(maxlen=WINDOW_SIZE)}
    last_adjustment_time = {'flow_rate': datetime.now(), 'feed_rate': datetime.now()}
    cooling_period = timedelta(seconds=5)  # 冷却期（6秒）
    # 定义多次缺陷注入的时间点和对应的缺陷参数
    # defect_injections = [
    #     {"time": datetime.now() + timedelta(seconds=50), "flow_rate": 160, "feed_rate": 50},  # 在30秒后注入第一个缺陷
    #     {"time": datetime.now() + timedelta(seconds=150), "flow_rate": 170, "feed_rate": 60},  # 在60秒后注入第二个缺陷
    #     {"time": datetime.now() + timedelta(seconds=250), "flow_rate": 30, "feed_rate": 150},  # 在90秒后注入第三个缺陷
    # ]
    # injected_defects = set()  # 用于跟踪已注入的缺陷

    while True:
        images = []
        for i, queue in enumerate(image_queues):
            image, angle = queue.get()
            transform = get_transforms(angle)
            image = transform(image)
            images.append(image)

        with torch.no_grad():
            input_images = [img.unsqueeze(0).to(device) for img in images]
            outputs = model(input_images)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        recorder.add_prediction(datetime.now(), prediction)

        # 将预测结果映射到flow_rate和feed_rate
        flow_rate_pred = 0 if prediction < 2 else (2 if prediction > 2 else 1)
        feed_rate_pred = 0 if prediction < 2 else (2 if prediction > 2 else 1)
        prediction_lists['flow_rate'].append(flow_rate_pred)
        prediction_lists['feed_rate'].append(feed_rate_pred)


        # 多次注入缺陷
        # for defect in defect_injections:
        #     if defect["time"] <= datetime.now() and defect["time"] not in injected_defects:
        #         # 注入缺陷：调节流量率和进给率到缺陷参数
        #         defect_flow_rate = defect["flow_rate"]
        #         defect_feed_rate = defect["feed_rate"]
        #         client.gcode(f"M221 S{defect_flow_rate}")  # 设置流量率
        #         client.gcode(f"M220 S{defect_feed_rate}")  # 设置进给率
        #         print(f"Injecting defect: Flow Rate set to {defect_flow_rate}%, Feed Rate set to {defect_feed_rate}%")
        #         # 将注入的缺陷参数写入CSV
        #         recorder.add_adjustment('flow_rate', defect_flow_rate - recorder.flow_rate_values[-1][1],
        #                                 defect_flow_rate)
        #         recorder.add_adjustment('feed_rate', defect_feed_rate - recorder.feed_rate_values[-1][1],
        #                                 defect_feed_rate)
        #         # 清空 prediction_lists，重新计数，避免注入缺陷后立即根据之前的样本调整参数
        #         prediction_lists['flow_rate'].clear()
        #         prediction_lists['feed_rate'].clear()
        #
        #         injected_defects.add(defect["time"])  # 记录已经注入的缺陷，防止重复注入
        # # 继续记录当前的参数值到CSV
        # recorder.save_data_to_file(datetime.now(), recorder.flow_rate_values[-1][1], recorder.feed_rate_values[-1][1])
        for param in ['flow_rate', 'feed_rate']:
            if len(prediction_lists[param]) == WINDOW_SIZE and (
                    datetime.now() - last_adjustment_time[param]) > cooling_period:
                adjust_parameter(param, list(prediction_lists[param]), client, recorder, prediction, parameter_tracker)
                prediction_lists[param].clear()
                last_adjustment_time[param] = datetime.now()
                print(f"Adjustment made for {param} at {datetime.now()}")

        prediction_queue.put(prediction)

def get_adjustment_factor(prediction):
    if prediction == 4: # 严重过度挤出
        return 1
    elif prediction in [0, 3]: # 严重挤出不足或轻微过度挤出
        return 0.6
    elif prediction == 1: # 轻微挤出不足
        return 0.20
    else: # 正常情况
        return 0.10

def adjust_parameter(param_name, predictions, client, recorder, last_prediction, parameter_tracker):
    """根据预测结果调整参数"""
    param_count = Counter(predictions)
    most_common_param, param_num = param_count.most_common(1)[0]
    param_frequency = param_num / len(predictions)

    thresholds = {'flow_rate': 0.5, 'feed_rate': 0.5}
    I_mins = {'flow_rate': 0.2, 'feed_rate': 0.4}
    A_pluses = {'flow_rate': 40, 'feed_rate': 40}
    A_subs = {'flow_rate': -50, 'feed_rate': -50}

    threshold = thresholds[param_name]
    adjustment_factor = get_adjustment_factor(last_prediction)

    if param_frequency >= threshold:
        if most_common_param == 0:
            base_adjustment = np.interp(param_frequency, [threshold, 1], [I_mins[param_name], 1]) * A_pluses[param_name]
            adjustment = base_adjustment * adjustment_factor
        elif most_common_param == 2:
            base_adjustment = np.interp(param_frequency, [threshold, 1], [I_mins[param_name], 1]) * A_subs[param_name]
            adjustment = base_adjustment * adjustment_factor
        else:
            print(f"No adjustment needed for {param_name}. Most common param: {most_common_param}")
            return

        adjustment = round(adjustment, 2)
        # 使用追踪器获取上一次的值
        current_value = parameter_tracker.get_last_value(param_name)
        new_value = max(min(current_value + adjustment, 150), 50)
        recorder.add_adjustment(param_name, adjustment, new_value)

        command = f"M221 S{new_value}" if param_name == 'flow_rate' else f"M220 S{new_value}"
        client.gcode(command)
        # 更新追踪器中的值
        parameter_tracker.update_value(param_name, new_value)
        print(f"Command '{command}' sent. Adjustment: {adjustment}%")
        with open(f'{param_name}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now(), str(new_value)])
    else:
        print(f"Frequency {param_frequency} below threshold {threshold} for {param_name}. No adjustment made.")

def getSnapshotStreamMjpeg(path, webcamUrl, image_name, camera_id):
    TIMEOUT = 5
    response = requests.get(webcamUrl, stream=True, timeout=TIMEOUT)
    if response.status_code != 200:
        raise Exception(f'Webcam {camera_id} Snapshot URL returned HTTP status code {response.status_code}')

    startTime = time.time()
    try:
        while True:
            if time.time() - startTime >= TIMEOUT:
                raise TimeoutError("Copy operation timed out")

            chunk = response.raw.read(1024)
            start = chunk.find(b'\xff\xd8')
            if start != -1:
                imageData = chunk[start:]
                while True:
                    chunk = response.raw.read(1024)
                    end = chunk.find(b'\xff\xd9')
                    if end != -1:
                        imageData += chunk[:end + 2]
                        break
                    else:
                        imageData += chunk
                break
    except TimeoutError:
        if os.path.isfile(path):
            os.remove(path)
        raise Exception('Webcam Stream took too long sending Data')

    imageBytes = io.BytesIO(imageData)
    image = Image.open(imageBytes)
    outputfilename = os.path.join(path, f"{camera_id}", f"{image_name}.jpg")
    image.save(outputfilename, 'JPEG', quality=100, subsampling=0)
    return image

def get_transforms(angle):                  # 不同的模型用对应的参数
    if angle == 45:
        MEAN = [0.3260, 0.3757, 0.4252]
        STD = [0.2702, 0.2806, 0.3142]
    elif angle == 90:
        MEAN = [0.3457, 0.3713, 0.4139]
        STD = [0.2488, 0.2535, 0.2835]
    else:  # besides
        MEAN = [0.3579, 0.3742, 0.3461]
        STD = [0.2763, 0.2757, 0.2873]

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

def crop_image(image, center_x, center_y, crop_size=320):
    left = center_x - crop_size // 2
    top = center_y - crop_size // 2
    right = center_x + crop_size // 2
    bottom = center_y + crop_size // 2
    return image.crop((left, top, right, bottom))


def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def create_csv_files():
    for param in ['flow_rate', 'feed_rate']:
        with open(f'{param}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", f"{param.capitalize()} Adjustment"])


def main():
    # 在主程序开始时初始化追踪器
    parameter_tracker = ParameterTracker()
    create_csv_files()
    print("Starting main function...")
    DATE = datetime.now().strftime("%Y%m%d%H%M%S") + "real_time_correction"

    url = 'http://192.168.124.15'
    apikey = 'FB0DA015FA9847D0BD2A1F3CE6728587'
    client = OctoRest(url=url, apikey=apikey)

    Image_path = os.path.join(r"E:\3D\ybw_data_3cam\my_train_feed_flow\my_picture", DATE)
    check_and_create_directory(Image_path)
    check_and_create_directory(os.path.join(Image_path, f"{DATE}-besides"))
    check_and_create_directory(os.path.join(Image_path, f"{DATE}-45"))
    check_and_create_directory(os.path.join(Image_path, f"{DATE}-90"))
    # CSV 文件保存路径
    data_file_path = os.path.join(Image_path, 'print_data.csv')
    defect_file_path = os.path.join(Image_path, 'defect_data.csv')
    print(f"CSV files will be saved at: {data_file_path} and {defect_file_path}")

    webcamUrls = [
        "http://192.168.124.15:8080/?action=stream",
        "http://192.168.124.15:8081/?action=stream",
        "http://192.168.124.15:8082/?action=stream",
    ]

    crop_centers = [
        (400.0, 270.0, 'besides'),
        (190.0, 160.0, 45),
        (380.0, 260.0, 90)
    ]

    print("Loading model...")
    model = MultiViewNet(num_classes=Config.num_classes, num_views=Config.num_views).to(device)
    checkpoint = torch.load(
        r'E:\3D\ybw_data_3cam\my_train_feed_flow\multi_results\multi_attention_20241020_191020\epoch=50-val_loss=0.00-val_acc=97.46.pth',
        map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")

    image_queues = [Queue() for _ in range(Config.num_views)]
    prediction_queue = Queue()
    # 初始化PredictionRecorder时传递硬编码的初始参数
    # 初始化PredictionRecorder时指定CSV文件保存路径
    recorder = PredictionRecorder(data_file=data_file_path, defect_file=defect_file_path)


    print("Creating image processing thread...")
    # 传递 initial_sample_count 而非 initial_print_duration
    image_process_thread = threading.Thread(target=process_images,
                                            args=(model, image_queues, prediction_queue, client, recorder, parameter_tracker
                                            ))
    image_process_thread.start()
    print("Image processing thread started.")

    print("等待打印机开始打印...")
    try:
        while True:
            if condition_is_met(client):
                print("打印机已开始打印。开始监控...")
                while True:
                    try:
                        for i, webcamUrl in enumerate(webcamUrls):
                            if i == 0:
                                camera_id = f"{DATE}-besides"
                            elif i == 1:
                                camera_id = f"{DATE}-45"
                            else:
                                camera_id = f"{DATE}-90"

                            image = getSnapshotStreamMjpeg(Image_path, webcamUrl, f"image-{i}", camera_id)

                            center_x, center_y, angle = crop_centers[i]
                            cropped_image = crop_image(image, center_x, center_y)

                            image_queues[i].put((cropped_image, angle))

                        if not prediction_queue.empty():
                            prediction = prediction_queue.get()
                            print(f"Current prediction: {prediction}")

                        time.sleep(0.5)

                    except Exception as e:
                        print(f"Error occurred: {e}")
                        raise

                    if not condition_is_met(client):
                        print("打印已停止。")
                        break
            else:
                print("Waiting for the printer to start printing...")
                time.sleep(5)
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序发生错误: {e}")
    finally:
        print("正在生成结果图...")
        save_defect_path = os.path.join(Image_path, "prediction_defect_states.png")
        save_rate_path = os.path.join(Image_path, "prediction_rate_adjustments.png")
        recorder.plot_defect_states(save_defect_path)
        recorder.plot_rate_adjustments(save_rate_path)
        print("程序结束")


if __name__ == "__main__":
    main()