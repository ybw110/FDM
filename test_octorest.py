from octorest import OctoRest


def get_current_parameter(client, param_name):
    """通过 G-code 从 OctoPrint 获取当前流量率或进给率"""
    try:
        if param_name == 'flow_rate':
            # 发送 M221 G-code 命令来查询流量率
            response = client.gcode("M221")
            flow_rate = parse_gcode_response(response)  # 解析 G-code 响应获取流量率
            print(f"Flow Rate from printer: {flow_rate}")
            return flow_rate
        elif param_name == 'feed_rate':
            # 发送 M220 G-code 命令来查询进给率
            response = client.gcode("M220")
            feed_rate = parse_gcode_response(response)  # 解析 G-code 响应获取进给率
            print(f"Feed Rate from printer: {feed_rate}")
            return feed_rate
        else:
            raise ValueError(f"Unsupported parameter: {param_name}")
    except Exception as e:
        print(f"Error occurred while getting {param_name}: {e}")
        return 100  # 如果出错，返回默认值100%

def parse_gcode_response(response):
    """解析 G-code 响应，提取数值"""
    try:
        # 假设 G-code 响应是类似 "Flow: 82%" 或 "Feed: 83%" 这类的字符串
        # 我们可以从响应中提取相应的数值
        if isinstance(response, str):
            percentage = response.split(":")[-1].strip().replace('%', '')
            return int(percentage)
        return 100  # 默认返回100
    except Exception as e:
        print(f"Error while parsing G-code response: {e}")
        return 100


def main():
    # OctoPrint服务器的URL和API密钥
    url = 'http://192.168.124.15'
    apikey = 'FB0DA015FA9847D0BD2A1F3CE6728587'

    # 初始化OctoRest客户端
    client = OctoRest(url=url, apikey=apikey)

    # 验证获取流量率和进给率
    print("Checking current Flow Rate...")
    current_flow_rate = get_current_parameter(client, 'flow_rate')

    print("Checking current Feed Rate...")
    current_feed_rate = get_current_parameter(client, 'feed_rate')

    # 输出当前的流量率和进给率
    print(f"Current Flow Rate: {current_flow_rate}")
    print(f"Current Feed Rate: {current_feed_rate}")


if __name__ == "__main__":
    main()