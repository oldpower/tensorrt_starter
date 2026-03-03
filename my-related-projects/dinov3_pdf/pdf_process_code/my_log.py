# 定义颜色代码
class Colors:
    DGREEN = "\033[1;36m"
    BLUE = "\033[1;34m"
    PURPLE = "\033[1;35m"
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    CLEAR = "\033[0m"

# 定义日志级别
from enum import Enum, auto

class Level(Enum):
    ERROR = auto()
    INFO = auto()
    VERB = auto()

def log_info(level: Level, message: str, *args):
    """
    根据日志级别打印带颜色的日志信息
    :param level: 日志级别
    :param message: 日志信息
    :param args: 用于格式化的参数
    """
    if level == Level.INFO:
        prefix = f"{Colors.YELLOW}[info]: {Colors.CLEAR}"
    elif level == Level.VERB:
        prefix = f"{Colors.DGREEN}[verb]: {Colors.CLEAR}"
    else:  # 默认为错误级别
        prefix = f"{Colors.RED}[error]: {Colors.CLEAR}"

    formatted_message = message % args  # 使用%操作符进行简单的字符串格式化
    print(f"{prefix}{formatted_message}")

# 示例用法
if __name__ == "__main__":
    log_info(Level.INFO, "这是一个信息消息，参数: %s", "参数值")
    log_info(Level.VERB, "这是一个详细信息消息")
    log_info(Level.ERROR, "这是一个错误消息")


# import logging

# # 创建logger
# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.DEBUG)  # 设置最低的日志级别

# # 创建控制台处理器并设置格式
# console_handler = logging.StreamHandler()
# console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(console_format)

# # 创建文件处理器并设置格式
# file_handler = logging.FileHandler('app.log')
# file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(file_format)

# # 添加处理器到logger
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# # 使用logger记录日志
# logger.debug('这是控制台和文件都会记录的调试信息')
