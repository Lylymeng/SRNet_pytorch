from PIL import Image
import numpy as np

def pad_image_to_square(input_image, fill_color=200):

    if isinstance(input_image, Image.Image):
        width, height = input_image.size
    # 如果输入是numpy数组，获取其shape
    elif isinstance(input_image, np.ndarray):
        height, width = input_image.shape[:2]
        input_image = Image.fromarray(input_image)

    # 计算长边的边长
    max_side_length = max(width, height)

    # 创建一个新的方形图像并用背景颜色填充
    new_image = Image.new("L", (max_side_length, max_side_length), fill_color)

    # 计算将原始图像放置在中心位置时的左上角坐标
    upper_left = (max_side_length - width) // 2
    upper_top = (max_side_length - height) // 2
    lower_right = upper_left + width
    lower_bottom = upper_top + height
    box = (upper_left, upper_top, lower_right, lower_bottom)
    # 将原始图像粘贴到新图像上
    new_image.paste(input_image, box)

    return new_image