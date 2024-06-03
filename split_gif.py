from PIL import Image
import os
import imageio

def split_gif_vertically(gif_path, n_parts):
    # 打开 GIF 文件
    gif = Image.open(gif_path)

    # 获取 GIF 的宽度和高度
    width, height = gif.size

    # 计算每一份的宽度
    part_width = width // n_parts

    # 初始化每一份的帧列表
    parts_frames = [[] for _ in range(n_parts)]

    # 遍历每一帧
    frame_index = 0
    while True:
        try:
            # 设置当前帧
            gif.seek(frame_index)

            # 遍历每一份
            for i in range(n_parts):
                # 计算切分的边界
                left = i * part_width
                right = (i + 1) * part_width if i < n_parts - 1 else width

                # 切分图像
                part = gif.crop((left, 0, right, height))

                # 将帧添加到对应的列表中
                parts_frames[i].append(part.copy())

            # 处理下一帧
            frame_index += 1

        except EOFError:
            # 没有更多的帧
            break

    # 保存每一份为 GIF
    for i, frames in enumerate(parts_frames):
        os.makedirs(os.path.splitext(gif_path)[0], exist_ok=True)
        output_gif_path = os.path.join(os.path.splitext(gif_path)[0], f"part_{i+1}.gif")
        imageio.mimsave(output_gif_path, frames, fps=25, loop=0)

# 示例：将 'input.gif' 竖直切分为 3 份
split_gif_vertically("assets/gifs/ldmk.gif", 5)