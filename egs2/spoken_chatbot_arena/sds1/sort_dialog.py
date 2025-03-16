import re
from operator import itemgetter

def sort_and_filter_dialog(input_file, output_file, dialog_id="sw02001"):
    # 读取输入文件
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 提取指定对话ID的行
    dialog_lines = []
    pattern = rf"{dialog_id}-[AB]_\d+-\d+"

    for line in lines:
        if re.match(pattern, line):
            # 提取时间戳的开始和结束时间
            timestamp_start, timestamp_end = map(
                int, line.split(" ")[0].split("_")[1].split("-")
            )
            # 提取说话者（A或B）
            speaker = line[len(dialog_id) + 1 : len(dialog_id) + 2]
            dialog_lines.append((timestamp_start, timestamp_end, speaker, line))

    # 按时间戳排序
    dialog_lines.sort(key=itemgetter(0))

    # 过滤被包含的对话
    filtered_dialogs = []
    filtered_dialogs.append(dialog_lines[0])
    for i in range(1, len(dialog_lines) - 1):
        start_current, end_current, speaker, current_line = dialog_lines[i]
        start_prev, end_prev, _, _ = dialog_lines[i - 1]
        start_next, end_next, _, _ = dialog_lines[i + 1]

        # 检查当前对话是否被前一个或后一个对话包含
        if not (start_prev <= start_current and end_prev >= end_current) and not (
            start_next <= start_current and end_next >= end_current
        ):
            filtered_dialogs.append((start_current, end_current, speaker, current_line))

    # 写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        for _, _, _, line in filtered_dialogs:
            f.write(line)
    print(f"对话 {dialog_id} 已排序并过滤，并保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    sort_and_filter_dialog("text", "sorted_dialog_02001.txt",dialog_id="sw02001")
    sort_and_filter_dialog("text", "sorted_dialog_02005.txt",dialog_id="sw02005")
    sort_and_filter_dialog("text", "sorted_dialog_02006.txt",dialog_id="sw02006")
    sort_and_filter_dialog("text", "sorted_dialog_02007.txt",dialog_id="sw02007")
    sort_and_filter_dialog("text", "sorted_dialog_02008.txt",dialog_id="sw02008")
    sort_and_filter_dialog("text", "sorted_dialog_02009.txt",dialog_id="sw02009")
    sort_and_filter_dialog("text", "sorted_dialog_02010.txt",dialog_id="sw02010")
    sort_and_filter_dialog("text", "sorted_dialog_02012.txt",dialog_id="sw02012")
    sort_and_filter_dialog("text", "sorted_dialog_02013.txt",dialog_id="sw02013")
    sort_and_filter_dialog("text", "sorted_dialog_02014.txt",dialog_id="sw02014")
    


