import json

def load_data_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)  # 从文件中加载数据
    return data