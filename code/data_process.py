import json


with open('data/trains6.txt', 'r', encoding='utf-8') as file:
    # 读取文件内容
    data = file.read()

# 将文件内容分割成单独的JSON字符串
json_strings = data.strip().split('\n')

# 遍历每个JSON字符串，解析并提取'content'字段
contents = []
for json_str in json_strings:
    if json_str:  # 确保不处理空字符串
        # 解析JSON数据
        data = json.loads(json_str)
        # 提取'content'字段并添加到列表中
        contents.append(data['content'])


# 如果需要，将提取的内容保存到新的文本文件中
with open('data/data6.txt', 'w', encoding='utf-8') as f:
    for content in contents:
        f.write(content + '\n')