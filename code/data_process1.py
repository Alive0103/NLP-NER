import json


with open('data/result4.txt', 'r', encoding='utf-8') as file:
    # 读取文件内容
    data = file.read()

paragraphs = data.split('\n\n')

# 格式化每个段落
formatted_paragraphs = [{"content": paragraph.strip(), "result_list": [], "prompt": ""} for paragraph in paragraphs if paragraph.strip()]


# 保存到文件
with open('dataset/data4_1.txt', 'w', encoding='utf-8') as file:
    for paragraph in formatted_paragraphs:
        file.write(json.dumps(paragraph, ensure_ascii=False) + '\n')
        
"""with open('data/trains6.txt', 'r', encoding='utf-8') as f1, open('dataset/data3.txt') as f2:
    content1 = f1.read()
    content2 = f2.read()

# 将内容合并并写入新文件
with open('dataset/train4.txt', 'w', encoding='utf-8') as outfile:
    outfile.write(content1 + '\n' + content2)"""