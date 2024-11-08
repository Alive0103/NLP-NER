import json

def text_to_json(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]  # 读取并去除空白行

    # 准备JSON对象的数据结构
       

    # 遍历每行文本，将其添加到result_list中
    json_data=[]
    count=0
    for line in lines:
        count=count+1
        json_data.append({
            "text": line ,
            "entities":[],
            "sample_id":count
        })

    # 将JSON对象写入到文件中
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(json_data, output_file, ensure_ascii=False, indent=4)
        
input_path = 'data/result6.txt' 
output_path = 'dataset/predict6.json'  
text_to_json(input_path, output_path)