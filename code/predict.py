# -*- coding: utf-8 -*-
from uie_predictor import UIEPredictor
from pprint import pprint
import time
import json
import argparse
# 1.装甲车辆；2.火炮；3.导弹；4.舰船舰艇；5.炸弹；6.太空装备



def parse_data(file_path, current_task, test_task, task_name, write_name):
    data = []
    #with open("../raw_data/{}".format(file_path), "r", encoding="utf-8") as f:
    with open(file_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    oo = open(write_name, "w", encoding="utf-8")

    for i, one in enumerate(test_data):
        try:
            text = one.get("text")
            sample_id = one.get("sample_id")
            res = ie(text)

            print(i)
            res_iter = res[0].get(task_name)
            for tt in res_iter:

                p = {
                    "current_task_id": current_task,
                    "test_task_id": test_task,
                    "sample_id": sample_id,
                    "text": "辽宁号",
                    "start": 20,
                    "end": 23,
                    "type": task_name
                }
                p["text"] = tt.get("text")
                p["start"] = tt.get("start")
                p["end"] = tt.get("end")
                data.append(p)
        except Exception as e:
            continue
    oo.write(json.dumps(data, ensure_ascii=False, indent=4))
# 1.装甲车辆；2.火炮；3.导弹；4.舰船舰艇；5.炸弹；6.太空装备

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file_path", default="../raw_data/1_ner_test.json", type=str,help="预测的原始文本路径")
    parser.add_argument("-t", "--task_name", default=5, type=int,
                        help="""任务的prompt index of list: ["装甲车辆", "火炮", "导弹", "舰船舰艇", "炸弹", "太空装备"]""")
    parser.add_argument("-w", "--write_name", default="./data/res_data1/1_ner_results.json", type=str, help="任务保存的路径")
    parser.add_argument("--current_task", type=int, default=6,
                        help="当前任务id")
    parser.add_argument("--test_task", type=int, default=6,
                        help="之前的任务id")
    parser.add_argument("-m","--model_path", default="../data/user_data/saved_model/model1/model_best", type=str,
                        help="训练好的模型路径，均用model_best下的权重")

    args = parser.parse_args()
    # yapf: enable
    # parse_data("1_ner_test.json",1,1,"装甲车辆","./data/res_data1/1_ner_results.json")
    pp = ["装甲车辆", "火炮", "导弹", "舰船舰艇", "炸弹", "太空装备"]
    schema = [pp[args.task_name]]
    ie = UIEPredictor(args.model_path, schema=schema, device="gpu")
    parse_data(args.file_path, args.current_task,args.test_task, pp[args.task_name], args.write_name)
