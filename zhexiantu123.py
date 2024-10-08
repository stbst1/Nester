import json
import os
from tqdm import tqdm
from hityper.typeobject import TypeObject
import re

# 加载数据集
with open(os.path.join("./data", "./testset_transformed.json")) as f:
    testset_trans = json.load(f)

# 加载结果数据
with open("./NSTI_return_after_sts.json") as f:
    results = json.load(f)

# 定义正则表达式匹配函数
def match_type_for_cot(string):
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    matched = re.findall(pattern, string)
    if not matched:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if not second_matched:
            return None
        return second_matched[-1].replace("`", "").replace('NoneType', 'None')
    return matched[-1].replace("`", "").replace('NoneType', 'None')

# 初始化统计字典
stats = {
    'local': {typ: {'total': 0, 'correct': 0} for typ in ['int', 'float', 'bool', 'str', 'bytes', 'list', 'tuple', 'dict']},
    'arg': {typ: {'total': 0, 'correct': 0} for typ in ['int', 'float', 'bool', 'str', 'bytes', 'list', 'tuple', 'dict']},
    'return': {typ: {'total': 0, 'correct': 0} for typ in ['int', 'float', 'bool', 'str', 'bytes', 'list', 'tuple', 'dict']}
}

# 遍历数据集
for key, value in tqdm(testset_trans.items()):
    parts = key.split('--')
    last_part = parts[-1]
    if last_part in stats and value[1] in stats[last_part]:
        stats[last_part][value[1]]['total'] += 1
        if key in results:
            predictions = [match_type_for_cot(pred) for pred in results[key] if match_type_for_cot(pred)]
            gttype = TypeObject.Str2Obj(value[1])
            for pred in predictions:
                if pred:
                    predtype = TypeObject.Str2Obj(pred)
                    if TypeObject.isIdenticalSet(gttype, predtype):
                        stats[last_part][value[1]]['correct'] += 1
                        break  # 只计算第一次正确预测

# 打印正确率
for context, types in stats.items():
    print(f"Context: {context}")
    for type_name, data in types.items():
        if data['total'] > 0:
            accuracy = (data['correct'] / data['total']) * 100
        else:
            accuracy = 0
        print(f"  {type_name}: {accuracy:.2f}% accuracy ({data['correct']}/{data['total']})")
