import json
import os
from tqdm import tqdm
from hityper.typeobject import TypeObject
import re
# 假设我们已经定义了 transform_sample_to_top 和其他必要的函数
# 这里假设这些函数已经在某个模块中被定义，我们从那里导入它们
# from your_module import transform_sample_to_top, match_type_for_cot

# 加载数据集
with open(os.path.join("./data", "./testset_transformed.json")) as f:
    testset_trans = json.load(f)

# 加载结果数据
with open("./NSTI_return_after_sts.json") as f:
    results = json.load(f)

# 初始化计数器
total_local_str = 0
correct_local_str = 0

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

# 遍历数据集
for key, value in tqdm(testset_trans.items()):
    parts = key.split('--')
    # 检查是否为local变量
    if parts[-1] == "arg":
        gttype = TypeObject.Str2Obj(value[1])  # 假设第二个元素是实际类型
        # 检查实际类型是否为str
        #print(testset_trans[key][2] == "user-defined")

#        if value[1] == 'bool':
        if testset_trans[key][2] == "user-defined":
            total_local_str += 1
            if key in results:
                # 处理结果数据，使用match_type_for_cot来获取预测
                predictions = [match_type_for_cot(pred) for pred in results[key] if match_type_for_cot(pred)]
                # 检查预测结果是否正确
                for pred in predictions:
                    if pred is not None:
                        predtype = TypeObject.Str2Obj(pred)
                        if TypeObject.isIdenticalSet(gttype, predtype):
                            correct_local_str += 1
                            break  # 只计算一次正确预测

# 计算正确率
if total_local_str > 0:
    accuracy = (correct_local_str / total_local_str) * 100
else:
    accuracy = 0

print(f"Total local 'str' variables: {total_local_str}")
print(f"Correctly identified local 'str' variables: {correct_local_str}")
print(f"Accuracy: {accuracy:.2f}%")