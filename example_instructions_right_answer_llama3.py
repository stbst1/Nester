# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import ast

from llama import Llama
import json
import re
import json
import os
from tqdm import tqdm

from collections import Counter


def filter_list(lst, keywords):
    return [item for item in lst if item in keywords]


def most_common_element(lst):
    # 使用 Counter 对列表中的元素进行计数
    counts = Counter(lst)
    # 找到出现次数最多的元素及其出现次数
    most_common = counts.most_common(1)
    # 返回出现次数最多的元素
    return most_common[0][0]


def extract_string_between_last_two_quotes(input_string):
    # 找到最后一个单引号的索引位置
    last_quote_index = input_string.rfind("`")

    # 找到倒数第二个单引号的索引位置
    second_last_quote_index = input_string.rfind("`", 0, last_quote_index)

    # 提取最后两个单引号之间的字符串
    if second_last_quote_index != -1 and last_quote_index != -1:
        extracted_string = input_string[second_last_quote_index + 1:last_quote_index]
        return extracted_string
    else:
        return "未找到足够的单引号"


def get_line_by_number(input_string, line_number):
    # 按行拆分字符串
    lines = input_string.split('\n')

    # 检查行号是否在有效范围内
    if 1 <= line_number <= len(lines):
        # 返回对应行的字符串
        return lines[line_number - 1]
    else:
        # 行号超出范围时返回空字符串或者抛出异常，取决于具体需求
        return "Invalid Line Number!"


def match_type_for_cot(string):
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
    # print(string)
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return None
        else:
            res = second_matched[-1].replace("`", "").replace('NoneType', 'None')  # .replace("is ", "")
            if (" " in res and "[" not in res) or res.lower() == "unknown":
                res = None
            return res
    else:
        res = matched[-1].replace("`", "").replace('NoneType', 'None')  # .replace("is ", "")
        if (" " in res and "[" not in res) or res.lower() == "unknown":
            res = None
        return res


def extract_outermost_brackets(input_string):
    stack = []
    outer_part = ""
    inner_part = ""

    for char in input_string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
                if not stack:
                    # 如果栈为空，说明当前右括号是最外层的右括号，不加入inner_part
                    continue

        if stack:
            inner_part += char
        else:
            outer_part += char

    return outer_part.strip(), inner_part.strip()


def extract_outermost_brackets_for_list(input_string):
    stack = []
    outer_part = ""
    inner_part = ""

    for char in input_string:
        if char == '[':
            stack.append(char)
        elif char == ']':
            if stack:
                stack.pop()
                if not stack:
                    # 如果栈为空，说明当前右括号是最外层的右括号，不加入inner_part
                    continue

        if stack:
            inner_part += char
        else:
            outer_part += char

    return outer_part.strip(), inner_part.strip()


def extract_parameters_from_method(method_declaration):
    pattern = r'\w+\s*\(([^)]*)\)'
    match = re.search(pattern, method_declaration)

    if match:
        parameters = [param.strip() for param in match.group(1).split(',') if param.strip()]
        return parameters
    else:
        return None


class IntraProceduralAnalysis:
    def __init__(self):
        # 控制流图
        self.control_flow_graph = {}
        # 数据流分析结果
        self.data_flow_analysis = {}

    def analyze_control_flow(self, code_lines):
        # 构建控制流图
        current_id = 1
        for line in code_lines:
            if 'if' in line:
                # 处理条件语句
                self.control_flow_graph[current_id] = [current_id + 1, current_id + 2]
                current_id += 2
            else:
                # 处理顺序执行语句
                self.control_flow_graph[current_id] = [current_id + 1]
                current_id += 1

    def analyze_data_flow(self, code_lines):
        # 数据流分析
        current_id = 1
        for line in code_lines:
            if 'def ' in line:
                parameters = extract_parameters_from_method(line)
                for item in parameters:
                    if item not in self.data_flow_analysis:
                        self.data_flow_analysis[item] = set()
                    self.data_flow_analysis[item].add(current_id)
            if ' = ' in line:
                # 处理赋值语句
                parts = line.split('=')
                variable = parts[0].strip()
                if variable not in self.data_flow_analysis:
                    self.data_flow_analysis[variable] = set()
                # 记录变量的定义点
                self.data_flow_analysis[variable].add(current_id)
            elif 'print' in line:
                # 处理打印语句
                parts = line.split('(')
                variable = parts[1].split(')')[0].strip()
                if variable not in self.data_flow_analysis:
                    self.data_flow_analysis[variable] = set()
                # 记录变量的使用点
                self.data_flow_analysis[variable].add(current_id)
            elif 'return' in line:
                if 'return' not in self.data_flow_analysis:
                    self.data_flow_analysis['return'] = set()
                pattern = r'return\s+(.*)'
                match = re.search(pattern, line)
                # print(match.group(1))
                # print(self.data_flow_analysis[match.group(1)])
                if match:
                    if match.group(1) == "None":
                        pattern = r'.*'
                        man = re.search(pattern, 'None')
                        self.data_flow_analysis['return'].add(man.group(0))
                        current_id += 1
                        continue;
                    if match.group(1) in self.data_flow_analysis:
                        self.data_flow_analysis['return'].add(match.group(1))
                        current_id += 1
                        continue;
                    blanket = 0
                    if '(' in match.group(1):
                        result_outer, result_inner = extract_outermost_brackets(match.group(1))
                        blanket = 1
                        if '.' not in result_outer:
                            # self.data_flow_analysis['return'].add(result_outer)
                            pattern = r'.*'
                            man = re.search(pattern, result_outer)
                            self.data_flow_analysis['return'].add(man.group(0))
                            current_id += 1
                            continue;
                    if '.' in match.group(1):
                        if blanket == 1:
                            parts = result_outer.split('.', 1)
                        else:
                            parts = match.group(1).split('.', 1)
                        if len(parts) > 1:
                            self.data_flow_analysis['return'].add(parts[0])
                            current_id += 1
                            continue;
                        else:
                            self.data_flow_analysis['return'].add(line)
                            current_id += 1
                            continue;
                elif 'yield' in line:
                    if 'yield' not in self.data_flow_analysis:
                        self.data_flow_analysis['return'] = set()
                    pattern = r'yield\s+(.*)'
                    match = re.search(pattern, line)
                    # print(match.group(1))
                    # print(self.data_flow_analysis[match.group(1)])
                    if match:
                        if match.group(1) == "None":
                            pattern = r'.*'
                            man = re.search(pattern, 'None')
                            self.data_flow_analysis['return'].add(man.group(0))
                            current_id += 1
                            continue;
                        if match.group(1) in self.data_flow_analysis:
                            self.data_flow_analysis['return'].add(match.group(1))
                            current_id += 1
                            continue;
                        blanket = 0
                        if '(' in match.group(1):
                            result_outer, result_inner = extract_outermost_brackets(match.group(1))
                            blanket = 1
                            if '.' not in result_outer:
                                self.data_flow_analysis['return'].add(result_outer)
                                current_id += 1
                                continue;
                        if '.' in match.group(1):
                            if blanket == 1:
                                parts = result_outer.split('.', 1)
                            else:
                                parts = match.group(1).split('.', 1)
                            if len(parts) > 1:
                                self.data_flow_analysis['return'].add(parts[0])
                                current_id += 1
                                continue;
                            else:
                                self.data_flow_analysis['return'].add(line)
                                current_id += 1
                                continue;
                else:
                    pattern = r'.*'
                    man = re.search(pattern, 'None')
                    self.data_flow_analysis['return'].add(man.group(0))

            current_id += 1

    def perform_analysis(self, code):
        # 将源码字符串拆分成行
        code_lines = code.split('\n')

        # 执行控制流分析
        self.analyze_control_flow(code_lines)

        # 执行数据流分析
        self.analyze_data_flow(code_lines)


def find_lines_with_keyword(code, keyword):
    lines_with_keyword = []

    # 将代码分成行
    code_lines = code.split('\n')

    # 遍历每一行
    for line in code_lines:
        # 检查关键字是否在该行中
        if keyword in line:
            lines_with_keyword.append(line)

    # 将匹配的行拼接成一个字符串
    result_string = '\n'.join(lines_with_keyword)

    return result_string


def infer_simple_type_from_assignment(assignment_string):
    # 匹配赋值语句的正则表达式
    assignment_pattern = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*(.*)\s*$')

    match = assignment_pattern.match(assignment_string)

    if match:
        variable_name, value_str = match.groups()

        # 匹配整数的正则表达式
        int_pattern = re.compile(r'^[+-]?\d+$')

        # 匹配浮点数的正则表达式
        float_pattern = re.compile(r'^[+-]?\d+\.\d+$')

        # 匹配布尔值的正则表达式
        bool_pattern = re.compile(r'^(True|False)$', re.IGNORECASE)

        # 匹配字符串的正则表达式
        str_pattern = re.compile(r'^\'(.*)\'$')

        # 匹配bytes的正则表达式
        bytes_pattern = re.compile(r'^b\'(.*)\'$')

        # 检查字符串格式并返回对应的类型
        if int_pattern.match(value_str):
            return "integer"
        elif float_pattern.match(value_str):
            return "float"
        elif bool_pattern.match(value_str):
            return "bool"
        elif str_pattern.match(value_str):
            return "str"
        elif bytes_pattern.match(value_str):
            return "byte"
        else:
            return None
    else:
        return None


def infer_simple_type_from_value(value_str):
    # 匹配整数的正则表达式
    int_pattern = re.compile(r'^[+-]?\d+$')

    # 匹配浮点数的正则表达式
    float_pattern = re.compile(r'^[+-]?\d+\.\d+$')

    # 匹配布尔值的正则表达式
    bool_pattern = re.compile(r'^(True|False)$', re.IGNORECASE)

    # 匹配字符串的正则表达式
    str_pattern = re.compile(r'^\'(.*)\'$')

    # 匹配bytes的正则表达式
    bytes_pattern = re.compile(r'^b\'(.*)\'$')

    # 检查字符串格式并返回对应的类型
    if int_pattern.match(value_str):
        return "integer"
    elif float_pattern.match(value_str):
        return "float"
    elif bool_pattern.match(value_str):
        return "bool"
    elif str_pattern.match(value_str):
        return "str"
    elif bytes_pattern.match(value_str):
        return "byte"
    else:
        return None


def generate_ast_and_detect_type(assignment_string):
    try:
        # 使用 ast 模块解析赋值语句为 AST
        parsed_ast = ast.parse(assignment_string, mode='exec')

        # 提取赋值语句的右侧值部分
        value_node = parsed_ast.body[0].value

        # 根据 AST 结构判断赋值语句的类型
        if isinstance(value_node, ast.Dict):
            return "dict"
        elif isinstance(value_node, ast.Set):
            return "set"
        elif isinstance(value_node, ast.List):
            return "list"
        elif isinstance(value_node, ast.Tuple):
            return "tuple"
        else:
            return None
    except SyntaxError as e:
        return f"Syntax Error: {e}"


def extract_elements(input_string):
    result = {"substring_before_bracket": None, "inner_substrings": None}

    # 找到等号并做标记
    equal_sign_index = input_string.find('=')

    if equal_sign_index != -1:
        # 从等号位置开始继续扫描，直到遇到左括号或者左方括号
        for i in range(equal_sign_index, len(input_string)):
            if input_string[i] == '(':
                # 从等号位置到左括号位置的子串
                substring_before_parenthesis = input_string[equal_sign_index + 1:i].strip()
                result["outer_type"] = "tuple"
                result["substring_before_bracket"] = substring_before_parenthesis

                # 倒着扫描，找到右括号位置
                for j in range(len(input_string) - 1, i, -1):
                    if input_string[j] == ')':
                        # 括号中间的子串
                        inner_substring = input_string[i + 1:j].strip()

                        # 遍历括号中间的子串，检查是否含有 '[' 或者 '('
                        if '[' in inner_substring or '(' in inner_substring:
                            result["inner_substrings"] = inner_substring
                            result["len"] = 1
                        else:
                            # 使用逗号分割子串
                            inner_substring_list = [part.strip() for part in inner_substring.split(',')]
                            result["len"] = len(inner_substring_list)
                            result["inner_substrings"] = inner_substring_list
                        break

                # 结束外层循环
                break
            elif input_string[i] == '[':
                # 从等号位置到左方括号位置的子串
                substring_before_bracket = input_string[equal_sign_index + 1:i].strip()
                result["outer_type"] = "list"
                result["substring_before_bracket"] = substring_before_bracket

                # 倒着扫描，找到右方括号位置
                for j in range(len(input_string) - 1, i, -1):
                    if input_string[j] == ']':
                        # 括号中间的子串
                        inner_substring = input_string[i + 1:j].strip()

                        # 遍历括号中间的子串，检查是否含有 '[' 或者 '('
                        if '[' in inner_substring or '(' in inner_substring:
                            result["inner_substrings"] = inner_substring
                            result["len"] = 1
                        else:
                            # 使用逗号分割子串
                            inner_substring_list = [part.strip() for part in inner_substring.split(',')]
                            result["len"] = len(inner_substring_list)
                            result["inner_substrings"] = inner_substring_list
                        break

                # 结束外层循环
                break
    else:
        print(input_string)
        print("Equal sign not found.")

    return result


def asignment_analysis(string_test, variable):
    # print("key:")
    # print(key)
    # print("string_test:")
    # print(string_test)
    result_type = infer_simple_type_from_assignment(string_test)
    # print("result_type:")
    # print(result_type)
    if result_type == None:
        # print(string_test)
        # result = re.sub(r'\([^)]*\)', '()', string_test)

        string_test_split = extract_elements(string_test)
        # print(key)
        # print(string_test_split)
        if string_test_split["inner_substrings"] and string_test_split["len"] != 1 and (
                generate_ast_and_detect_type(string_test) == "list" or generate_ast_and_detect_type(
            string_test) == "tuple"):
            # if string_test_split["inner_substrings"] and string_test_split["len"] != 1:
            # print(string_test_split)
            part_type = {}
            index = 0
            for part in string_test_split["inner_substrings"]:
                # print(part)
                part_type_ir = infer_simple_type_from_value(part)
                if part_type_ir == None:
                    instructions = [
                        [
                            {
                                "role": "system",
                                "content": "You are a helpful, respectful and honest assistant. You can the the type of the variable when i give you source code. Please provide me with an answer in the following format:the type of the variable is str/int/float/bool/byte/list/tuple/dict/set/unknow"
                            },
                            {
                                "role": "user",
                                "content": part,
                            }
                        ],
                    ]
                    try:
                        results = generator.chat_completion(
                            instructions,  # type: ignore
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        part_type[part] = results[0]['generation']['content']
                    except:
                        part_type[part] = []
                        pass;
                    # print(part_type[part])
                    # 定义匹配类型的正则表达式
                    type_pattern = re.compile(r'(str|int|float|bool|byte|list|tuple|dict|set|unknow)')

                    # 使用findall方法找到所有匹配的类型
                    matches = type_pattern.findall(part_type[part])

                    # 输出匹配的类型
                    if matches:
                        part_type[part] = ', '.join(matches)
                    else:
                        part_type[part] = None
                else:
                    part_type[part] = part_type_ir
            print(part_type)
            values = list(part_type.values())
            # 使用Counter统计每个值的出现次数
            # value_counts = Counter(values)

            # 找到出现次数最多的值
            # most_common_value = value_counts.most_common(1)[0][0]
            if string_test_split["outer_type"] == "list":
                if all(value == values[0] for value in values):
                    result_type = "list[" + values[0] + "]"
                else:
                    # result_type = "list[typing.Optional[" +most_common_value +"]]"
                    result_type = "list[typing.Any]"
            elif string_test_split["outer_type"] == "tuple":
                if all(value == values[0] for value in values):
                    result_type = "tuple[" + values[0] + "]"
                else:
                    result_type = None
    return result_type


def find_string_in_file(filename, search_string, exact_match=True):
    """
    在文件中搜索字符串。

    参数:
    filename (str): 要搜索的文件名。
    search_string (str): 需要搜索的字符串。
    exact_match (bool): 如果为True，则进行整行匹配；如果为False，则进行部分匹配。

    返回:
    bool: 如果找到字符串，返回True；否则，返回False。
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if (exact_match and line.strip() == search_string) or \
                        (not exact_match and search_string in line.strip()):
                    return True
        return False
    except FileNotFoundError:
        print("指定的文件未找到。")
        return False
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return False

with open("./local_repo_usagegraph.json") as f:
    local_graph = json.load(f)
# 示例用法
with open(os.path.join("./data", "./testset_transformed.json")) as f:
    testset_trans = json.load(f)
with open(os.path.join("./data", "./testset_source.json")) as f:
    testset = json.load(f)
with open(os.path.join("./data", "./testset_usertypes.json")) as f:
    test_user_types = json.load(f)
with open("./data/testset_staticsliced_hop3.json") as f:
    testset_return = json.load(f)

with open("./NSTI_local_preprocessed.json") as f:
    NSTI_local = json.load(f)


#改这个
with open("./NSTI_redundancy0_llama3_preprocessed.json") as f:
    NSTI = json.load(f)
with open("./NSTI_return0.json") as f:
    NSTI_old = json.load(f)
with open("./redundancy4_preprocessed.json") as f:
    TypeGen = json.load(f)
with open("./NSTI_redundancy0_llama3_preprocessed.json") as f:
    redundancy1 = json.load(f)
with open("./NSTI_redundancy1_llama3_preprocessed.json") as f:
    redundancy2 = json.load(f)
with open("./NSTI_redundancy3_llama3_preprocessed.json") as f:
    redundancy3 = json.load(f)
#with open("./NSTI_one_control_preprocessed.json") as f:
#    redundancy3 = json.load(f)

with open("./redundancy4_preprocessed.json") as f:
    redundancy4 = json.load(f)
#规则
with open("./NSTI_return_filter_1_rules_preprocessed.json") as f:
    NSTI_return_rules = json.load(f)


def main(
        ckpt_dir: str,

        temperature: float = 0.2,
        top_p: float = 0.95,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
):
    zero = 0
    total = 0
    total_simple_correct = 0
    predictions = {}
    for key in tqdm(testset_trans.keys()):

        # zero = zero + 1
        # if zero == 3000:
        #    break;
        parts = key.split('--')
        # print(parts[-2])
        # exit(1)

        # if testset_trans[key][2] == "simple" and parts[-1] == "local":
        # if local_graph[parts[0]] == '{}' and parts[-1] == "local":
        if parts[-1] == "local":

            NSTI[key] = NSTI_local[key]
            continue


            user_types = test_user_types[key][1]
            total = total + 1
            string_test = testset[key]
            string_test = find_lines_with_keyword(string_test, parts[-2] + " =")
            # print(string_test)

            filename1 = 'differ1_gpt_local_3333.txt'
            #filename2 = 'differ2_local.txt'
            #filename3 = 'differ3_local.txt'
            # filename4 = 'differ4_return.txt'

            search_string = key

            found1 = find_string_in_file(filename1, search_string, exact_match=False)
            #found2 = find_string_in_file(filename2, search_string, exact_match=False)

            #found3 = find_string_in_file(filename3, search_string, exact_match=False)
            ## found4 = find_string_in_file(filename4, search_string, exact_match=Fal0se)
            # 0.687
            if found1:  # 0.722
                NSTI[key] = redundancy2[key]
            #if found2:  # 0.730
            #    NSTI[key] = redundancy3[key]
            #if found3:  # 0.739#
            #    NSTI[key] = redundancy4[key]
        elif parts[-1] == "arg":



            string_test = testset[key]
            string_test = find_lines_with_keyword(string_test, parts[-2])
            user_types = test_user_types[key][1]


            filename1 = 'differ_llama3_args1.txt'
            filename2 = 'differ_llama3_args2.txt'
            #filename3 = 'differ3_args.txt'
            # filename4 = 'differ4_args.txt'

            search_string = key

            found1 = find_string_in_file(filename1, search_string, exact_match=False)
            found2 = find_string_in_file(filename2, search_string, exact_match=False)

            #found3 = find_string_in_file(filename3, search_string, exact_match=False)
            # found4 = find_string_in_file(filename4, search_string, exact_match=Fal0se)
            # 0.633
            if found1:  # 0.73 #0.695
                NSTI[key] = redundancy2[key]
            if found2:  # 0.745 #0.703
                NSTI[key] = redundancy3[key]
            #if found3:  # 0.769
            #    NSTI[key] = redundancy4[key]

        elif parts[-1] == "return":

            # if key == "repos/AleksanderGondek/pipwatch/api/pipwatch_api/namespaces/version_one.py--get_api_version_one@global--get_api_version_one--return":
            #    print(predictions["repos/AntoineToubhans/MongoTs/mongots/aggregateby.py--parse_aggregateby@global--parse_aggregateby--return"])
            #    exit(1)

            string_test = testset[key]
            user_types = test_user_types[key][1]
            # 创建 IntraProceduralAnalysis 实例
            #intra_procedural_analysis = IntraProceduralAnalysis()

            # 执行分析
            #try:
            #    intra_procedural_analysis.perform_analysis(testset[key])
            #except:

            #    predictions[key] = []
            #    continue
            # 输出控制流图和数据流分析结果
            #control_graph = intra_procedural_analysis.control_flow_graph
            #data_graph = intra_procedural_analysis.data_flow_analysis
            # print(data_graph)

            filename1 = 'differ_llama3_return1.txt'
            filename2 = 'differ_llama3_return2.txt'
            #filename3 = 'differ3_return.txt'
            #filename4 = 'differ4_return.txt'

            search_string = key

            found1 = find_string_in_file(filename1, search_string, exact_match=False)
            found2 = find_string_in_file(filename2, search_string, exact_match=False)

            #found3 = find_string_in_file(filename3, search_string, exact_match=False)
            #found4 = find_string_in_file(filename4, search_string, exact_match=Fal0se)
            #0.514
            if found1:#0.623 #0.582
                NSTI[key] = redundancy2[key]
            if found2:#0.664 #0.60
                NSTI[key] = redundancy3[key]
            #if found3:#0.695
            #    NSTI[key] = redundancy4[key]

            #elif not found4:#0.695
            #    NSTI[key] = redundancy4[key]
            #if found:
                #print(key)
            #    NSTI[key] = TypeGen[key]
            #else:
            #    print("字符串在文件中没有找到。")


            #if 'return' not in data_graph.keys():
            #    zero = zero + 1
            #    NSTI[key] = TypeGen[key]

    for key in tqdm(testset_trans.keys()):#0.697
        if key in NSTI_return_rules.keys():
            NSTI[key] = NSTI_return_rules[key]
        # zero = zero + 1
        # if zero == 3000:
        #    break;
        parts = key.split('--')
    output_json_file = "./NSTI_after_continue2_llama3.json"

    with open(output_json_file, "w") as json_file:
        json.dump(NSTI, json_file, indent=2)
    print(f"Results have been written to {output_json_file}.")


if __name__ == "__main__":
    fire.Fire(main)#5401+
