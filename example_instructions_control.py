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
    #print(string)
    matched = re.findall(pattern, string)
    if len(matched) == 0:
        second_pattern = re.compile(r'\`[a-zA-Z\.\,\[\] ]+\`')
        second_matched = re.findall(second_pattern, string)
        if len(second_matched) == 0:
            return None
        else:
            res = second_matched[-1].replace("`", "").replace('NoneType', 'None')#.replace("is ", "")
            if (" " in res and "[" not in res) or res.lower() == "unknown":
                res = None
            return res
    else:
        res = matched[-1].replace("`", "").replace('NoneType', 'None')#.replace("is ", "")
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
                #print(match.group(1))
                #print(self.data_flow_analysis[match.group(1)])
                if match:
                    if match.group(1) == "None":
                        pattern = r'.*'
                        man = re.search(pattern,'None')
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
                            #self.data_flow_analysis['return'].add(result_outer)
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

        # 匹配列表的正则表达式
        list_pattern = re.compile(r'^\s*\[.*\]\s*$')

        # 匹配元组的正则表达式
        tuple_pattern = re.compile(r'^\s*\((.*)\)\s*$')

        # 匹配字典的正则表达式
        dict_pattern = re.compile(r'^\s*\{.*:.*\}\s*$')

        # 匹配集合的正则表达式
        set_pattern = re.compile(r'^\s*\{[^:{}]*\}\s*$')

        # 检查字符串格式并返回对应的类型
        if int_pattern.match(value_str):
            return "int"
        elif float_pattern.match(value_str):
            return "float"
        elif bool_pattern.match(value_str):
            return "bool"
        elif str_pattern.match(value_str):
            return "str"
        elif bytes_pattern.match(value_str):
            return "bytes"
        elif list_pattern.match(value_str):
            return "list"
        elif tuple_pattern.match(value_str):
            return "tuple"
        elif dict_pattern.match(value_str):
            return "dict"
        elif set_pattern.match(value_str):
            return "set"
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
    #else:
    #    print(input_string)
    #    print("Equal sign not found.")

    return result

def asignment_analysis(string_test, variable):
    # print("key:")
    # print(key)
    # print("string_test:")
    # print(string_test)
    result_type = infer_simple_type_from_assignment(string_test)

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
            #print(part_type)
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
def extract_last_word(sentence):
    # 使用正则表达式查找最后一个单词
    match = re.search(r'\b(\w+)\b[^\w]*$', sentence)
    if match:
        return match.group(1)
    else:
        return None


def determine_return_type(line):
    int_pattern = r'return\s+(-?\d+)\s*$'
    list_pattern = r'return\s+\[(.*)\]\s*$'  # 允许空列表
    tuple_pattern = r'return\s+\((.*)\)\s*$'  # 允许空元组


    if re.match(int_pattern, line):
        return 'int'
    elif re.match(list_pattern, line):
        list_contents = re.match(list_pattern, line).group(1)
        if all(re.match(r'\s*\'[^\']*\'\s*$', item.strip()) for item in list_contents.split(',')):
            return 'list[str]'
        elif all(re.match(r'\s*(-?\d+)\s*$', item.strip()) for item in list_contents.split(',')):
            return 'list[int]'
        else:
            return 'list[mixed]'
    elif re.match(tuple_pattern, line):
        tuple_contents = re.match(tuple_pattern, line).group(1)
        if tuple_contents.strip() == '':  # Handling empty tuple
            return 'tuple'
        if all(re.match(r'\s*\'[^\']*\'\s*$', item.strip()) for item in tuple_contents.split(',')):
            return 'tuple[str]'
        elif all(re.match(r'\s*(-?\d+)\s*$', item.strip()) for item in tuple_contents.split(',')):
            return 'tuple[int]'
        else:
            return 'tuple[mixed]'
    return 'unknown'

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
with open(os.path.join("./data", "./testset_staticsliced_hop3.json")) as f:
    testset = json.load(f)
with open(os.path.join("./data", "./testset_usertypes.json")) as f:
    test_user_types = json.load(f)
with open("E:\gaocunyun\TypeGen\data\\testset_source_filter_1.json") as f:
    testset_return = json.load(f)
#with open("./NSTI_return_after_sts.json") as f:
#    NSTI_return = json.load(f)

with open("./redundancy3_preprocessed.json") as f:
    redundancy3 = json.load(f)
with open("./NSTI_return_after_sts.json") as f:
    NSTI_after_sts = json.load(f)

def main(

    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):

    zero = 0
    total = 0
    ut = 0
    total_simple_correct = 0
    #predictions = {}
    keys_args_with_if = []

    keys_local_with_if = []

    for key in tqdm(testset_trans.keys()):
        #zero = zero + 1
        #if zero == 3000:
        #    break;
        parts = key.split('--')
        # print(parts[-2])
        # exit(1)

        #if testset_trans[key][2] == "simple" and parts[-1] == "local":
        #if local_graph[parts[0]] == '{}' and parts[-1] == "local":
        if parts[-1] == "local":


            #user_types = test_user_types[key][1]

            total = total + 1
            string_test = testset[key]
            keywords_local = find_lines_with_keyword(string_test, parts[-2] + " =")

            #print(string_test)

            keywords_local = keywords_local.split('\n')

            keywords_local = [line.lstrip() for line in keywords_local]


            for line_local in keywords_local:
                string_test_split = extract_elements(line_local)
                #if key == "repos/Hanaasagi/pymem/pymem/snippets/objects.py--summarize@global--otype--local":
                #    print(string_test_split)
                #    exit(1)
                if string_test_split["substring_before_bracket"] == "type":
                   NSTI_after_sts[key] = ["`typing.Type`"]

                if string_test_split["substring_before_bracket"] == "deque":
                   NSTI_after_sts[key] = ["`typing.Deque`"]

                if string_test_split["substring_before_bracket"] == "re.compile":
                   NSTI_after_sts[key] = ["`typing.Pattern`"]
                if string_test_split["substring_before_bracket"] == "re.search":
                   NSTI_after_sts[key] = ["`typing.Optional[typing.Match]`"]

                if string_test_split["substring_before_bracket"] == "defaultdict":
                   NSTI_after_sts[key] = ["`typing.DefaultDict`"]

                if string_test_split["substring_before_bracket"] == "tempfile.NamedTemporaryFile":
                   NSTI_after_sts[key] = ["`typing.IO`"]



            #if len(keywords_local) > 2:




            if "typing.Optional" in  testset_trans[key][1]:

                keys_local_with_if.append(key)  # Save the key that meets the condition

            if "if " in string_test and parts[-2] in string_test:
                lines = string_test.split('\n')
                for line in lines:
                    if ("if " + parts[-2] +" is None") in line:
                        try:
                            a = match_type_for_cot(redundancy3[key][0])
                            NSTI_after_sts[key] = ["`typing.Optional[" + a + "]`"]
                            NSTI_after_sts[key] = ["`" + a + "`"]
                        except:
                            NSTI_after_sts[key] = redundancy3[key][0]
                    #elif
            continue



            local_type = {}
            value_count = {}



            if len(keywords_local) > 2:
                for line_local in keywords_local:



                    try:

                        local_type[line_local] = asignment_analysis(line_local, parts[-2])

                    except:
                        local_type[line_local] = None
                        pass
                    # 计算local_type中每个key的非None值的数量



                    if local_type[line_local] != None:

                        if line_local in value_count:
                            value_count[local_type[line_local]] += 1
                        else:
                            value_count[local_type[line_local]] = 1
                    # 找到非None值数量大于等于2的key
                if len(value_count) > 1:
                    print(key)




                    #for user in user_types:
                    #    if user in line_local:
                    #        local_type[line_local] = user
                    #        break
                    #if user_types in line_local:
                   #     local_type[line_local] =


                   #     question = line_local

                        #instructions = [
                        #    [
                        #        {
                        #            "role": "system",
                        #            "content": "You are a helpful, respectful and honest assistant. You can infer the type of the variable when i give you source code and some user-defined type hints. Please provide me with an answer in the following format:the type of the variable is `here is your predict`"
                        #        },
                        #        {
                        #            "role": "user",
                        #            "content": "Python code:\n" + question + "\nAvailable user-defined types:\n" + ",".join(user_types) + "Q: What is the type of variable " + parts[-2] + "?\nA: "
                        #        }
                        #    ],
                        #]

                        #try:
                        #    results = generator.chat_completion(
                        #        instructions,  # type: ignore
                        #        max_gen_len=max_gen_len,
                        #        temperature=temperature,
                        #        top_p=top_p,
                        #    )
                        #    local_type[line_local] = results[0]['generation']['content']
                        #except:
                        #    local_type[line_local] = None
                        #    pass;


            if local_type != {}:
                NSTI_after_sts[key] = [local_type]
        elif parts[-1] == "arg":
            continue


            string_test = testset[key]



            # Check if "if" and parts[-2] are in the same line of string_test
            if "if " in string_test and parts[-2] in string_test:
                lines = string_test.split('\n')
                for line in lines:
                    if "if " in line and parts[-2] in line:
                        if ("(not " + parts[-2] +")")  in line:
                            NSTI_after_sts[key] = ["`bool`"]

                        if "'" in line and "['" not in line and "('" not in line and "None" not in line and ("if " + parts[-2])  in line and (parts[-2] + ".")  not in line and "*" not in line:
                            NSTI_after_sts[key] = ["`str`"]

                        if (parts[-2] + " >=") in line or (parts[-2] + " <=") in line:
                            NSTI_after_sts[key] = ["`int`"]

                            #if "None" in line:
                            #    predictions[key] = ["`typing.Optional[str]`"]
                        keys_args_with_if.append(key)  # Save the key that meets the condition
                        break

            string_test = find_lines_with_keyword(string_test, parts[-2])
            user_types = test_user_types[key][1]
            question = string_test
            # print(question)
            # question = question
            # print(question)
            # print(f"{key}: {value}")


        elif parts[-1] == "return":
            continue

            #if key == "repos/AleksanderGondek/pipwatch/api/pipwatch_api/namespaces/version_one.py--get_api_version_one@global--get_api_version_one--return":
            #    print(predictions["repos/AntoineToubhans/MongoTs/mongots/aggregateby.py--parse_aggregateby@global--parse_aggregateby--return"])
            #    exit(1)
            string_test = testset[key]
            user_types = test_user_types[key][1]
            # 创建 IntraProceduralAnalysis 实例
            intra_procedural_analysis = IntraProceduralAnalysis()

            # 执行分析
            try:
                intra_procedural_analysis.perform_analysis(testset[key])
            except:
                pass

            # 输出控制流图和数据流分析结果
            control_graph = intra_procedural_analysis.control_flow_graph
            data_graph = intra_procedural_analysis.data_flow_analysis
            #print(data_graph)
            return_type = {}

            if 'return' not in data_graph.keys():
                #predictions[key] = []
                continue

            #print("len_return:")
            #print(len(data_graph["return"]))
            #print("data_graph")
            #print(data_graph)
            keywords_return = find_lines_with_keyword(string_test, " return")
            keywords_return = keywords_return.split('\n')
            keywords_return = [line.lstrip() for line in keywords_return]

            direct_type = {}
            return_from_line = []

            for line_return in keywords_return:
                # if key == "repos/awslabs/gluon-ts/src/gluonts/mx/distribution/box_cox_transform.py--event_shape@InverseBoxCoxTransformOutput--event_shape--return":

                #   print("000000000000000000000")
                #   print(line_return)
                #   line_return = line_return.strip()
                #   print(determine_return_type(line_return))
                #   exit(1)
                if determine_return_type(line_return) != 'unknown':
                    return_type[line_return] = determine_return_type(line_return)
                    direct_type[line_return] = 1
                    continue
                direct_type[line_return] = 0
                # print("dddddddddddddddddddddddddd")
                pattern = r'return\s+(.*)'
                match = re.search(pattern, line_return)

                # print("match.group(1)")
                # print(match.group(1))
                # exit(1)
                if match:
                    if match.group(1) == "None":
                        pattern = r'.*'
                        man = re.search(pattern, 'None')
                        return_from_line.append(man.group(0))
                    blanket = 0
                    if '(' in match.group(1):
                        result_outer, result_inner = extract_outermost_brackets(match.group(1))
                        blanket = 1
                        if '.' not in result_outer:
                            # print("result_outer:")
                            # print(result_outer)
                            # print("result_inner:")
                            # print(result_inner)
                            # print("key:")
                            # print(key)
                            # self.data_flow_analysis['return'].add(result_outer)
                            pattern = r'.*'
                            return_from_line.append(result_outer)
                    elif '.' in match.group(1):
                        if blanket == 1:
                            parts_blank = result_outer.split('.', 1)
                        else:
                            parts_blank = match.group(1).split('.', 1)
                        if len(parts_blank) > 1:
                            return_from_line.append(parts_blank[0])
                        else:
                            return_from_line.append(line_return)
                    elif '[' in match.group(1):
                        result_outer, result_inner = extract_outermost_brackets_for_list(match.group(1))
                        return_from_line.append(result_outer)
                    else:
                        return_from_line.append(match.group(1))
                else:
                    pattern = r'.*'
                    man = re.search(pattern, 'None')
                    return_from_line.append(man.group(0))
            #print(return_from_line)
            #print(keywords_return)

            for line_return in keywords_return:
                if direct_type[line_return] == 0:
                    # for first_return in data_graph["return"]:
                    for first_return in return_from_line:
                        # if key == "repos/ChrisCummins/phd/datasets/benchmarks/gpgpu/gpgpu.py--benchmarks@DummyJustForTesting--benchmarks--return":
                        #    print(return_from_line)
                        #    exit(1)
                        # print("first_return:")
                        # print(first_return)
                        # print("datagraph:")
                        # print(data_graph)

                        lines = ""
                        all_to_gpt = 0
                        if first_return not in data_graph:

                            if first_return == 'None' or first_return == "'\\n'" or first_return == "" or first_return == "None":
                                return_type[first_return] = "`None`"
                                continue
                            # elif first_return in user_types:
                            #    return_type[first_return] = first_return
                            else:
                                all_to_gpt = 1

                        else:
                            for line_num in data_graph[first_return]:
                                line = get_line_by_number(string_test, line_num)
                                line_first_return = find_lines_with_keyword(string_test,
                                                                            "return " + first_return)  # string_test是全部的代码，first_return是关键字

                                # lines = lines + line + '\n' + line_first_return
                                # return_type[first_return] = None
                                # continue

                                type_ir = asignment_analysis(line, first_return)
                                lines = lines + line + '\n'
                                if type_ir != None:
                                    return_type[first_return] = type_ir
                                else:
                                    all_to_gpt = 1
                        # print("return_type[first_return]:")
                        # print(return_type[first_return])
                        # print("line_first_return:")
                        # print(line_first_return)
                        # print("lines:")
                        # print(lines)
                # print("return_type:")
                # print(return_type)
            if return_type:
                if len(return_type) == 1:
                    predictions[key] = [next(iter(return_type.values()))]
                else:
                    print(key)
                    print(return_type)
                    zero = zero + 1
                    if zero == 10:
                        exit(1)
            else:
                # Handle the case when return_type is empty
                # For example, you could set predictions[key] to some default value
                # predictions[key] = []  # or any other default value
                continue

            # print("all_to_gpt:")
            # print(all_to_gpt)
            if all_to_gpt == 1:
                continue


            #print("keywords_return:")
            #print(keywords_return)

    # Write the collected keys to a file named control_flow_args.txt with UTF-8 encoding

    with open('control_flow_local.txt', 'w', encoding='utf-8') as f:
        for key in keys_local_with_if:
            f.write(f"{key}\n")

    with open('control_flow_args.txt', 'w', encoding='utf-8') as f:
        for key in keys_args_with_if:
            f.write(f"{key}\n")

    output_json_file = "./NSTI_after_symbolic.json"

    with open(output_json_file, "w") as json_file:
        json.dump(NSTI_after_sts, json_file, indent=2)
    print(f"Results have been written to {output_json_file}.")


if __name__ == "__main__":
    fire.Fire(main)