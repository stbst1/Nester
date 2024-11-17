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
    counts = Counter(lst)
    most_common = counts.most_common(1)
    return most_common[0][0]

def extract_string_between_last_two_quotes(input_string):
    last_quote_index = input_string.rfind("`")

    second_last_quote_index = input_string.rfind("`", 0, last_quote_index)
    if second_last_quote_index != -1 and last_quote_index != -1:
        extracted_string = input_string[second_last_quote_index + 1:last_quote_index]
        return extracted_string
    else:
        return
def get_line_by_number(input_string, line_number):
    lines = input_string.split('\n')

    if 1 <= line_number <= len(lines):
        return lines[line_number - 1]
    else:
        return "Invalid Line Number!"
def match_type_for_cot(string):
    pattern = re.compile(r'\`[a-zA-Z\.]+(?:\[[a-zA-Z\. ]+(?:\,[a-zA-Z\. ]+)*\])*\`')
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
                if not stack
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
        self.control_flow_graph = {}
        self.data_flow_analysis = {}

    def analyze_control_flow(self, code_lines):
        current_id = 1
        for line in code_lines:
            if 'if' in line:
                self.control_flow_graph[current_id] = [current_id + 1, current_id + 2]
                current_id += 2
            else:
                self.control_flow_graph[current_id] = [current_id + 1]
                current_id += 1

    def analyze_data_flow(self, code_lines):
        current_id = 1
        for line in code_lines:
            if 'def ' in line:
                parameters = extract_parameters_from_method(line)
                for item in parameters:
                    if item not in self.data_flow_analysis:
                        self.data_flow_analysis[item] = set()
                    self.data_flow_analysis[item].add(current_id)
            if ' = ' in line:
                parts = line.split('=')
                variable = parts[0].strip()
                if variable not in self.data_flow_analysis:
                    self.data_flow_analysis[variable] = set()
                self.data_flow_analysis[variable].add(current_id)
            elif 'print' in line:
                parts = line.split('(')
                variable = parts[1].split(')')[0].strip()
                if variable not in self.data_flow_analysis:
                    self.data_flow_analysis[variable] = set()
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
        code_lines = code.split('\n')

        self.analyze_control_flow(code_lines)

        self.analyze_data_flow(code_lines)
def find_lines_with_keyword(code, keyword):
    lines_with_keyword = []

    code_lines = code.split('\n')

    for line in code_lines:
        if keyword in line:
            lines_with_keyword.append(line)

    result_string = '\n'.join(lines_with_keyword)

    return result_string

def infer_simple_type_from_assignment(assignment_string):
    assignment_pattern = re.compile(r'^\s*([a-zA-Z_]\w*)\s*=\s*(.*)\s*$')

    match = assignment_pattern.match(assignment_string)

    if match:
        variable_name, value_str = match.groups()

        int_pattern = re.compile(r'^[+-]?\d+$')

        float_pattern = re.compile(r'^[+-]?\d+\.\d+$')

        bool_pattern = re.compile(r'^(True|False)$', re.IGNORECASE)

        str_pattern = re.compile(r'^\'(.*)\'$')

        bytes_pattern = re.compile(r'^b\'(.*)\'$')

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
    int_pattern = re.compile(r'^[+-]?\d+$')

    float_pattern = re.compile(r'^[+-]?\d+\.\d+$')

    bool_pattern = re.compile(r'^(True|False)$', re.IGNORECASE)

    str_pattern = re.compile(r'^\'(.*)\'$')

    bytes_pattern = re.compile(r'^b\'(.*)\'$')

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
        parsed_ast = ast.parse(assignment_string, mode='exec')

        value_node = parsed_ast.body[0].value

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

    equal_sign_index = input_string.find('=')

    if equal_sign_index != -1:
        for i in range(equal_sign_index, len(input_string)):
            if input_string[i] == '(':
                substring_before_parenthesis = input_string[equal_sign_index + 1:i].strip()
                result["outer_type"] = "tuple"
                result["substring_before_bracket"] = substring_before_parenthesis

                for j in range(len(input_string) - 1, i, -1):
                    if input_string[j] == ')':
                        inner_substring = input_string[i + 1:j].strip()

                        if '[' in inner_substring or '(' in inner_substring:
                            result["inner_substrings"] = inner_substring
                            result["len"] = 1
                        else:
                            inner_substring_list = [part.strip() for part in inner_substring.split(',')]
                            result["len"] = len(inner_substring_list)
                            result["inner_substrings"] = inner_substring_list
                        break

                break
            elif input_string[i] == '[':
                substring_before_bracket = input_string[equal_sign_index + 1:i].strip()
                result["outer_type"] = "list"
                result["substring_before_bracket"] = substring_before_bracket

                for j in range(len(input_string) - 1, i, -1):
                    if input_string[j] == ']':
                        inner_substring = input_string[i + 1:j].strip()

                        if '[' in inner_substring or '(' in inner_substring:
                            result["inner_substrings"] = inner_substring
                            result["len"] = 1
                        else:
                            inner_substring_list = [part.strip() for part in inner_substring.split(',')]
                            result["len"] = len(inner_substring_list)
                            result["inner_substrings"] = inner_substring_list
                        break

                break
    else:
        print(input_string)
        print("Equal sign not found.")

    return result

def asignment_analysis(string_test, variable):
    result_type = infer_simple_type_from_assignment(string_test)
    if result_type == None:

        string_test_split = extract_elements(string_test)
        if string_test_split["inner_substrings"] and string_test_split["len"] != 1 and (
                generate_ast_and_detect_type(string_test) == "list" or generate_ast_and_detect_type(
                string_test) == "tuple"):
            part_type = {}
            index = 0
            for part in string_test_split["inner_substrings"]:
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
                    type_pattern = re.compile(r'(str|int|float|bool|byte|list|tuple|dict|set|unknow)')

                    matches = type_pattern.findall(part_type[part])

                    if matches:
                        part_type[part] = ', '.join(matches)
                    else:
                        part_type[part] = None
                else:
                    part_type[part] = part_type_ir
            print(part_type)
            values = list(part_type.values())
            if string_test_split["outer_type"] == "list":
                if all(value == values[0] for value in values):
                    result_type = "list[" + values[0] + "]"
                else:
                    result_type = "list[typing.Any]"
            elif string_test_split["outer_type"] == "tuple":
                if all(value == values[0] for value in values):
                    result_type = "tuple[" + values[0] + "]"
                else:
                    result_type = None
    return result_type
def extract_last_word(sentence):
    match = re.search(r'\b(\w+)\b[^\w]*$', sentence)
    if match:
        return match.group(1)
    else:
        return None
with open("./local_repo_usagegraph.json") as f:
    local_graph = json.load(f)
with open(os.path.join("./data", "./testset_transformed.json")) as f:
    testset_trans = json.load(f)
with open(os.path.join("./data", "./testset_source.json")) as f:
    testset = json.load(f)
with open(os.path.join("./data", "./testset_usertypes.json")) as f:
    test_user_types = json.load(f)
with open("/home/ligen/lg/codellama/data/testset_staticsliced_hop3.json") as f:
    testset_return = json.load(f)
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    zero = 0
    total = 0
    ut = 0
    total_simple_correct = 0
    predictions = {}
    for key in tqdm(testset_trans.keys()):
        parts = key.split('--')
        if parts[-1] == "local":

            user_types = test_user_types[key][1]
            total = total + 1
            string_test = testset[key]
            keywords_local = find_lines_with_keyword(string_test, parts[-2] + " =")

            keywords_local = keywords_local.split('\n')

            local_type = {}

            for line_local in keywords_local:
                try:
                    local_type[line_local] = asignment_analysis(line_local, parts[-2])
                except:
                    local_type[line_local] = None
                    pass
                if local_type[line_local] == None:
                    local_type = {}
                    break

                    question = line_local

                    instructions = [
                        [
                            {
                                "role": "system",
                                "content": "You are a helpful, respectful and honest assistant. You can infer the type of the variable when i give you source code and some user-defined type hints. Please provide me with an answer in the following format:the type of the variable is `here is your predict`"
                            },
                            {
                                "role": "user",
                                "content": "Python code:\n" + question + "\nAvailable user-defined types:\n" + ",".join(user_types) + "Q: What is the type of variable " + parts[-2] + "?\nA: "
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
                        local_type[line_local] = results[0]['generation']['content']
                    except:
                        local_type[line_local] = None
                        pass;
            if local_type != {}:

                if len(local_type) == 1:
                    predictions[key] = [next(iter(local_type.values()))]
                    continue
                elif len(local_type) > 1:
                    union_type = []

                    for value in local_type.values():
                        if value == None:
                            continue
                        if " " in value:

                            if "`" in value or "'" in value:
                                index = len(value) - 1
                                in_quote = False
                                content = ""

                                while index >= 0:
                                    if (value[index] == "`") or (value[index] == "'"):
                                        in_quote = not in_quote
                                    elif in_quote:
                                        content = value[index] + content
                                    elif content:
                                        break
                                    index -= 1
                                value = content

                                if value == "":
                                    pass
                            else:
                                value = extract_last_word(value)

                        union_type.append(value)


                    simple_type = 0
                    user_defiend_type = 0
                    generic_type = 0
                    if 'str' in union_type \
                            or 'string' in union_type \
                            or 'int' in union_type \
                            or 'integer' in union_type \
                            or 'float' in union_type \
                            or 'bool' in union_type \
                            or 'Bool' in union_type \
                            or 'Boolean' in union_type \
                            or 'boolean' in union_type \
                            or 'bytes' in union_type:
                        simple_type = 1
                        keywords_not_del = ['str', 'string', 'int', 'integer', 'float', 'bool', 'Bool', 'Boolean',
                                            'boolean', 'bytes', 'None', '`None`']
                        union_type = filter_list(union_type, keywords_not_del)

                    if 'None' in union_type or '`None`' in union_type:
                        if 'str' in union_type or 'string' in union_type:
                            local_type = "`typing.Optional[str]`"
                        elif 'int' in union_type or 'integer' in union_type:
                            local_type = "`typing.Optional[int]`"
                        elif 'float' in union_type:
                            local_type = "`typing.Optional[float]`"
                        elif 'bool' in union_type or 'Bool' in union_type or 'Boolean' in union_type or 'boolean' in union_type:
                            local_type = "`typing.Optional[bool]`"
                        elif 'bytes' in union_type:
                            local_type = "`typing.Optional[bytes]`"
                        else:
                            local_type = "`typing.Optional[" + most_common_element(union_type) + "]`"

                    else:
                        if simple_type == 1:
                            unique_list = list(set(union_type))
                            local_type = "`typing.Union[" + ','.join(unique_list) + "]"
                        else:
                            local_type = {}

            if local_type == "`typing.Optional[]`" or local_type == "`typing.Optional[None]`" or local_type == "`typing.Optional[`None`]`":
                local_type = {}

            if local_type == {}:
                try:
                    local_type = asignment_analysis(string_test, parts[-2])
                except:
                    local_type = None
                    pass

                if local_type != None:
                    local_type = "`" + local_type + "`"

                else:
                    instructions = [
                        [
                            {
                                "role": "system",
                                "content": "You are a helpful, respectful and honest assistant. You can infer local type when i give you source code and some user-defined type hints. Please provide me with an answer in the following format:the variable type is `here is your predict`"
                            },
                            {
                                "role": "user",
                                "content": "Python code:\n" + testset[key] + "\nAvailable user-defined types:\n" + ",".join(
                                    user_types) + "Q: What is the type of the variable " + parts[-2] + "?\nA: "
                            }
                        ],
                    ]
                    # print(instructions)
                    try:
                        results = generator.chat_completion(
                            instructions,  # type: ignore
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        local_type = results[0]['generation']['content']
                    except:
                        local_type = ""
                        pass;

            predictions[key] = [local_type]
        elif parts[-1] == "arg":
            continue
            string_test = testset[key]
            string_test = find_lines_with_keyword(string_test, parts[-2])
            user_types = test_user_types[key][1]
            question = string_test


            instructions = [
                [
                    #{
                    #    "role": "system",
                    #    "content": "You are a helpful, respectful and honest assistant. You can inference the type of the variable when i give you source code and some type hints.Please provide me with an answer in the following format:the type of the variable is `here is your predict`"
                    #},
                    {
                        "role": "user",
                        "content": "Python code:\n"+question+"\nAvailable user-defined types:\n"+",".join(user_types)+"Q: What is the type of the argument "+parts[-2]+"?\nA: "
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
                predictions[key] = [results[0]['generation']['content']]
            except:
                predictions[key] = []
                pass;
        elif parts[-1] == "return":
            continue

            string_test = testset[key]
            user_types = test_user_types[key][1]

            intra_procedural_analysis = IntraProceduralAnalysis()


            try:
                intra_procedural_analysis.perform_analysis(testset[key])
            except:
                instructions = [
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful, respectful and honest assistant. You can infer return type when i give you source code and some user-defined type hints. Please provide me with an answer in the following format:the return type of the function is `here is your predict`"
                        },
                        {
                            "role": "user",
                            "content": "Python code:\n" + testset[key] + "\nAvailable user-defined types:\n" + ",".join(
                                user_types) + "Q: What is the type of the yield " + parts[-2] + "?\nA: "
                        }
                    ],
                ]
                # print(instructions)
                try:
                    results = generator.chat_completion(
                        instructions,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    predictions[key] = [results[0]['generation']['content']]
                except:
                    predictions[key] = []
                    pass;
                continue

            control_graph = intra_procedural_analysis.control_flow_graph
            data_graph = intra_procedural_analysis.data_flow_analysis
            #print(data_graph)
            return_type = {}

            if 'return' not in data_graph.keys():
                predictions[key] = []
                continue

            
            keywords_return = find_lines_with_keyword(string_test, " return")
            keywords_return = keywords_return.split('\n')
            #print("keywords_return:")
            #print(keywords_return)

            return_type_static = {}

            for line_return in keywords_return:


                return_type_static[line_return] = 0

                instructions = [
                    [
                        {
                            "role": "system",
                            "content": "assume you are an expert in type inference. you can infer the type of return value from a single line of code,and if you dont know, say you dont know.  Answer the question according to examples Q&A."
                        },
                        {
                            "role": "user",
                            "content": 'Q:code:return dest\nA:I do not know. Please give me more information.\n' +
                                       'Q:code:return result\nA:I do not know. Please give me more information.\n' +
                                       'Q:code:return create_many(CategoryDTO, data)\nA:I do not know. Please give me more information.\n' +
                                       'Q:code:return first.string\nA:the return type is \'string\'.\n' +
                                       'Q:code:return first_p.text\nA:the return type is \'string\'.\n' +
                                       'Q:code:return HttpResponse\nA:the return type is \'HttpResponse\'.\n' +
                                       'Q:code:return Position(float(x),float(y))\nA:the return type is \'Position\'\n' +
                                       'Q:code:return\nA:the return type is \'None\'.\n' + 'Q:code:' + line_return + '\nA:'
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
                except:
                    continue


                first_line = results[0]['generation']['content'].split('\n')[0]
                last_line = results[0]['generation']['content'].split('\n')[-1]


                if ("do not know" in first_line) or (
                        ("Sure" in first_line) and (("do not know" in last_line) or ("the same" in last_line))):  # 静态分析
                    return_type_static[line_return] = 1
                elif "Sure" in first_line and "do not know" not in last_line:

                    return_type[line_return] = last_line
                    continue
                else:

                    try:
                        return_type[line_return] = first_line
                    except:
                        pass
                    continue


            return_from_line = []

            for line_return in keywords_return:
                
                if return_type_static[line_return] == 1:

                    
                    pattern = r'return\s+(.*)'
                    match = re.search(pattern, line_return)

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



            for first_return in return_from_line:


                lines = ""
                all_to_gpt = 0
                if first_return not in data_graph:

                    if first_return == 'None' or first_return == "'\\n'" or first_return == "" or first_return=="None":
                        return_type[first_return] = "`None`"
                        continue
                    elif first_return in user_types:
                        return_type[first_return] = first_return
                    else:
                        
                        all_to_gpt = 1

                else:
                    for line_num in data_graph[first_return]:
                        line = get_line_by_number(string_test, line_num)
                        line_first_return = find_lines_with_keyword(string_test, "return " + first_return)
                        if "def" in line:
                            string_outter = extract_outermost_brackets(line_first_return.replace("return ", ''))
                            for u in user_types:
                                if u in string_outter:
                                    return_type[first_return] = u
                                    break
                            return_type[first_return] = type_ir

                            instructions = [
                                [
                                    {
                                        "role": "system",
                                        "content": "you are good at type inference in python.just give me the type without other words."
                                    },
                                    {
                                        "role": "user",
                                        "content": "what is the return value of " + string_outter[0]
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
                                return_type[first_return] = results[0]['generation']['content']
                            except:
                                return_type[first_return] = ""
                                pass;
                            break

                        type_ir = asignment_analysis(line, first_return)
                        lines = lines + line + '\n'
                        if type_ir != None:
                            return_type[first_return] = type_ir
                        else:
                            return_type[first_return] = None

                    if lines == "":
                        lines = testset[key]
                    else:
                        lines = lines = lines + '\n' + line_first_return
                    if return_type[first_return] == None:  # gpt(lines+typehints)
                        #if key == "repos/10sr/webtools/export_as_bookmark/bookmark_exporter.py--export@BookmarkExporter--export--return":
                        #    print(lines)
                        #    exit(1)
                        instructions = [
                            [
                                {
                                    "role": "system",
                                    "content": "You are a helpful, respectful and honest assistant. You can infer the type of the variable when i give you source code and some user-defined type hints. Please provide me with an answer in the following format:the type of the variable is `here is your predict`"
                                },
                                {
                                    "role": "user",
                                    "content": "Python code:\n" + lines + "\nAvailable user-defined types:\n" + ",".join(user_types) + "Q: What is the type of variable " + first_return + "?\nA: "
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
                            return_type[first_return] = results[0]['generation']['content']
                        except:
                            return_type[first_return] = ""
                            pass;

                if all_to_gpt == 1:
                    instructions = [
                        [
                            {
                                "role": "system",
                                "content": "You are a helpful, respectful and honest assistant. You can infer the type of the variable when i give you source code and some user-defined type hints. Please provide me with an answer in the following format:the type of the variable is `here is your predict`"
                            },
                            {
                                "role": "user",
                                "content": "Python code:\n" + testset[
                                    key] + "\nAvailable user-defined types:\n" + ",".join(
                                    user_types) + "Q: What is the return type of function " + parts[-2] + "?\nA: "
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
                        return_type[parts[-2]] = results[0]['generation']['content']
                    except:
                        return_type[parts[-2]] = ""
                        pass;


            if len(return_type) == 1:
                predictions[key] = [next(iter(return_type.values()))]
                continue
            elif len(return_type) > 1:
                union_type = []

                for value in return_type.values():
                    if " " in value:
                        #print("value[-1]")

                        index = len(value) - 1
                        in_quote = False
                        content = ""

                        while index >= 0:
                            if (value[index] == "`") or (value[index] == "'"):
                                in_quote = not in_quote
                            elif in_quote:
                                content = value[index] + content
                            elif content:
                                break
                            index -= 1
                        value = content

                        if value == "":
                            pass




                    union_type.append(value)
                

                simple_type = 0
                if 'str' in union_type \
                or 'string' in union_type\
                or 'int' in union_type\
                or 'integer' in union_type\
                or 'float' in union_type\
                or 'bool' in union_type\
                or 'Bool' in union_type\
                or 'Boolean' in union_type\
                or 'boolean' in union_type\
                or 'bytes' in union_type:
                    simple_type = 1
                    keywords_not_del = ['str', 'string', 'int', 'integer','float' , 'bool','Bool' , 'Boolean','boolean' ,'bytes', 'None','`None`']
                    union_type = filter_list(union_type, keywords_not_del)

                if 'None' in union_type or '`None`' in union_type:
                    if 'str' in union_type or 'string' in union_type:
                        return_type = "`typing.Optional[str]`"
                    elif 'int' in union_type or 'integer' in union_type:
                        return_type = "`typing.Optional[int]`"
                    elif 'float' in union_type:
                        return_type = "`typing.Optional[float]`"
                    elif 'bool' in union_type or 'Bool' in union_type or 'Boolean' in union_type or 'boolean' in union_type:
                        return_type = "`typing.Optional[bool]`"
                    elif 'bytes' in union_type:
                        return_type = "`typing.Optional[bytes]`"
                    else:
                        return_type = "`typing.Optional[" + most_common_element(union_type) + "]`"

                else:
                    if simple_type == 1:
                        unique_list = list(set(union_type))
                        return_type = "`typing.Union["+ ','.join(unique_list) + "]"
                    else:


                
                        instructions = [
                            [
                                {
                                    "role": "system",
                                    "content": "Attempt tp combile mutiple user-defined types into one type based on inheritance relationship. You can directly use father class as the answer. just give me the combined type without other words."
                                },
                                {
                                    "role": "user",
                                    "content": ', '.join(union_type)
                                }
                            ],
                        ]
                        results = generator.chat_completion(
                            instructions,  # type: ignore
                            max_gen_len=max_gen_len,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        return_type = "`" + results[0]['generation']['content'] + "`"

            if return_type == "`typing.Optional[]`" or return_type == "`typing.Optional[None]`" or return_type == "`typing.Optional[`None`]`":
                return_type = {}

            if return_type == {}:
                instructions = [
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful, respectful and honest assistant. You can infer return type when i give you source code and some user-defined type hints. Please provide me with an answer in the following format:the return type of the function is `here is your predict`"
                        },
                        {
                            "role": "user",
                            "content": "Python code:\n" + testset[key] + "\nAvailable user-defined types:\n" + ",".join(user_types) + "Q: What is the type of the return " + parts[-2] + "?\nA: "
                        }
                    ],
                ]
                # print(instructions)
                try:
                    results = generator.chat_completion(
                        instructions,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    return_type = results[0]['generation']['content']
                except:
                    return_type = ""
                    pass;

            predictions[key] = [return_type]


    output_json_file = "/home/ligen/lg/nsti/NSTI_local.json"

    with open(output_json_file, "w") as json_file:
        json.dump(predictions, json_file, indent=2)
    print(f"Results have been written to {output_json_file}.")


if __name__ == "__main__":
    fire.Fire(main)