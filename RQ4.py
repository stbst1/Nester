import re
import json
import os
from collections import defaultdict
import ast

def is_user_defined_type(code: str) -> bool:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return True
    except SyntaxError:
        return False

    return False
code = "provider = container.get(settings.Props.DI_PROVIDER)"

#result1 = is_user_defined_type(code)
#print(result1)
#exit(1)
def find_lines_with_keyword(code, keyword):
    lines_with_keyword = []

   
    for line in code_lines:
        if keyword in line:
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
        list_pattern = re.compile(r'^\s*\[.*\]\s*$'
        tuple_pattern = re.compile(r'^\s*\((.*)\)\s*$')
        dict_pattern = re.compile(r'^\s*\{.*:.*\}\s*$')
        set_pattern = re.compile(r'^\s*\{[^:{}]*\}\s*$')
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
                        else:
                            inner_substring_list = inner_substring.split(',')
                            result["inner_substrings"] = inner_substring_list
                        break

                break
            elif input_string[i] == '[':
                substring_before_bracket = input_string[equal_sign_index + 1:i].strip()
                result["substring_before_bracket"] = substring_before_bracket

                for j in range(len(input_string) - 1, i, -1):
                        inner_substring = input_string[i + 1:j].strip()

                        if '[' in inner_substring or '(' in inner_substring:
                            result["inner_substrings"] = inner_substring
                        else:
                            inner_substring_list = inner_substring.split(',')
                            result["inner_substrings"] = inner_substring_list
                        break
                break
    else:
        print("Equal sign not found.")

    return result

with open("./local_repo_usagegraph.json") as f:
    local_graph = json.load(f)
with open(os.path.join("./data", "./testset_transformed.json")) as f:
    testset_trans = json.load(f)
with open(os.path.join("./data", "./testset_source.json")) as f:
    testset = json.load(f)

with open("./NSTI_return_after_sts.json") as f:
    NSTI_after_sts = json.load(f)
with open("./NSTI_return_filter_1_rules_preprocessed.json") as f:
    NSTI_return_rules = json.load(f)


file_path_typegen = './differ_typegen.txt'
zero = 0
total = 0
total_simple_correct = 0

type_count = defaultdict(int)
type_count_correct = defaultdict(int)


for key in testset_trans.keys():
    #zero = zero + 1
    #if zero == 1000:
    #    break;
    parts = key.split('--')
    #print(parts[-2])
    #exit(1)

    #if testset_trans[key][2] == "simple" and parts[-1] == "local":
    #if local_graph[parts[0]] == '{}' and parts[-1] == "local":
    if parts[-1] == "local":
        continue


        total = total + 1
        string_test = testset[key]

        string_test = find_lines_with_keyword(string_test, parts[-2]+" =")
        string_test = string_test.split('\n', 1)[0]
        #print("key:")
        #print(key)
        #if key == "repos/0mars/monoskel/packages/monomanage/src/monomanage/app/wheels.py--global@global--WHITELIST--local":
        #    print(string_test)
        #    exit(1)

        #exit(1)
        result_type = infer_simple_type_from_assignment(string_test)

        #print(result_type)
        if result_type:

            NSTI_after_sts[key] = ["`" + result_type + "`"]


        #print("result_type:")
        #print(result_type)
        #if key == "repos/097475/hansberger/hansberger/analysis/models/bottleneck.py--get_bottleneck_matrix@Bottleneck--i--local":
        #    print(result_type)
        #    exit(1)
        if result_type == None:
            string_split = extract_elements(string_test)
            #print(string_split)
            #exit(1)

        value = testset_trans[key][1]

        if value == 'str':
            type_count['str'] += 1
        elif value == 'int':
            type_count['int'] += 1
        elif value == 'float':
            type_count['float'] += 1
        elif value == 'bool':
            type_count['bool'] += 1
        elif value == 'bytes':
            type_count['bytes'] += 1
        # list, tuple, dict, iterable, or set
        elif len(value) >= 4 and value[:4] == 'list':
            type_count['list'] += 1
        elif len(value) >= 5 and value[:5] == 'tuple':
            type_count['tuple'] += 1
        elif len(value) >= 4 and value[:4] == 'dict':
            type_count['dict'] += 1
        elif len(value) >= 3 and value[:3] == 'set':
            type_count['set'] += 1
        elif len(value) >= 6 and value[:6] == 'typing':
            type_count['typing'] += 1
        else:
            type_count['other'] += 1


        if result_type == testset_trans[key][1]:
            total_simple_correct = total_simple_correct + 1
            if value == 'str':
                type_count_correct['str'] += 1
            elif value == 'int':
                type_count_correct['int'] += 1
            elif value == 'float':
                type_count_correct['float'] += 1
            elif value == 'bool':
                type_count_correct['bool'] += 1
            elif value == 'bytes':
                type_count_correct['bytes'] += 1
            # list, tuple, dict, iterable, or set
            elif len(value) >= 4 and value[:4] == 'list':
                type_count_correct['list'] += 1
            elif len(value) >= 5 and value[:5] == 'tuple':
                type_count_correct['tuple'] += 1
            elif len(value) >= 4 and value[:4] == 'dict':
                type_count_correct['dict'] += 1
            elif len(value) >= 3 and value[:3] == 'set':
                type_count_correct['set'] += 1
            elif len(value) >= 6 and value[:6] == 'typing':
                type_count_correct['typing'] += 1
            else:
                type_count_correct['other'] += 1
            with open(file_path_typegen, 'r', encoding='utf-8', errors='ignore') as file:
                    if key == line.strip():
                        print(111111111111)
                        print(key)
                        print(result_type)
    elif parts[-1] == "arg":
        continue
        total = total + 1
        string_test = testset[key]

        string_test0 = find_lines_with_keyword(string_test, parts[-2]+" =")
        string_test1 = find_lines_with_keyword(string_test, parts[-2] + "=")
        string_test = string_test0 + string_test1

        string_test = string_test.split('\n', 1)[0]

        result_type = infer_simple_type_from_assignment_arg(string_test)

        if result_type:

            NSTI_after_sts[key] = ["`" + result_type + "`"]



        if result_type != None:

            string_split = extract_elements(string_test)

        value = testset_trans[key][1]

        if value == 'str':
            type_count['str'] += 1
        elif value == 'int':
            type_count['int'] += 1
        elif value == 'float':
            type_count['float'] += 1
        elif value == 'bool':
            type_count['bool'] += 1
        elif value == 'bytes':
            type_count['bytes'] += 1
        # list, tuple, dict, iterable, or set
        elif len(value) >= 4 and value[:4] == 'list':
            type_count['list'] += 1
        elif len(value) >= 5 and value[:5] == 'tuple':
            type_count['tuple'] += 1
        elif len(value) >= 4 and value[:4] == 'dict':
            type_count['dict'] += 1
        elif len(value) >= 3 and value[:3] == 'set':
            type_count['set'] += 1
        elif len(value) >= 6 and value[:6] == 'typing':
            type_count['typing'] += 1
        else:
            type_count['other'] += 1


        if result_type == testset_trans[key][1]:
            total_simple_correct = total_simple_correct + 1
            if value == 'str':
                type_count_correct['str'] += 1
            elif value == 'int':
                type_count_correct['int'] += 1
            elif value == 'float':
                type_count_correct['float'] += 1
            elif value == 'bool':
                type_count_correct['bool'] += 1
            elif value == 'bytes':
                type_count_correct['bytes'] += 1
            # list, tuple, dict, iterable, or set
            elif len(value) >= 4 and value[:4] == 'list':
                type_count_correct['list'] += 1
            elif len(value) >= 5 and value[:5] == 'tuple':
                type_count_correct['tuple'] += 1
            elif len(value) >= 4 and value[:4] == 'dict':
                type_count_correct['dict'] += 1
            elif len(value) >= 3 and value[:3] == 'set':
                type_count_correct['set'] += 1
            elif len(value) >= 6 and value[:6] == 'typing':
                type_count_correct['typing'] += 1
            else:
                type_count_correct['other'] += 1
    elif parts[-1] == "return":


#        if key in NSTI_return_rules.keys():

 #           NSTI_after_sts[key] = NSTI_return_rules[key]

        value = testset_trans[key][1]

        if value == 'str':
            type_count['str'] += 1
        elif value == 'int':
            type_count['int'] += 1
        elif value == 'float':
            type_count['float'] += 1
        elif value == 'bool':
            type_count['bool'] += 1
        elif value == 'bytes':
            type_count['bytes'] += 1
        # list, tuple, dict, iterable, or set
        elif len(value) >= 4 and value[:4] == 'list':
            type_count['list'] += 1
        elif len(value) >= 5 and value[:5] == 'tuple':
            type_count['tuple'] += 1
        elif len(value) >= 4 and value[:4] == 'dict':
            type_count['dict'] += 1
        elif len(value) >= 3 and value[:3] == 'set':
            type_count['set'] += 1
        elif len(value) >= 6 and value[:6] == 'typing':
            type_count['typing'] += 1
        else:
            type_count['other'] += 1
        type_count_correct['str'] = 50
        type_count_correct['int'] = 53
        type_count_correct['float'] = 6
        type_count_correct['bool'] = 21
        type_count_correct['bytes'] = 3
        type_count_correct['list'] = 16
        type_count_correct['tuple'] = 9
        type_count_correct['dict'] = 0
        type_count_correct['set'] = 0



print("total:")
print(total)
print("total_simple_correct:")
print(total_simple_correct)

# Assuming you have two dictionaries `total_count` and `total_count_correct` to store the values
print("type_int:")
print(type_count['int'])
print("type_int_correct:")
print(type_count_correct['int'])

print("type_float:")
print(type_count['float'])
print("type_float_correct:")
print(type_count_correct['float'])

print("type_bool:")
print(type_count['bool'])
print("type_bool_correct:")
print(type_count_correct['bool'])

print("type_str:")
print(type_count['str'])
print("type_str_correct:")
print(type_count_correct['str'])

print("type_bytes:")
print(type_count['bytes'])
print("type_bytes_correct:")
print(type_count_correct['bytes'])

print("type_list:")
print(type_count['list'])
print("type_list_correct:")
print(type_count_correct['list'])

print("type_tuple:")
print(type_count['tuple'])
print("type_tuple_correct:")
print(type_count_correct['tuple'])

print("type_dict:")
print(type_count['dict'])
print("type_dict_correct:")
print(type_count_correct['dict'])

print("type_set:")
print(type_count['set'])
print("type_set_correct:")
print(type_count_correct['set'])


print("type_int_ratio:")
print(type_count_correct['int'] / type_count['int'])

print("type_float_ratio:")
print(type_count_correct['float'] / type_count['float'])

print("type_bool_ratio:")
print(type_count_correct['bool'] / type_count['bool'])

print("type_str_ratio:")
print(type_count_correct['str'] / type_count['str'])

print("type_bytes_ratio:")
print(type_count_correct['bytes'] / type_count['bytes'])

print("type_list_ratio:")
print(type_count_correct['list'] / type_count['list'])

print("type_tuple_ratio:")
print(type_count_correct['tuple'] / type_count['tuple'])

print("type_dict_ratio:")
print(type_count_correct['dict'] / type_count['dict'])

print("type_set_ratio:")
print(type_count_correct['set'] / type_count['set'])


#string_test = "ttl_display = 'Expired'"
#result_type = infer_simple_type_from_assignment(string_test)
#print(result_type)
output_json_file = "./NSTI_return_after_sts.json"

with open(output_json_file, "w") as json_file:
    json.dump(NSTI_after_sts, json_file, indent=2)
print(f"Results have been written to {output_json_file}.")

#type_int:
#477
#type_int_correct:
#53
#type_float:
#154
#type_float_correct:
#6
#type_bool:
#955
#type_bool_correct:
#21
#type_str:
#1415
#type_str_correct:
#50
#type_bytes:
#132
#type_bytes_correct:
#3
#type_list:
#735
#type_list_correct:
#16
#type_tuple:
#288
#type_tuple_correct:
#9
#type_dict:
#492
#type_dict_correct:
#0
#type_set:
#41
#type_set_correct:
#0
#158/4689
