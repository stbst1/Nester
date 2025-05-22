import json
import json
import re

# Read data from the specified JSON file
resfile = "E:\\gaocunyun\\TypeGen\\data\\testset_source.json"
with open(resfile, 'r', encoding='utf-8') as file:
    results = json.load(file)



for r in results:
    # Extracting file name, location, name, and scope from the key
    split_info = r.rsplit(".py", 1)
    r_dict = {
        "file": split_info[0] + ".py",
        "loc": split_info[1][2:].split("--")[0],
        "name": split_info[1][2:].split("--")[1],
        "scope": split_info[1][2:].split("--")[2]
    }
    split_info = r.rsplit("--", 2)
    scope = split_info[2]

    if isinstance(results[r], list):
        code_lines = results[r]
    else:
        code_lines = results[r].splitlines() if isinstance(results[r], str) else [str(results[r])]

    updated_lines = []
    return_count = 0

    for line in code_lines:
        line = line.rstrip()  # 移除行尾的空白字符

        if "local" in scope:
            # 检查并替换指定的变量赋值
            name_assignment_pattern = rf'^(\s*){re.escape(r_dict["name"])}\s*=\s*(.*)$'
            line = re.sub(name_assignment_pattern, rf'\1{r_dict["name"]} = Assignment_Analysis(\2)', line)

            # 检查并替换 if/elif 语句（统一替换成 If_Analysis）
            if_elif_pattern = r'^\s*(if|elif)\s+(.+?):'
            line = re.sub(if_elif_pattern, r'If_Analysis(\2):', line)

        elif "arg" in scope:
            if line.strip().startswith("def "):
                line = re.sub(r'^(\s*def\s+\w+\s*\(.*\)):', rf'Function_Analysis(\1):', line)
            elif re.match(r'^\s*(if|elif)\s+', line):  # 匹配 if 或 elif
                line = re.sub(r'^\s*(if|elif)\s+(.+?):', r'If_Analysis(\2):', line)
            else:
                line = f'Argument_Analysis({line.strip()})'

        elif "return" in scope:
            if line.strip().startswith("return "):
                return_count += 1
                return_var = f"return_type{return_count}"
                line = re.sub(r'^\s*return\s+(.*)', rf'return {return_var} = Return_Analysis(\1)', line)
            if re.match(r'^\s*(if|elif)\s+', line):  # 匹配 if 或 elif
                line = re.sub(r'^\s*(if|elif)\s+(.+?):', r'If_Analysis(\2):', line)

        # 检查该行是否包含任何分析标记，如果没有则跳过（相当于删除）
        analysis_keywords = ["Assignment_Analysis", "If_Analysis", "Function_Analysis", "Argument_Analysis", "Return_Analysis"]
        if any(keyword in line for keyword in analysis_keywords):
            updated_lines.append(line)

    if "return" in scope and return_count > 0:
        combine_call = f"Combine({', '.join([f'return_type{i+1}' for i in range(return_count)])})"
        updated_lines.append(combine_call)

    results[r] = "\n".join(updated_lines)



with open('high_level_right.json', 'w', encoding='utf-8') as outfile:
    json.dump(results, outfile, ensure_ascii=False, indent=4)
