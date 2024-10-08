def filter_lines(input_file, output_file, keyword):
    with open(input_file, 'r', encoding='utf-8') as f_input, \
         open(output_file, 'w', encoding='utf-8') as f_output:
        for line in f_input:
            #print(line)
            #exit(1)
            if keyword in line:
                f_output.write(line)

# 使用示例
input_file = './differ.txt'  # 输入文件名
output_file = './differ_Typegen_yuanban.txt'  # 输出文件名
keyword = '--local'  # 要删除的关键字
filter_lines(input_file, output_file, keyword)
print("已删除包含关键字的行，并保存到 output.txt 中。")