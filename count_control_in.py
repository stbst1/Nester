import json
with open("./data/testset_staticsliced_hop3.json") as f:
    data = json.load(f)

with open("./data/testset_transformed.json") as f:
    simple_data = json.load(f)

with open("./predictions_codellama_qudiaozhahu_instruct.json") as f:
    pred = json.load(f)



def count_occurrences(main_string, sub_string):
    count = 0
    start_index = 0

    while True:
        # Find the next occurrence of sub_string starting from start_index
        index = main_string.find(sub_string, start_index)

        # If sub_string is found
        if index != -1:
            # Increment count
            count += 1
            # Update start_index to search for next occurrence
            start_index = index + 1
        else:
            # If sub_string is not found, break the loop
            break

    return count
total = 0
total_dayuer = 0
zero = 0

new_pred = {}

for r, value in data.items():
    r1 = r
    zero = zero + 1
    #if zero == 1000:
    #    break;
    # 使用split方法按照"--"分割字符串，得到各个部分的信息
    split_info = r.rsplit(".py", 1)
    # 将分割后的信息分别赋值给r字典的各个键
    r = {
        "file": split_info[0] + ".py",
        "loc": split_info[1][2:].split("--")[0],
        "name": split_info[1][2:].split("--")[1],
        "scope": split_info[1][2:].split("--")[2]
    }
    key = '{}--{}--{}--{}'.format(r["file"], r["loc"], r["name"], r["scope"])
    name = r["name"]
    scope = r["scope"]
    loc = r["loc"]
    filename = r["file"]
    for r_pred, value_pred in pred.items():
        if r_pred == r1:
            occurrences = count_occurrences(value, " if ")
            if occurrences >= 1:
                new_pred[r1] = pred[r1]

with open("new_if_predictions.json", "w") as f:
    json.dump(new_pred, f, indent=4)
