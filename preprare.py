import json
import os
from tqdm import tqdm
from hityper.typeobject import TypeObject
import re
# It is assumed that functions such as transform_sample_to_top and other necessary functions
# have already been defined in a module from which they are imported here
# from your_module import transform_sample_to_top, match_type_for_cot

# Load the dataset
with open(os.path.join("./data", "testset_transformed.json")) as f:
    testset_trans = json.load(f)

# Load the results data
with open("./NSTI_return_preprocessed.json") as f:
    results = json.load(f)

# Initialize counters
total_local_str = 0
correct_local_str = 0

# Define a regular expression function to match types for Chain of Thought processing
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

# Iterate over the dataset
for key, value in tqdm(testset_trans.items()):
    parts = key.split('--')
    # Check if the variable is local
    #if parts[-1] == "return":
    gttype = TypeObject.Str2Obj(value[1])  # Assuming the second element represents the actual type
    # Check if the actual type is 'str'
    #print(testset_trans[key][2] == "user-defined")

    #if value[1] == 'bool':
    #if testset_trans[key][2] == "user-defined":
    try:
        if results[key] != []:
            if "`typing.Union[" in results[key][0]:

                total_local_str += 1

                if key in results:
                    # Process
