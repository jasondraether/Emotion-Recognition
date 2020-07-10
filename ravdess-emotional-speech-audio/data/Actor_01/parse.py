import os
import re

file_list = os.listdir(".")

for fname in file_list:
    res = re.findall("
