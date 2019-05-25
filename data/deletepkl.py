# -*- encoding:utf-8 -*-
# delete all files with the pkl suffix

import os

path = "."
g = os.walk(path)
for path, dir_list, file_list in g:
    for file in file_list:
        if file.endswith(".pkl"):
            temp_file = os.path.join(path, file)
            print("file: [{}] is deleted".format(temp_file))
            os.remove(temp_file)

