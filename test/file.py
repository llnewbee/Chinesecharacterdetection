import os
import json

folder_path = "/Users/ymhave/Downloads/file"

# 支持的图片扩展名（小写）
image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

files = os.listdir(folder_path)
files.sort()  # 按文件名排序

d={}

for filename in files:
    ext = os.path.splitext(filename)[1].lower()
    if ext in image_extensions:
        full_path = os.path.join(folder_path, filename)
        if os.path.isfile(full_path):
            d[filename]=full_path
            print(full_path)
file_dir = "/Users/ymhave/Downloads/1"
file_name = 'test'
file_path=file_dir+'/'+file_name+'.json'
 
# 使用with语句打开文件，并写入json
with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(d, json_file, ensure_ascii=False, indent=4)
import code
code.interact(local=locals())