import json

# 假设 json 文件路径
core_json_path = '/Users/ymhave/Documents/归档/数学建模/Expert_Work.json'
corncer_json_path = "/Users/ymhave/Documents/归档/数学建模/metrics.json"

with open(core_json_path, "r", encoding="utf-8") as f:
    data_core = json.load(f)  # 将 JSON 文件内容转换为 Python 字典或列表等对象

with open(corncer_json_path, "r", encoding="utf-8") as f:
    data_cor = json.load(f)  # 将 JSON 文件内容转换为 Python 字典或列表等对象
    
# 现在 data 是个字典（或列表），例如访问字典特定键：
if data_core['average_centroid_distance']==0:
    core=0
else:
    core=data_core["average_centroid_distance"]/100
cor=data_cor["avg_procrustes_weighted"]

de_average=core*0.5+cor*0.5
score=(1-de_average)*100
print(f'{score}%')