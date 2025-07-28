# Chinese character detection

汉字检测与评分系统

```
Chinesecharacterdetection
         ├── HAVE.py  # 所有模块整合的自动化python程序，输出例可见test.zip
         ├── README.md # 自述文件
         └── test # 分离的模块文件
             ├── _core_i_mark.py # 已经舍弃的重心检测模块
             ├── _cut.py # 已经舍弃（旧版）的作品图片裁剪模块
             ├── _outport_debug_overlay.py # Procrustes距离计算模块 degbug版本
             ├── badapple # 默认文件
             ├── core_mark.py # 批量识别出汉字的重心并计算出平均差异值的模块
             ├── corner_mark.py # 批量识别角点的模块
             ├── cut_plus.py # 作品图像透视变化和裁剪模块（预处理模块）
             ├── file.py # 识别文件夹内的所有图片并保存为带有路径的字典模块
             ├── main.py # 最终分数计算模块
             ├── outport_code.py # Procrustes距离计算模块
             └── read.py # 读取json字典模块
```

## 依赖

本程序 HAVE.py 依赖于以下的第三方库

- [numpy](https://numpy.org/)      （数值运算库）

- [opencv-python](https://opencv.org/)  （图像处理，导入名为 `cv2`）

- [Pillow](https://python-pillow.org/)  （图像处理，导入名为 `PIL`）

- [pandas](https://pandas.pydata.org/)  （数据分析）

- [matplotlib](https://matplotlib.org/) （绘图库）

- [fastdtw](https://pypi.org/project/fastdtw/) （动态时间规整算法）

- [scipy](https://scipy.org/)      （科学计算，含 `scipy.spatial`）

  **一键安装命令示例**

```
pip install numpy opencv-python Pillow pandas matplotlib fastdtw scipy
```

## 使用例

* 对于实验的输出例 可见保存在百度网盘的 [test.zip](https://pan.baidu.com/s/1UutECyrJsRjAA7OfpkCDrA?pwd=7bs4)

需要在约 631 行处修改数据

**例：**

```
or_work_path = '～/test/work' #原始待评价作品目录
or_expert_path = '～/test/expert' #原始专家作品目录
path = '～/test/down' #工作目录(保存文件)
N_ROWS = 11   #规定字帖行数
N_COLS = 8   #规定字帖列数
```

所有数据都会输出至工作目录中，其中：

- score.json 是计算出的所有得分和每个作品最终得分组成的字典
- final.csv / final.json 是最终得分前30%的作品和分数
- plot.png 是所有最终得分的箱形图分析
- work/expert 文件夹是对两组作品的数据化处理存放处
  - cuting 文件夹内是每个作品经过裁切后的文字图像
  - json 文件夹内是每个作品经过数据化的角点点集字典
- json 文件夹是最终得到的参赛作品与专家作品逐一对比的数据
  - 文件夹命名规则：参与比较的专家作品原文件名_参与比较的参赛作品原文件名
    - coreall.json 记录的是两个作品每个汉字之间的重心差距以及平均差距
    - metrics.json / dtw_procrustes_scores.csv 记录的是四种不同的算法得出的角点点集平均差异度

**原始作品目录要求**：常规的图片格式存放于作品目录，不宜添加其他杂项目。作品要求拍摄清晰，无明显污损、裁切、歪斜等，且所有作品统一格式、内容等要素