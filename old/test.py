import os
from ctypes import *

p = os.getcwd() + '/test.so'  #表示.so文件的绝对路径，如果你没在当前路径打开python则可能需要修改
f = CDLL(p)  #读取.so文件并赋给变量f

# 变量为整数
a = c_float(-60.0)
f.lif.restype = c_float # #指定函数的返回值类型
print(f.lif(a))
print(f.add(1, 2))
