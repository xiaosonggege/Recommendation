#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: data_processing
@time: 2020/8/7 6:51 下午
'''
import pandas as pd
import numpy as np

a = pd.DataFrame(data=np.arange(20).reshape(4, 5), index=[str(i)+'r' for i in range(4)],
                 columns=[str(i)+'c' for i in range(5)])
a.iloc[0, 0] = a.iloc[0, 1]
a.iloc[2, 2] = a.iloc[2, 3]
print(a)
b = a.drop_duplicates()
print(b)

if __name__ == '__main__':
    pass
