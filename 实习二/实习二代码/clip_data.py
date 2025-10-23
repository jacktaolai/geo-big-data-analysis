# 将0.1的数据划分为验证集,并存为csv文件,同时从原数据里删除这一部分
import pandas as pd

import os

# 指定运行环境为源文件所在目录
os.chdir(os.path.dirname(__file__))
df=pd.read_csv('../实习二数据/train.csv')
print("数据集大小为：",len(df))

df_val=df.sample(frac=0.1)
input("会删除10%的数据！请按回车确认...")
df_val.to_csv('../实习二数据/test.csv',index=False)
df_train=df.drop(df_val.index)
df_train.to_csv('../实习二数据/train.csv',index=False)
print("划分完成")