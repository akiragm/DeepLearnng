##2022/12 TAKE AKI
import sys
import numpy as np 

##読み込み用データの作成
f = open('TecnologyFile.txt', 'r', encoding='UTF-8')

data = f.read()

dataA = np.array(data);

print(dataA)

f.close()
