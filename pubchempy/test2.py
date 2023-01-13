#モジュールインポート
import re
import pubchempy as pcp
import pandas as pd
import chardet

#取得したい情報のリストを定義
properties = ['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES', 'IUPACName']

#CAS番号から情報を取得する
result = pcp.get_properties(properties, '64-17-5', 'name')

#化合物ごとにレコードを記録した辞書のリストが返される。
print(result)
"""
output
[{'CID': 702,
  'MolecularFormula': 'C2H6O',
  'MolecularWeight': 46.07,
  'CanonicalSMILES': 'CCO',
  'IUPACName': 'ethanol'}]
"""
#複数化合物情報の取得
#リストに格納された複数のCAS番から化合物情報を取得してpandasのデータフレームに格納する。


import pubchempy as pcp
import pandas as pd

#情報を取得したい化合物のCAS番号のリストと取得したい情報のリストを定義
properties = ['MolecularFormula','CanonicalSMILES', 'IUPACName']
cas_list = ['100-21-0','1002-62-6','10028-15-6','1002-84-2','1002-89-7', '999-9-9']

#pubchem に情報のない化合物はエラーとなるので、エラーが発生しても途中で終了させずに処理を継続できるようにtry文を使用
df = pd.DataFrame()
for cas in cas_list:
    try:
        temp = pcp.get_properties(properties, cas, 'name',as_dataframe=True)
        #as_dataframe=True とすることでpandas dataframe として情報を取得可能
        temp['CAS'] = cas
        df = pd.concat([df,temp],axis=0,join='outer',sort=True)
    except:
        pass
