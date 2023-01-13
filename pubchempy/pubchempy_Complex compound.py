import pubchempy as pcp
import pandas as pd

#情報を取得したい化合物のCAS番号のリストと取得したい情報のリストを定義
properties = ['MolecularFormula','CanonicalSMILES', 'IUPACName']
cas_list = ['alanine','silicone','germanium','tungsten','1002-89-7', '999-9-9']

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

#detaframeを出力
print(df)

