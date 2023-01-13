import pubchempy as pcp

#取得したい情報のリストを定義
properties = ['MolecularFormula', 'MolecularWeight', 'CanonicalSMILES', 'IUPACName']

#CAS番号から情報を取得する
result = pcp.get_properties(properties, 'tungsten', 'name')

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
