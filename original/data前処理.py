# ## コーパスの前処理
# ・ファイルから文章データを取り出す
# ・文章ごとに単語に分割する
# In[1]:


import re
import pickle

import warnings
warnings.simplefilter('ignore')

#  テキストファイルの読み込み(なお、前処理は考慮していない)
#  この例では、kaggleにある英語のデータで試しています。
with open("ptb.test_original.txt", mode="r", encoding="utf-8") as f:  
    wagahai = f.read()


seperator = "\n"  # このサンプルは、改行で文章を区切っているのでセパレータを改行コードにした。日本語だと「。」にすることが多い
words_list = wagahai.split(seperator)  # セパレーターを使って文章をリストに分割する
#words_list.pop() # 最後の要素は空の文字列になるので、削除
#words_list = [x+seperator for x in words_list]  # 文章の最後に。を追加


words = []
for sentence in words_list:
    data = sentence.split() #英語のため空白で分割する。日本語の場合は、形態素の分けます。
    words.append(data)   # 文章ごとに単語に分割し、リストに格納
    #words.append(t.tokenize(sentence, wakati=True))  日本語の場合はコチラ
    
#ここで、pickleに保存します。
with open('PTB_words.pkl', mode='wb') as f:  # pickleに保存
    pickle.dump(words, f)
