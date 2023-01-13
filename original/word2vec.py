#!/usr/bin/env python
# coding: utf-8

# # word2vecによる分散表現と２次元グラフに単語のプロットする
# 
# ・word2vecによる分散表現に変換する
# ・分散表現をした単語をPCAを使って２次元に圧縮する
# ・圧縮した分散表現を２次元の図にプロットする
# ・参考にした参考動画(Udemy自然言語処理とチャットボット: AIによる文章生成と会話エンジン開発講座より「word2vecによる分散表現」)
# https://www.udemy.com/course/ai-nlp-bot/?deal_code=JPA8DEAL2PERCENTAGE&aEightID=s00000016735001
# 

# ## コーパスの前処理
# ・ファイルから文章データを取り出す
# ・文章ごとに単語に分割する

# In[1]:


import re
import pickle
from janome.tokenizer import Tokenizer
import warnings
warnings.simplefilter('ignore')

#  テキストファイルの読み込み(なお、前処理は考慮していない)
#  この例では、kaggleにある英語のデータで試しています。
with open("ptb.test_original.txt", mode="r", encoding="utf-8") as f:  
    wagahai = f.read()


seperator = "\n"  # このサンプルは、改行で文章を区切っているのでセパレータを改行コードにした。日本語だと「。」にすることが多い
wagahai_list = wagahai.split(seperator)  # セパレーターを使って文章をリストに分割する
#wagahai_list.pop() # 最後の要素は空の文字列になるので、削除
#wagahai_list = [x+seperator for x in wagahai_list]  # 文章の最後に。を追加
        
t = Tokenizer()


wagahai_words = []
for sentence in wagahai_list:
    data = sentence.split() #英語のため空白で分割する。日本語の場合は、形態素の分けます。
    wagahai_words.append(data)   # 文章ごとに単語に分割し、リストに格納
    #wagahai_words.append(t.tokenize(sentence, wakati=True))  日本語の場合はコチラ
    
#ここで、pickleに保存します。
with open('wagahai_words.pickle', mode='wb') as f:  # pickleに保存
    pickle.dump(wagahai_words, f)


# ## gensimのword2vecを用いた学習をする
# 今回はword2vecのためにライブラリgensimを使います。  
# 
# gensimについては、以下を参照 
# https://radimrehurek.com/gensim/
# 
# 以下では、word2vecを用いてコーパスの学習を行い、学習済みのモデルを作成します。

# In[2]:


from gensim.models import word2vec

with open('wagahai_words.pickle', mode='rb') as f:
    wagahai_words = pickle.load(f)
    
    
# size : 中間層のニューロン数・数値に応じて配列の大きさが変わる。数値が多いほど精度が良くなりやすいが、処理が重くなる。
# min_count : この値以下の出現回数の単語を無視
# window : 対象単語を中心とした前後の単語数
# iter : epochs数
# sg : skip-gramを使うかどうか 0:CBOW 1:skip-gram
model = word2vec.Word2Vec(wagahai_words,
                          size=200,
                          min_count=5,
                          window=5,
                          iter=20,
                          sg = 0)


# In[3]:


#学習結果を確認します
print(model.wv.vectors.shape)  # 分散表現の形状　Word2Vecのsizeの設定が反映されているのがわかります
print(model.wv.vectors)  #実際の分散表現、size次元の配列になっていることがわかるかと思います。


# In[4]:


print(len(model.wv.index2word))  # 語彙の数
print(model.wv.index2word[:10])  # 最初の10単語を表示します。頻出単語の順番で出力されるようです。


# In[5]:


#print(model.wv.vectors[0])  # 最初のベクトル
#print(model.wv.__getitem__("maskhttp"))  # 最初の単語「の」のベクトル


# In[6]:


#類似している(ベクトルの距離的に近い)単語を出力してみる
model.most_similar(positive=['earthquake'], topn=20)


# ## 単語を２次元の図にプロットします
# ・PCAを用いてN次元の分散表現を2次元に圧縮する
# ・２次元に圧縮した分散表現を２次元の図にプロットします

# In[8]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#プロットしたい単語を設定する
#FIreなどの災害の意味で使われそうなものを設定
words = []
words.append(["fire","r"])
words.append(["earthquake","r"])
words.append(["accident","r"])
words.append(["involving","r"])
words.append(["Hawaii","r"])
words.append(["Africa","c"])
words.append(["San","c"])
words.append(["Francisco","c"])
words.append(["maskusername","c"])

#頻出単語もセットする
print(model.wv.index2word[:50])
for s in model.wv.index2word[:50]:
    words.append([s,"b"])


length = len(words)
data = []
 
j = 0
while j < length:
    data.append(model[words[j][0]])
    j += 1
    

#主成分分析により２次元に圧縮する
pca = PCA(n_components=2)
pca.fit(data)
data_pca= pca.transform(data)
 
length_data = len(data_pca)

#プロットの設定
#fig=plt.figure(figsize=(10,6),facecolor='w')
fig=plt.figure(figsize=(20,12),facecolor='w')

plt.rcParams["font.size"] = 10
i = 0
while i < length_data:
    #点プロット
    plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x", color=words[i][1])
 
    #文字プロット
    plt.annotate(words[i][0], (data_pca[i][0], data_pca[i][1]), size=12)
 
    i += 1

plt.show()


# ## プロットの結果
# プロットの結果、なんとなく近い単語が集まっている。
# 例えば、「was」「or」「a」「is」などのstopwordに設定されやすい単語が集まっているのがわかるかと思います。
