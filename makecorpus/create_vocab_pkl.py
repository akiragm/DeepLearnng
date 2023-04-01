# coding: utf-8
import sys
sys.path.append('..')
from common import config
# GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
# ===============================================
# config.GPU = True
# ===============================================
from common.util import preprocess
import pickle


path = 'C:\Git\DeepLearnng\makecorpus\mainPD_modulePD_TEST .txt'

##読み込み用データの作成
f = open(path, 'r', encoding="utf-8")
text = f.read()
f.close()


##コーパス作成
corpus, word_to_id, id_to_word = preprocess(text)
params = {}
params= id_to_word
params= word_to_id 
params= corpus

#pickle作成
pkl_file = 'ptb.vocab.pkl'
with open(pkl_file, 'wb') as f:
 #  pickle.dump(params, f, -1)
  pickle.dump((word_to_id,id_to_word), f)

