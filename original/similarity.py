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



##読み込み用データの作成
f = open('mydatafile.txt', 'r')
text = f.read()
f.close()

##コーパス作成
corpus, word_to_id, id_to_word = preprocess(text)
params = {}
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word

#pickle作成
pkl_file = 'ptb.vocab.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)

print(word_to_id)
print(id_to_word)
