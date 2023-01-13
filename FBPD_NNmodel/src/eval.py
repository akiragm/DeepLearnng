# coding: utf-8
import sys
sys.path.append('..')
from common.util import preprocess,most_similar, analogy,sentence_similarity,sentence_preprocess
import pickle

pkl_file = 'cbow_params.pkl'
#pkl_file = 'skipgram_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

### most similar task
##querys = ['y5@cs-we063', 'y3@cs-z7075', 'y3@csp1dq12', 'dummy']
##for query in querys:
##    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
##
### analogy task
##print('-'*50)
##analogy('y5@cs-we063', 'y3@cs-z7075', 'queen',  word_to_id, id_to_word, word_vecs)
##analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)
##analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)
##analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs)
##
##
#sentence  simmility task
 #特徴ベクトル数
num_features = 300
sentence_1 =" Y5@CS-WE063 17 Branch Sub 0"
sentence_2 ="AL@E00FLI66 00 Branch Sub 0"

#sentence_preprocess
 #presentence1 ['Y5@CS-WE063':0,'17':1,'Branch':2,'Sub':3,'0':4]
sentence_preprocess(sentence_1)
sentence1_word_to_id = sentence_word_to_id
sentence1_id_to_word = sentence_id_to_word
 #presentence2 ['AL@E00FLI66':0,'00':1,'Branch':2,'Sub':3,'0':4]
sentence_preprocess(sentence_2)
sentence2_id_to_word = sentence_word_to_id
sentence2_id_to_word = sentence_id_to_word


#
#result = sentence_similarity(sentence_presentence1, sentence_presentence2, word_to_id, id_to_word, word_vecs, num_features)
#print(result)
####sentence_similarity
     if sentence_presentence1 not in word_to_id:
        print('%s is not found' % sentence_presentence1)
        return
    if sentence_presentence2 not in word_to_id:
        print('%s is not found' % sentence_presentence2)
        return

    print('\n[sentence_presentence1] ' + sentence_presentence1)
    ##0からsentenceの各wordに対してword_to_idからベクトルを取得し、sentence_vecに加算
    for i in range(sentence_presentence1)
        sentence_id = word_to_id[sentence_1[i]]
        sentence1_vec += word_matrix[sentence_id]

    
    for i in range(sentence_presentence2)
        sentence_id = word_to_id[sentence_2[i]]
        sentence2_vec += word_matrix[sentence_id]


    sentence1_vocab_size = len(sentence1_id_to_word)
    sentence1_vec = sentence1_vec/sentence1_vocab_size

    sentence2_vocab_size = len(id_to_word)
    sentence2_vec = sentence1_vec/sentence1_vocab_size
    
##cos類似度により比較################################################################    
    similarity1 = np.zeros(sentence1_vocab_size)
    similarity2 = np.zeros(sentence2_vocab_size)
    for i in range(vocab_size):
        similarity1[i] = cos_similarity(sentence1_vec , sentence2_vec) ←ここから確認
        ##cos_similarity
        #
        #   '''コサイン類似度の算出
        #   :param x: ベクトル  ←sentence1_vec
        #   :param y: ベクトル　←sentence2_vec
        #   :param eps: ”0割り”防止のための微小値
        #   :return:
        #  '''
           nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
           ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
           return np.dot(nx, ny)
        #
        ##
    count = 0
    
##類似度が格納されたsimilarity配列をソートし、topを取得する
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return
