import pickle
import numpy as np
objects = []
pkl_file = 'ptb.vocab_moto.pkl'
pkl_file1 = 'skipgram_params.pkl'
pkl_file2 = 'cbow_params.pkl'

with (open(pkl_file, "rb")) as f:
    params = pickle.load(f)
print(params[0])

#with (open(pkl_file2, "rb")) as f2:
#    params = pickle.load(f2)
#    word_vecs = params['word_vecs']
#    word_to_id = params['word_to_id']
#    id_to_word = params['id_to_word']
#
#print(word_to_id)
##
##n=0
##
##print(word_to_id[n])
##print(word_to_id[n+1])
##print(word_to_id[n+2])
##print(word_to_id[n+3])
##print(word_to_id[n+4])


###始まり要素
##id_ = 6
###要素数
##num =20
##
##x=np.empty(num,dtype=str)
##
##k = 0
##for id_ in range(num): 
##    x[k] = id_to_word[id_]
##    print('id_to_word[' + str(k)  +']:' + id_to_word [k])
##    k = k+1
##    
##
##print()

