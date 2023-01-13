import pickle
objects = []
pkl_file1 = 'skipgram_params.pkl'
pkl_file2 = 'cbow_params.pkl'
with (open(pkl_file2, "rb")) as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']


#print(params)
#print(word_vecs[0])
print(id_to_word [4])
####################################################
