import pickle
objects = []
pkl_file0 = 'ptb.vocab.pkl'
pkl_file1 = 'skipgram_params.pkl'
pkl_file2 = 'cbow_params.pkl'
with (open(pkl_file0, "rb")) as f0:
    params = pickle.load(f0)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

print(data[])

##
##with (open(pkl_file2, "rb")) as f:
##    params = pickle.load(f)
##    word_vecs = params['word_vecs']
##    word_to_id = params['word_to_id']
##    id_to_word = params['id_to_word']
##
##
##print(objects)
##print(id_to_word [0])
##################################################
