import fasttext

# Skipgram model
model = fasttext.skipgram('../../resource/THUCNews_ch/t1_cut_words_cnews.train.txt', 'model')
print(model.words) # list of words in dictionary

# CBOW model
model = fasttext.cbow('../../resource/THUCNews_ch/t1_cut_words_cnews.train.txt', 'model')
print(model.words) # list of words in dictionary