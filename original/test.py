# coding: utf-8
import sys
sys.path.append('..')
from common import config
# GPUで実行する場合は、下記のコメントアウトを消去（要cupy）
# ===============================================
# config.GPU = True
# ===============================================
from common.np import *
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram

from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb
import numpy
import time
import matplotlib.pyplot as plt
from common.np import *  # import numpy as np
from common.util import clip_grads


# ハイパーパラメータの設定
window_size = 1
hidden_size = 1
batch_size = 1
max_epoch = 10

# データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# モデルなどの生成
#model = CBOW(vocab_size, hidden_size, window_size, corpus)
model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

#print(target)
# 学習開始
#trainer.fit(contexts, target, max_epoch, batch_size)
#def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
x = contexts
t = target
max_epoch=10
batch_size=32
max_grad=None
eval_interval=20

##datasizeの配列をidxとして使用する。
##indexErrorはdatasizeの値によるもの
print("datasize")
print(len(x))
data_size = len(x)
max_iters = data_size // batch_size
#self.eval_interval = eval_interval
#model, optimizer = self.model, self.optimizer
total_loss = 0
loss_count = 0

start_time = time.time()
for epoch in range(max_epoch):
# シャッフル
    idx = numpy.random.permutation(numpy.arange(data_size))
    
    x = x[idx]
    t = t[idx]
    print("x")
    print(len(x))

    for iters in range(max_iters): 
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]
        print("batch_t")
        print(len(batch_t))
        # 勾配を求め、パラメータを更新
 ####################################################

        loss = model.forward(batch_x, batch_t)
 ####################################################
## def forward(self, contexts, target):
##        h = self.in_layer.forward(target)
##
##        loss = 0
##        for i, layer in enumerate(self.loss_layers):
##            loss += layer.forward(h, contexts[:, i])



  ####################################################       
##        model.backward()
##        params, grads = remove_duplicate(model.params, model.grads)  # 共有された重みを1つに集約
##        if max_grad is not zNone:
##            clip_grads(grads, max_grad)
##        optimizer.update(params, grads)
##        total_loss += loss
##        loss_count += 1
##
##        # 評価
##        if (eval_interval is not None) and (iters % eval_interval) == 0:
##            avg_loss = total_loss / loss_count
##            elapsed_time = time.time() - start_time
##            print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
##                % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
##            self.loss_list.append(float(avg_loss))
##            total_loss, loss_count = 0, 0
##        self.current_epoch += 1

#
#
#
