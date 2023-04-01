from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# データを読み込む
data = ...
X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2, random_state=42)

# モデルのパラメータの範囲を定義する
params = {
    'size': Integer(50, 300),
    'window': Integer(2, 10),
    'min_count': Integer(1, 5),
    'sg': Integer(0, 1),
    'hs': Integer(0, 1),
    'negative': Integer(5, 20),
    'alpha': Real(0.01, 0.1),
    'min_alpha': Real(0.0001, 0.01),
}

# CBOWのモデルを定義する
def cbow_model(size, window, min_count, sg, hs, negative, alpha, min_alpha):
    model = Word2Vec(
        sg=sg,
        hs=hs,
        negative=negative,
        alpha=alpha,
        min_alpha=min_alpha,
        size=size,
        window=window,
        min_count=min_count,
        workers=4
    )
    model.build_vocab(X_train)
    model.train(X_train, total_examples=len(X_train), epochs=10)
    y_pred = [model.wv.similarity(w1, w2) for w1, w2 in X_test]
    return -mean_squared_error(y_test, y_pred)  # 最適化目的は最小二乗誤差の最大化

# モデルのパラメータを最適化する
search = BayesSearchCV(
    cbow_model,
    params,
    n_iter=20,  # 試行回数
    cv=5,  # 交差検証の分割数
    n_jobs=-1
)
search.fit(data)
print(search.best_params_)  # 最適なパラメータを表示する
