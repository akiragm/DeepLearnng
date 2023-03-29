cd C:\Users\t0915526\Downloads
#whlファイル
python -m pip install --no-deps gensim-4.3.1-cp310-abi3-win_amd64.whl
python -m pip install --no-deps gensim-4.3.1-cp310-cp310-macosx_11_0_arm64.whl
python -m pip install --no-deps gensim-4.3.1-cp38-cp38-macosx_11_0_arm64.whl

pip install --no-deps pip-23.0.1.tar.gz --use-deprecated=legacy-resolver



#pip cp確認コマンド
from pip._internal.utils.compatibility_tags import get_supported
get_supported()



#バージョンが一緒なのにエラーの場合
#numpy?1.11.2+mkl?cp35?cp35m?win_amd64.whl　というファイル名だった場合，
#numpy?1.11.2+mkl?cp35?none?win_amd64.whl に変更する
対応cpに変更する。cpは同一であれば、その他は変更成功すること実証済み