cd C:\Users\t0915526\Downloads
#whlファイル
python -m pip install --no-deps wxPython-4.2.0-cp38-abi3-win_amd64.whl

pip install --no-deps wxPython-4.2.0.tar.gz --use-deprecated=legacy-resolver



#pip cp確認コマンド
from pip._internal.utils.compatibility_tags import get_supported
get_supported()


#バージョンが一緒なのにエラーの場合
#numpy?1.11.2+mkl?cp35?cp35m?win_amd64.whl　というファイル名だった場合，
#numpy?1.11.2+mkl?cp35?none?win_amd64.whl に変更する