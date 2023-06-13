# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:54:59 2023

@author: t0915526
"""

import pandas as pd

df = pd.DataFrame({
    'city': ['Tokyo', 'Tokyo', 'Osaka', 'Osaka', 'Fukuoka', 'Fukuoka'],
    'temp': [25, 28, None, 30, None, 32],
    'humidity': [50, 60, 55, 65, None, 70]
})



def fill_missing_data(df, col_to_fill, ref_cols, fill_method='mean'):
    """
    指定されたカラムに対する欠損値を参照カラムを使って補完する。

    Parameters
    ----------
    df : pandas.DataFrame
        欠損値を補完するDataFrame。
    col_to_fill : str
        欠損値を補完する対象のカラム名。
    ref_cols : list of str
        参照カラム名のリスト。
    fill_method : str, optional
        補完方法。'mean'（平均値）または'median'（中央値）を指定できる。
        デフォルトは'mean'。

    Returns
    -------
    pandas.DataFrame
        欠損値を補完したDataFrame。
    """

    # 欠損値を持つ行を取得
    missing_rows = df[df[col_to_fill].isnull()]

    # 参照カラムが同一の行をグループ化
    grouped = df.groupby(ref_cols)

    # 参照カラムが同一の行の対象カラムの平均値または中央値を算出
    if fill_method == 'mean':
        fill_value = grouped[col_to_fill].mean()
    elif fill_method == 'median':
        fill_value = grouped[col_to_fill].median()
    else:
        raise ValueError(f"Unsupported fill method '{fill_method}'.")

    # 欠損値を持つ行の参照カラムを結合
    merged = pd.merge(missing_rows[ref_cols], fill_value, on=ref_cols, how='left')

    # 欠損値を補完
    merged.columns = list(ref_cols) + [col_to_fill]
    df = pd.merge(df, merged, on=ref_cols, how='left')
    df[col_to_fill] = df[col_to_fill].fillna(df[col_to_fill+'_y'])
    df = df.drop([col_to_fill+'_x', col_to_fill+'_y'], axis=1)

    return df


df2 = fill_missing_data(df, 'temp', ['city'], fill_method='mean')

print(df)

