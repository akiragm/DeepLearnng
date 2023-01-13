# -*- coding: utf-8 -*-

import wx
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MyFileDropTarget(wx.FileDropTarget):
    def __init__(self, window):
        wx.FileDropTarget.__init__(self)
        self.window = window

    def OnDropFiles(self, x, y, filenames):
        # ファイルパスをテキストフィールドに表示
        for file in filenames:
            self.window.text.SetValue(file)
        f = np.load(''.join(filenames))#ファイルパスからファイルロード
        df_cmx = pd.DataFrame(f)#pandas変換
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cmx, annot=True)#seabornで混同行列表示
        plt.show()#可視化
        return True


class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Drop Target", size=(500, 200))
        p = wx.Panel(self)
        sizer = wx.BoxSizer(wx.HORIZONTAL)

        label = wx.StaticText(p, -1, "File name:")
        self.text = wx.TextCtrl(p, -1, "", size=(400, -1))
        sizer.Add(label, 0, wx.ALL, 5)
        sizer.Add(self.text, 0, wx.ALL, 5)
        p.SetSizer(sizer)

        dt = MyFileDropTarget(self)
        self.SetDropTarget(dt)
        self.Show()


if __name__ == '__main__':
    app = wx.App()
    MyFrame()
    app.MainLoop()
