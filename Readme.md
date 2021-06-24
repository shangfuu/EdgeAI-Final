# Edge AI Final

## 想要自行註冊人臉
如果要自己註冊人臉資料庫的話，需要在 register 資料夾內放置想要註冊的人臉，並且需要使用資料夾包裝，作為名稱。
```
|__ register/
        |__ Name1
              |__ name.jpg
              |__ name2.jpg
        |__ Name2
```
指令:
<code>python register.py -m pre --image musk.jpg --register</code>

## 測試
想要測試的文件請放在下列資料夾內：
```
|__ Data
      |__ Images/
            |__ test.jpg
      |__ Videos/
            |__ test.mp4
```

- 測試圖片:
  - <code>python register.py -m pre --image test.jpg</code>
- 測試影片:
  - <code>python register.py -m pre --video test.mp4</code>

想知道所有指令的話下 <code>python register.py -h</code> 應該就會有了。
  - -m 代表使用哪個模型，分為 mine / tune / pre
  - --image 後面接圖片名稱
  - --video 後面接影片名稱

## 訓練
訓練的部分，我懶得寫 argparse，所以你想要救自己去翻一下 train.py 和 config.py。

<code>python train.py</code>
