我将pytorch的Model默认下载地址改到了工程文件夹中
需要修改：
Anaconda3\envs\pytorch\Lib\site-packages\torch\hub.py
中的第60行 
原来为 DEFAULT_CACHE_DIR = '~/.cache'
改为 DEFAULT_CACHE_DIR = './.cache'
这样你在运行时就不需要重新下载模型了:)
当然，如果你运行其他人作业时，已经下好了模型，请忽略以上修改。

另外注：models文件夹是迁移学习后的模型