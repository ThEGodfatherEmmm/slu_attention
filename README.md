### 依赖

pytorch >= 1.7.1(GPU vision)
transformers == 2.2.2

### 运行
    
在script下运行
  baseline 模型：
+ python slu_baseline.py 

attention 模型：
+ python slu_attention.py

test代码：
+ python slu_test.py
test 有可选项attention模型或baseline模型，在--test_model参数中设置

### 代码说明

+ `utils/args.py`:定义了所有涉及到的可选参数，如需改动某一参数可以在运行的时候将命令修改成
        
        python scripts/slu_baseline.py --<arg> <value>
    其中，`<arg>`为要修改的参数名，`<value>`为修改后的值
+ `utils/initialization.py`:初始化系统设置，包括设置随机种子和显卡/CPU
+ `utils/vocab.py`:构建编码输入输出的词表
+ `utils/word2vec.py`:读取词向量
+ `utils/example.py`:读取数据
+ `utils/batch.py`:将数据以批为单位转化为输入
+ `model/slu_baseline_tagging.py`:baseline模型
+ `model/GatedAttentionSLU.py`:attention模型
+ `scripts/slu_baseline.py`:主程序脚本
+ `scripts/slu_attention.py`:使用attention机制的主程序脚本
+ `scripts/slu_test.py`:在data文件夹下生成test.json的脚本
+ `trained_model/baseline_model.pth`:使用baseline生成的模型
+ `trained_model/attention_model.pth`:使用attention生成的模型
+ `data/test.json`:生成的测试文件
+ `log.txt`:baseline训练时的日志文件
+ `log_attention.txt`:attention模型训练时的日志文件

