# CAIL_Bert_MRC

###1. ensemble数据划分

1. 准备数据在如下路径：data/data.json

2. 配置交叉验证的参数，在config/args.py
cross_validation_k = 4

3. 运行如下命令,在data/目录下，生成k折训练使用的数据:
```bash
python prepro.py
```

###2. 配置参数，使用数据
1. config/args.py中的do_ensemble控制读取上个步骤中生成的examples文件

（1）do_ensemble=False

表示普通的模式，使用train.json,dev.json训练

（2）do_ensemble = True

配置要训练使用的数据，在config/args.py；

index_ensemble_model = 'civil'

index_ensemble_model = 'criminal'

index_ensemble_model = 0

index_ensemble_model = 1

index_ensemble_model = 2,...,k-1

2. 运行如下命令:
```bash
python train_start.py
```

##3. 预测使用配置

运行如下命令:
```bash
python main.py
```

##4. 本地评估

运行如下命令:
```bash
python gold_evaluate.py --data-file "../data/data.json" --pred-file "../result/result.json"
# 或者
python gold_evaluate.py
```



