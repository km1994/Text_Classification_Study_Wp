# 文本分类实战（十）—— BERT 预训练模型
介绍网址：https://www.cnblogs.com/jiangxinyang/p/10241243.html
模型下载：https://github.com/google-research/bert

之后就可以直接执行run_classifier.py文件，执行脚本如下：
python run_classifier.py  --data_dir=../data/   --task_name=imdb   --vocab_file=../modelParams/uncased_L-12_H-768_A-12/vocab.txt   --bert_config_file=../modelParams/uncased_L-12_H-768_A-12/bert_config.json   --output_dir=../output/   --do_train=true   --do_eval=true   --init_checkpoint=../modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt   --max_seq_length=200   --train_batch_size=16   --learning_rate=5e-5  --num_train_epochs=2.0 

其中DATA_DIR是你的要训练的文本的数据所在的文件夹，BERT_BASE_DIR是你的bert预训练模型存放的地址。task_name要求和你的DataProcessor类中的名称一致。下面的几个参数，do_train代表是否进行fine tune，do_eval代表是否进行evaluation，还有未出现的参数do_predict代表是否进行预测。如果不需要进行fine tune，或者显卡配置太低的话，可以将do_trian去掉。max_seq_length代表了句子的最长长度，当显存不足时，可以适当降低max_seq_length。