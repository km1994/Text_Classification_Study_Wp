# 文本分类实战（十）—— BERT 预训练模型
介绍网址：https://www.cnblogs.com/jiangxinyang/p/10241243.html
模型下载：https://github.com/google-research/bert

之后就可以直接执行run_classifier.py文件，执行脚本如下：
python run_classifier.py  --data_dir=../data/   --task_name=imdb   --vocab_file=../modelParams/uncased_L-12_H-768_A-12/vocab.txt   --bert_config_file=../modelParams/uncased_L-12_H-768_A-12/bert_config.json   --output_dir=../output/   --do_train=true   --do_eval=true   --init_checkpoint=../modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt   --max_seq_length=200   --train_batch_size=16   --learning_rate=5e-5  --num_train_epochs=2.0 