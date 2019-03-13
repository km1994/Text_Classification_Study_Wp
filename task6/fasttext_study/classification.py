import logging
import fasttext
import pandas as pd
import codecs

basedir = 'resource/'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 训练
classifier = fasttext.supervised(basedir + "train_save.txt", basedir + "news.dat.seg.model", label_prefix="__label__", word_ngrams=3, bucket=2000000)

# 测试并输出 F-score
result = classifier.test(basedir + "train_save.txt")
print(result.precision * result.recall * 2 / (result.recall + result.precision))

# 读取验证集
validate_texts = []
with open(basedir + 'train_save.txt', 'r', encoding='utf-8') as infile:
    for line in infile:
        validate_texts += [line]

# 预测结果
labels = classifier.predict(validate_texts)

# 结果文件
result_file = codecs.open(basedir + "result.txt", 'w', 'utf-8')

validate_data = pd.read_table(basedir + 'train_save.txt', header=None, error_bad_lines=False)
validate_data.drop([2], axis=1, inplace=True)
validate_data.columns = ['id', 'text']

# 写入
for index, row in validate_data.iterrows():
    outline = row['id'] + '\t' + labels[index][0] + '\tNULL\tNULL\n'
    result_file.write(outline)
    result_file.flush()

result_file.close()
