# _*_ coding:utf-8 _*_

'''
@Author: King
@Date: 2019.03.13
@Purpose: 处理wikizh语料库
@Link:https://dumps.wikimedia.org/zhwiki/20180801/
@Reference: https://kexue.fm/archives/4176
@opencc安装命令:pip install opencc-python-reimplemented
@opencc reference: https://github.com/yichen0831/opencc-python
'''

from gensim.corpora.wikicorpus import extract_pages,filter_wiki
import bz2file
import re
from os import path
#import opencc
from opencc import OpenCC
OpenCC = OpenCC('t2s') # convert from Simplified Chinese to Traditional Chinese
from tqdm import tqdm
import codecs

data_dir = 'resource/'
wiki = extract_pages(bz2file.open(path.join(data_dir, 'zhwiki-20180801-pages-articles-multistream.xml.bz2')))

def wiki_replace(d):
    s = d[1]
    s = re.sub(':*{\|[\s\S]*?\|}', '', s)
    s = re.sub('<gallery>[\s\S]*?</gallery>', '', s)
    s = re.sub('(.){{([^{}\n]*?\|[^{}\n]*?)}}', '\\1[[\\2]]', s)
    s = filter_wiki(s)
    s = re.sub('\* *\n|\'{2,}', '', s)
    s = re.sub('\n+', '\n', s)
    s = re.sub('\n[:;]|\n +', '\n', s)
    s = re.sub('\n==', '\n\n==', s)
    s = u'【' + d[0] + u'】\n' + s
    return OpenCC.convert(s).strip()
    
i=0

with codecs.open(path.join(data_dir, 'wiki.txt'),"w","utf-8") as f:
    w=tqdm(wiki,desc=u"已获得0篇文章")
    for d in w:
        if not re.findall('^[a-zA-Z]+:', d[0]) and d[0] and not re.findall(u'^#', d[1]):
            s=wiki_replace(d)
            f.write(s+"\n\n\n")
            i += 1
            if i%5000 == 0:
                w.set_description(u'已获取%s篇文章'%i)
                break


