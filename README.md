# NLP-ZeroStart
Some basic programs of NLP;
本仓库的代码仅为代码实战练习，实现了简单的文本分类、文本匹配、序列标注、文本生成等能力；
部分项目已经包含了训练的数据，其他项目的数据可以联系作者获取； 微信:lby23571113;

Classification项目：
  为新闻的18分类，其中利用了Bert,RNN,RGU,LSTM,CNN,fast_text,text_cnn,gated_cnn,rcnn等多种方式实现，可以在config文件中配置相关参数；

SentenceMatch项目:
  该项目为表示型文本匹配实现的移动当客服场景的常见问题匹配；

NER_test：
  采用BIO的方式，基于Bert实现的命名实体识别；

GenerateAbstrat项目：
  基于自己实现的transformer架构，实现了简单的摘要生成，数据为摘要和标题对，摘要作为文本，标题作为待生成的摘要；
GenerateAbstrat1项目：
  将GenerateAbstrat项目中的transformer架构结构梳理到多个文件中；

KGQABaseOnSentenceMatch项目：
  通过模板匹配的方式实现的基于知识图谱的问答；




