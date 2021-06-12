# -*- coding: utf-8 -*-
# @Time : 2020/12/24 13:27
# @Author : Jclian91
# @File : load_data.py
# @Place : Yangpu, Shanghai
import json
from transformers import BertTokenizer

from util import train_file_path, event_type, BERT_MODEL_DIR

dict_path = './{}/vocab.txt'.format(BERT_MODEL_DIR)
token_dict = {}
with open(dict_path, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR)
id_token_dict = {v: k for k, v in token_dict.items()}


# Get the word result after BERT parsing
def bert_encode(word):
    code = tokenizer.encode(word)
    bert_test = [id_token_dict[_] for _ in code]
    words = []
    for word in bert_test:
        if word not in ["[CLS]", "[SEP]"]:
            words.append(word)
    return words


# Read data set
def read_data(file_path):
    # Read data set
    with open(file_path, "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    # Read the line number where the blank line is located
    index = [-1]
    index.extend([i for i, _ in enumerate(content) if not _])
    index.append(len(content))

    # Split by blank lines, read the original sentence and label sequence
    sentences, tags = [], []
    for j in range(len(index)-1):
        sent, tag = [], []
        segment = content[index[j]+1: index[j+1]]
        for line in segment:
            word, bio_tag = line.split()[0], line.split()[-1]
            sent.append(word)
            for _ in bert_encode(word):
                tag.append(bio_tag)

        sentences.append(" ".join(sent))
        tags.append(tag)

    # Remove empty sentences and label sequences, generally put at the end
    sentences = [_ for _ in sentences if _]
    tags = [_ for _ in tags if _]

    return sentences, tags


# Read the training set data
# Convert label to id
def label2id():

    _, train_tags = read_data(train_file_path)

    # The label is converted to id and saved as a file
    unique_tags = []
    for seq in train_tags:
        for _ in seq:
            if _ not in unique_tags and _ != "O":
                unique_tags.append(_)

    label_id_dict = {"O": 0}
    label_id_dict.update(dict(zip(unique_tags, range(1, len(unique_tags)+1))))

    with open("%s_label2id.json" % event_type, "w", encoding="utf-8") as g:
        g.write(json.dumps(label_id_dict, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sentences, tags = read_data(train_file_path)
    for sent, tag in zip(sentences[:10], tags[:10]):
        print(sent, tag)
    label2id()
