from transformers import BertForTokenClassification, BertTokenizer

from util import BERT_MODEL_DIR

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
    print(code)
    bert_test = [id_token_dict[_] for _ in code]
    print(bert_test)
    words = []
    for word in bert_test:
        if word not in ["[CLS]", "[SEP]"]:
            words.append(word)
    return words


words = bert_encode("cosl")
print(words)