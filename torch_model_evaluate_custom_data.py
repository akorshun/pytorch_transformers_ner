# Use the seqeval module to evaluate the results of sequence annotation
from seqeval.metrics import classification_report

from util import test_file_path, BERT_MODEL_DIR
import sys

from load_data import bert_encode
from util import event_type, BERT_MODEL_DIR, MAX_LEN, VALID_BATCH_SIZE
from torch_model_train import BERTClass, CustomDataset, idx2tag

import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader

import datetime

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BERTClass()
model.to(dev)
model.load_state_dict(torch.load(sys.argv[1].format(event_type))) #name of model to use
tokenizer = BertTokenizer.from_pretrained('./{}'.format(BERT_MODEL_DIR))

def get_text_predict(text):
    test_set = CustomDataset(tokenizer, [text], [[]], MAX_LEN)
    test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_loader = DataLoader(test_set, **test_params)

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            tags = [idx2tag[_] for _ in [list(p) for p in np.argmax(logits, axis=2)][0]]

            # Output prediction results
            real_tag = []
            i = 0
            for word in text.split():
                new_word = bert_encode(word)
                if i < len(tags):
                    real_tag.append(tags[i])
                    i += len(new_word)

    return real_tag

if __name__ == '__main__':
    event_type = sys.argv[2] #name of dateset to test on it


    # Read test set data
    with open('./data/' + event_type + '.test', "r", encoding="utf-8") as f:
        content = [_.strip() for _ in f.readlines()]

    # Read the line number where the blank line is located
    index = [-1]
    index.extend([i for i, _ in enumerate(content) if not _])
    index.append(len(content))

    # Split by blank lines, read the original sentence and label sequence
    sentences, tags = [], []
    for j in range(len(index) - 1):
        sent, tag = [], []
        segment = content[index[j] + 1: index[j + 1]]
        for line in segment:
            word, bio_tag = line.split()[0], line.split()[-1]
            sent.append(word)
            tag.append(bio_tag)

        sentences.append(" ".join(sent))
        tags.append(tag)

    # Remove empty sentences and label sequences, generally put at the end
    input_test = [_ for _ in sentences if _]
    result_test = [_ for _ in tags if _]

    for sent, tag in zip(input_test[:10], result_test[:10]):
        print(sent, tag)

    # Test set
    i = 1
    true_tag_list = []
    pred_tag_list = []
    for test_text, true_tag in zip(input_test, result_test):
        print("Predict %d samples" % i)
        print("test text: ", test_text)
        pred_tag = get_text_predict(text=test_text)
        true_tag_list.append(true_tag)
        print("true tag: ", true_tag)
        print("pred tag: ", pred_tag)
        if len(true_tag) <= len(pred_tag):
            pred_tag_list.append(pred_tag[:len(true_tag)])
        else:
            pred_tag_list.append(pred_tag+["O"]*(len(true_tag)-len(pred_tag)))
        i += 1

    report_results = classification_report(true_tag_list, pred_tag_list, digits=4)
    print(report_results)
    text_file = open('model_' + sys.argv[1][:-4] + '_on_dataset_' + event_type + '_eval_results_' + datetime.datetime.now().isoformat() + '.txt', "w")
    text_file.write(report_results)
    text_file.close()