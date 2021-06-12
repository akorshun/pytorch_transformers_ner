# -*- coding: utf-8 -*-
# @Time : 2021/1/31 15:01
# @Author : Jclian91
# @File : torch_model_train.py
# @Place : Yangpu, Shanghai
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.nn import BCEWithLogitsLoss
from torch.utils.data.sampler import WeightedRandomSampler

from torch.utils.data import Dataset, DataLoader
from transformers import BertForTokenClassification, BertTokenizer, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_pretrained_bert.optimization import BertAdam


from util import event_type, train_file_path, test_file_path
from util import MAX_LEN, BERT_MODEL_DIR, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_WARMUP_STEPS, EPS
from load_data import read_data

# tokenizer and label_2_id_dict
with open("{}_label2id.json".format(event_type), "r", encoding="utf-8") as f:
    tag2idx = json.loads(f.read())
    idx2tag = {v: k for k, v in tag2idx.items()}


class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            # pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        label = self.labels[index]
        label.extend([0] * MAX_LEN)
        label = label[:MAX_LEN]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len

# Creating the customized model
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        #config = BertConfig.from_pretrained("./bert-base-uncased", num_labels=len(list(tag2idx.keys())))
        #self.l1 = BertForTokenClassification.from_pretrained('./bert-base-uncased', config=config)

        #self.l0 = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=hidden_dim//2, batch_first=True) #bilstm

        #config = BertConfig.from_pretrained("./bert-base-multilingual-cased", num_labels=len(list(tag2idx.keys()))) #bert
        #self.l1 = BertForTokenClassification.from_pretrained('./bert-base-multilingual-cased', config=config)
        config = BertConfig.from_pretrained(BERT_MODEL_DIR, num_labels=len(list(tag2idx.keys()))) #bert
        self.l1 = BertForTokenClassification.from_pretrained(BERT_MODEL_DIR, config=config)

        #self.l2 = nn.LSTM(768, 256, batch_first=True,bidirectional=True)
        #self.linear = nn.Linear(256*2, 4, batch_first=True)

        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 200)

    def forward(self, ids, mask, labels):
        output_1 = self.l1(ids, mask, labels=labels) #bert

        #sequence_output, pooled_output = self.l1(
        #       ids, 
        #       attention_mask=mask)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        #lstm_output, (h,c) = self.l2(sequence_output) ## extract the 1st token's embeddings
       # hidden = torch.cat((lstm_output[:,-1, :256],lstm_output[:,0, 256:]),dim=-1)
        #linear_output = self.linear(lstm_output[:,-1].view(-1,256*2)) ### assuming that you are only using the output of the last LSTM cell to perform classification
               
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1


def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()
    return np.sum(flat_preds == flat_labels)/len(flat_labels)


def valid(model, testing_loader):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)

            output = model(ids, mask, labels=targets)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# def make_weights_for_balanced_classes(images, nclasses):                        
#     count = [0] * nclasses                                                      
#     for item in images:                                                         
#         count[item[1]] += 1                                                     
#     weight_per_class = [0.] * nclasses                                      
#     N = float(sum(count))                                                   
#     for i in range(nclasses):                                                   
#         weight_per_class[i] = N/float(count[i])                                 
#     weight = [0] * len(images)                                              
#     for idx, val in enumerate(images):                                          
#         weight[idx] = weight_per_class[val[1]]                                  
#     return weight

class DataLoaderHard(Dataset):

    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        text = self.data.text[index]
        X, _ = prepare_features(text)
        y = self.data.label[index]

        return X, y

    def __len__(self):
        return self.len

class DataLoaderSmoothing(Dataset):

    def __init__(self, dataframe, alpha):
        self.len = len(dataframe)
        self.data = dataframe
        self.alpha = alpha
        self.num_classes = len(self.data.label.unique())

    def __getitem__(self, index):
        text = self.data.text[index]
        X, _ = prepare_features(text)
        y = self.data.loc[index].iloc[2:].values.astype(float)
        label = self.data.loc[index].label
        for ind in range(len(y)):
            if ind == label:
                y[ind] = 1 - self.alpha + self.alpha / self.num_classes
                pass
            else:
                y[ind] = self.alpha / self.num_classes

        return X, y

# feature preparation for BERT
def prepare_features(seq_1, max_seq_length=512, zero_pad=True,
                     include_CLS_token = True, include_SEP_token = True):
    # Tokenzine Input
    tokens_a = tokenizer.tokenize(seq_1)

    # Truncate
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
    # Initialize Tokens
    tokens = []
    if include_CLS_token:
        tokens.append(tokenizer.cls_token)
    # Add Tokens and separators
    for token in tokens_a:
        tokens.append(token)

    if include_SEP_token:
        tokens.append(tokenizer.sep_token)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Input Mask
    input_mask = [1] * len(input_ids)
    # Zero-pad sequence lenght
    if zero_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
    return torch.tensor(input_ids).unsqueeze(0), input_mask

if __name__ == '__main__':

    # Preparing for CPU or GPU usage
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = BertTokenizer.from_pretrained('./{}'.format(BERT_MODEL_DIR))

    # Creating the Dataset and DataLoader for the neural network
    train_sentences, train_labels = read_data(train_file_path)
    train_labels = [[tag2idx.get(l) for l in lab] for lab in train_labels]
    test_sentences, test_labels = read_data(test_file_path)
    test_labels = [[tag2idx.get(l) for l in lab] for lab in test_labels]
    print("TRAIN Dataset: {}".format(len(train_sentences)))
    print("TEST Dataset: {}".format(len(test_sentences)))

    training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_LEN)
    testing_set = CustomDataset(tokenizer, test_sentences, test_labels, MAX_LEN)


    # df_train = pd.read_csv('data/wnut17.train', sep='\t', header=None)
    # df_train = pd.read_csv('data/wnut17.train', sep='\t', header=None)
    # df_train = df_train.dropna()
    # new_df = df_train[1].value_counts().to_frame()
    # new_df[2] = (new_df[1] / new_df[1].sum())

    # samples_weight = np.around(new_df[2].values, decimals = 4)
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weigth = samples_weight.double()
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))





    # target_list = []
    # for t in training_set:
    #     target_list.append(t)
    
    # target_list = torch.tensor(target_list)
    # target_list = target_list[torch.randperm(len(target_list))]

    # class_count = [i for i in get_class_distribution(y_train).values()]
    # class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    # print(class_weights)

    # class_weights_all = class_weights[target_list]

    # weighted_sampler = WeightedRandomSampler(
    #     weights=class_weights_all,
    #     num_samples=len(class_weights_all),
    #     replacement=True
    # )   





    # weights= [0.2, 0.3, 0.7, 0.2, 0.3, 0.7, 0.2, 0.3, 0.7, 0.2, 0.3, 0.7, 0,5]
    # sampler = WeightedRandomSampler(weights, num_samples=len(weights))



    # weights = make_weights_for_balanced_classes(training_set.sentences, len(training_set.labels))
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # weights = [0.1, 0.2, 0.3, 0.4]
    # weights = torch.tensor(weights)

    # sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    alpha = 0.1  # smoothing parameters for true label

    # load datasets
    train_dataset = pd.read_csv(train_file_path, sep='\t', header=None).dropna()
    val_dataset = pd.read_csv(test_file_path, sep='\t', header=None).dropna()
    train_dataset.columns = ['text', 'label']
    val_dataset.columns = ['text', 'label']
    train_dataset['label'] = train_dataset['label'].apply(lambda x: tag2idx[x])
    val_dataset['label'] = val_dataset['label'].apply(lambda x: tag2idx[x])
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VAL Dataset: {}".format(val_dataset.shape))
    training_set = DataLoaderSmoothing(train_dataset, alpha)
    validating_set = DataLoaderHard(val_dataset)

    # initialize batch sampler
    target = train_dataset.label.values
    print('target train 0/1: {}/{}'.format(
        len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = weight[target]
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    #train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'sampler': sampler}
    #test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'sampler': sampler}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # train the model
    model = BERTClass()
    model.to(dev)

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    optimizer = BertAdam(params=model.parameters(), lr=LEARNING_RATE)
    #optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, eps=EPS)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=len(training_loader)*EPOCHS)
    for epoch in range(EPOCHS):
        model.train()
        for _, data in enumerate(training_loader):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)

            loss = model(ids, mask, labels=targets)[0]

            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {_}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    # model evaluate
    valid(model, testing_loader)
    torch.save(model.state_dict(), BERT_MODEL_DIR[2:] + '_{}_ner.pth'.format(event_type))
