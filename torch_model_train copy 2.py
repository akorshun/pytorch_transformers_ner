# -*- coding: utf-8 -*-
# @Time : 2021/1/31 15:01
# @Author : Jclian91
# @File : torch_model_train.py
# @Place : Yangpu, Shanghai
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
import numpy as np
import pandas as pd
import catalyst

from torch.utils.data import Dataset, DataLoader, Sampler
from torch import Tensor
from torch._six import int_classes as _int_classes
from transformers import BertForTokenClassification, BertTokenizer, BertModel, BertConfig, BertPreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_pretrained_bert.optimization import BertAdam

from transformers.modeling_outputs import TokenClassifierOutput
from transformers.file_utils import ModelOutput

from functools import reduce
from operator import add


from tqdm import tqdm
from typing import List, Optional, Tuple, Sequence

from util import event_type, train_file_path, test_file_path
from util import MAX_LEN, BERT_MODEL_DIR, ELMO_PATH, ELMO_LAYERS, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_WARMUP_STEPS, EPS
from load_data import read_data

from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler, SequentialSampler, BatchSampler
from catalyst.data.sampler import DynamicBalanceClassSampler, BalanceClassSampler, BalanceBatchSampler
#from detectron2.data.samplers import GroupedBatchSampler, TrainingSampler, InferenceSampler, RepeatFactorTrainingSampler
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
from torch.utils.data.sampler import RandomSampler
import imblearn
from imblearn.over_sampling import SMOTE

from laserembeddings import Laser
from allennlp.modules.elmo import Elmo, batch_to_ids


# tokenizer and label_2_id_dict
with open("{}_label2id.json".format(event_type), "r", encoding="utf-8") as f:
    tag2idx = json.loads(f.read())
    idx2tag = {v: k for k, v in tag2idx.items()}

# Preparing for CPU or GPU usage
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #make sure "dev" is global var coz sometimes it cant see it

laser = Laser() # to see it everywhere

def get_class_weights():
    df = pd.read_csv(train_file_path, sep='\t', header=None)
    df.columns = ['Text', 'Label']
    new_df = df['Label'].value_counts().to_frame()
    new_df['label'] = new_df.index
    new_df.columns = ['count','label', ]
    new_df['percentage'] = 1 - new_df['count'] / new_df['count'].sum()
    class_weights = new_df['percentage'].to_list()
    class_weights = torch.FloatTensor(class_weights).to(dev)
    return class_weights

def get_class_weights_new(): #num sum of class_weights = 1 but somehow it can be worse than usual 
    df = pd.read_csv(train_file_path, sep='\t', header=None)
    df.columns = ['Text', 'Label']
    new_df = df['Label'].value_counts().to_frame()
    new_df['label'] = new_df.index
    new_df.columns = ['count','label', ]
    new_df['percentage'] = 1 - new_df['count'] / new_df['count'].sum()
    new_df['new_weights'] = new_df['percentage'] / new_df['percentage'].sum()
    class_weights = new_df['new_weights'].to_list()
    class_weights = torch.FloatTensor(class_weights).to(dev)
    return class_weights

def get_class_weights_normed(): #trash
    df = pd.read_csv(train_file_path, sep='\t', header=None)
    df.columns = ['Text', 'Label']
    new_df = df['Label'].value_counts().to_frame()
    new_df['label'] = new_df.index
    new_df.columns = ['count','label', ]

    class_weights = new_df['count'].to_list()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = 1.0 / class_weights
    class_weights = class_weights / class_weights.sum()

    class_weights = torch.FloatTensor(class_weights).to(dev)
    return class_weights

def get_class_weights_other(): #1/count
    df = pd.read_csv(train_file_path, sep='\t', header=None)
    df.columns = ['Text', 'Label']
    new_df = df['Label'].value_counts().to_frame()
    new_df['label'] = new_df.index
    new_df.columns = ['count','label', ]
    new_df['percentage'] = 1.0 / new_df['count']
    new_df['new_weights'] = new_df['percentage'] / new_df['percentage'].sum() * len(new_df['label'])
    weigts = new_df['new_weights'].to_list()
    class_weights = new_df['percentage'].to_list()
    class_weights = torch.FloatTensor(class_weights).to(dev)
    return class_weights

def get_class_weights_corrected(): #do not use it makes things worse
    df = pd.read_csv(train_file_path, sep='\t', header=None)
    df.columns = ['Text', 'Label']
    df = df.loc[df['Label'] != 'O']
    new_df = df['Label'].value_counts().to_frame()
    new_df['label'] = new_df.index
    new_df.columns = ['count','label', ]
    new_df['percentage'] = 1 - new_df['count'] / new_df['count'].sum()
    class_weights = new_df['percentage'].to_list()
    class_weights.insert(0, 0.1) #insert very small for O tag, try to increase f1 scores
    class_weights = torch.FloatTensor(class_weights).to(dev)
    return class_weights

def get_class_fixed_weights(): #manually create weights to test
    class_weights = torch.ones((21,))
    #class_weights = [12016, 674, 616, 511, 387, 256, 177, 130, 125, 107, 70, 60, 48, 34, 31, 21, 19, 11, 4, 3, 1]
    #class_weights = [1, 1, 1, 1, 1]
    class_weights = torch.FloatTensor(class_weights).to(dev)
    return class_weights

class_weights = get_class_weights_new()

def get_laser_vectors():
    # Read train set data
    with open(train_file_path, "r", encoding="utf-8") as f:
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

    # Test set
    i = 1
    sentences_list = []
    for test_text, true_tag in zip(input_test, result_test):
        #print("Predict %d samples" % i)
        sentences_list.append(test_text)

    laser = Laser()
    
    laser_embeddings = laser.embed_sentences(sentences_list, lang='en')
    laser_embeddings = torch.tensor(laser_embeddings) # make it tensors
    laser_embeddings = laser_embeddings.to(dev) #move to gpu

    return laser_embeddings

laser_embeddings_global = get_laser_vectors()


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
        label = label[:MAX_LEN] #choose first MAX_LEN(128) symbols coz it works better than last or some case of first and last half
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            #'elmo_ids': elmo_ids.clone().detach(),
            'mask': torch.tensor(mask, dtype=torch.long),
            #'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len


# Creating the customized BertForTokenClassification
class BertForTokenClassificationCustom(BertPreTrainedModel):
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.elmo_layers = ELMO_LAYERS
        self.elmo = Elmo(ELMO_PATH + '/options.json', ELMO_PATH + '/elmo.hdf5', self.elmo_layers, dropout=0, requires_grad=False)
        elmo_embedding_dim = 512 # it's always fixed
        self.elmo2hidden = nn.Linear(config.hidden_size+elmo_embedding_dim*self.elmo_layers, config.hidden_size)


        rnn_dim = 512
        self.bilstm = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
        out_dim = rnn_dim*2

        #self.multihead_attn = nn.MultiheadAttention(config.hidden_size, num_heads=1)

        self.bilstm2tag = nn.Linear(out_dim, config.num_labels)

        self.crf = CRF(config.num_labels, batch_first=True)

        self.softmax = nn.Softmax(dim=2)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        bert_output = outputs[0]

        # Get batch sentences
        tokenizer = BertTokenizer.from_pretrained('./{}'.format(BERT_MODEL_DIR))
        sentences_for_embedding = tokenizer.batch_decode(input_ids, skip_special_tokens=True) #decode batch tokens back to original sentences
        sentences_for_embedding_splitted = [sentence.split(" ") for sentence in sentences_for_embedding] #split all sentences into words

        # lang is only used for tokenization
        laser_embeddings_en = torch.tensor(laser.embed_sentences(sentences_for_embedding, lang='en')).to(dev, dtype=torch.long) # make it tensors and move to gpu
        #laser_embeddings_en_fr = torch.tensor(laser.embed_sentences(sentences_for_embedding, lang=['en', 'fr'])).to(dev, dtype=torch.long) # dont matches batches so ignore for now
        laser_embeddings_en = torch.unsqueeze(laser_embeddings_en, 1) # add fake dim in center (from [32, 1024] to [32, 1, 1024])

        # We need to pad laser embeddings to have the same sequence length as BERT embeddings.
        zero_vector = torch.zeros(laser_embeddings_en.shape[0], bert_output.shape[1]-laser_embeddings_en.shape[1], laser_embeddings_en.shape[2]).to(dev, dtype=torch.long)
        if laser_embeddings_en.shape[1] < bert_output.shape[1]:
            laser_embeddings_en = torch.cat((laser_embeddings_en, zero_vector), dim=1)

        laser_bert_embeddings = torch.cat((bert_output, laser_embeddings_en), dim=-1) #concat along last dim

        laser_bert_embeddings = self.elmo2hidden(laser_bert_embeddings) #dense to standart bert hidden size (from 1792 to 768) (if last dim was concatted)

        elmo_ids = batch_to_ids(sentences_for_embedding_splitted).to(dev, dtype=torch.long) #create token ids for elmo and move to gpu
        elmo_hiddens = self.elmo(elmo_ids)
        elmo_embeddings = elmo_hiddens['elmo_representations'][0] #example size [32, 74, 1024]
        #   32    - the batch size
        #   74    - the sequence length of the batch
        #   1024  - the length of each ELMo vector

        # We need to pad elmo embeddings to have the same sequence length as BERT embeddings.
        zero_vector = torch.zeros(elmo_embeddings.shape[0], bert_output.shape[1]-elmo_embeddings.shape[1], elmo_embeddings.shape[2]).to(dev, dtype=torch.long)
        if elmo_embeddings.shape[1] < bert_output.shape[1]:
            elmo_embeddings = torch.cat((elmo_embeddings, zero_vector), dim=1) # add zeros tensor to fill elmo embeddung to match bert embedding size (example: from 32, 74, 1024 to 32, 128, 1024)


        elmo_laser_bert_embeddings = torch.cat((laser_bert_embeddings, elmo_embeddings), dim=-1)

        
        end_output = self.elmo2hidden(elmo_laser_bert_embeddings) #dense to standart bert hidden size (from 1792 to 768) (if last dim was concatted)

        lstm_output, _ = self.bilstm(end_output) #bilstm for elmo_bert_embeddings

        logits = self.bilstm2tag(lstm_output) #classify bilstm layer
        logits = self.dropout2(logits) # dropout logits

        softmax_logits = self.softmax(logits)

        loss = None
        if labels is not None:
            #loss_fct = CrossEntropyLoss()
            loss_fct = CrossEntropyLoss(weight=class_weights, reduction='mean') #Weighted loss to get better results
            
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                #loss = -self.crf(logits, labels, attention_mask.byte()) #worse worser than CrossEntropyLoss but works
                #logits = self.crf.decode(logits, attention_mask.byte())
                #loss_mask = labels.gt(-1)
                #loss = self.crf(softmax_logits, labels, loss_mask) * (-1) #doesnt work only O
                #loss = self.crf.negative_log_loss(logits, attention_mask, labels) #doesnt work only O
                #loss = -self.crf(torch.log_softmax(logits, dim=2), labels, head_mask, reduction='mean') #doesnt work only O
                #loss = -self.crf(logits, labels, attention_mask.byte(), reduction='mean') #doesnt work almost all O
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((logits,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        #works but worser than CrossEntropyLoss
        # if labels is not None:
        #     # Only keep active parts of the loss
        #     if attention_mask is not None:
        #       attenion_mask_byte = attention_mask.byte()
        #       loss = -1*self.crf(logits, labels, mask=attention_mask.byte())
        #       logits = self.crf.decode(logits, mask=attention_mask.byte())
        #       logits = torch.FloatTensor(logits).to(dev)
        #     else:
        #       loss, logits = -1*self.crf(logits, labels), self.crf.decode(logits)
        #     outputs2 = loss
        # else:
        #   logits = self.crf.decode(logits, mask=attention_mask.byte())
        #   outputs2 = logits

        # return TokenClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )



        # #works but worser than CrossEntropyLoss
        # # If labels were given, we'll want to use them to train the CRF layer as well
        # if labels is not None:
        #     # Note that this mutates labels. This is only okay because we don't use reuse labels in
        #     # the learner. If this ever changes you can fix this by using labels.clone()
        #     for i in range(labels.shape[0]):
        #         for j in range(labels.shape[1]):
        #             if labels[i][j] == self.crf.num_tags:
        #                 # Change 'X' label to 'O' so crf doesn't try to access wrong index
        #                 # Using a mask does not fix this.
        #                 labels[i][j] = 0

        #     # After 'X' labels have been removed, pass the emission scores through the CRF layer
        #     loss = -self.crf(softmax_logits, tags=labels)

        # # if not return_dict:
        # #     output = (logits,) + outputs[2:]
        # #     return ((logits,) + output) if loss is not None else output

        # return TokenClassifierOutput(
        #     loss=loss,
        #     logits=softmax_logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )


# Creating the customized model
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        config = BertConfig.from_pretrained(BERT_MODEL_DIR, num_labels=len(list(tag2idx.keys()))) #bert
        self.bertForToken = BertForTokenClassificationCustom.from_pretrained(BERT_MODEL_DIR, config=config)

    def forward(self, ids, mask, labels):
        sequence_output = self.bertForToken(ids, mask, labels=labels) #bert

        return sequence_output


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

            output = model(ids, mask, targets)
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

class CustomWeightedRandomSampler(Sampler[int]):
    weights: Tensor
    num_samples: int
    replacement: bool

    def __init__(self, weights: Sequence[float], num_samples: int,
                 replacement: bool = True, generator=None) -> None:
        if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        return iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples

if __name__ == '__main__':

    # Preparing for CPU or GPU usage
    #dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    train_labels2 = training_set.labels
    _, counts = np.unique(train_labels2, return_counts=True)
    train_weights = 1./counts
    train_weights = train_weights / train_weights.sum() * len(list(tag2idx.keys()))

    sampler = WeightedRandomSampler(weights=train_weights, replacement=True, num_samples=len(training_set) * 2)
    batch_sampler = BatchSampler(sampler, batch_size=TRAIN_BATCH_SIZE, drop_last=False)


    #train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'sampler': sampler}
    train_params = {'shuffle': False, 'num_workers': 0, 'batch_sampler': batch_sampler}
    #train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # train the model
    model = BERTClass()
    model.to(dev)

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    optimizer = BertAdam(params=model.parameters(), lr=LEARNING_RATE, eps=EPS) #seems to be the best for bert based models
    #optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE, eps=EPS)

    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=len(training_loader)*EPOCHS) #sometimes it will better but without BertAdam
    for epoch in range(EPOCHS):
        model.train()
        for _, data in enumerate(training_loader):
            ids = data['ids'].to(dev, dtype=torch.long)
            mask = data['mask'].to(dev, dtype=torch.long)
            targets = data['tags'].to(dev, dtype=torch.long)

            output = model(ids, mask, targets)
            loss = output[0] # bert model loss

            if _ % 50 == 0:
                print(f'Epoch: {epoch}, Batch: {_}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

        #valid(model, testing_loader) #validate model after each epoch, bad idea if dataset is big

    # model evaluate
    valid(model, testing_loader)
    torch.save(model.state_dict(), BERT_MODEL_DIR[2:] + '_{}_ner.pth'.format(event_type))
