This project uses PyTorch and transformers modules to implement English sequence labeling, in which BERT is fine-tuned.

Fully worked with Collab standard libs on 12/06/2021

### dataset

1. [Conll2003](https://www.clips.uantwerpen.be/conll2003/ner/)

    Conll2003.train 14987 data and conll2003.test 3466 data, a total of 4 kinds of labels：
    
    + [x] LOC
    + [x] PER
    + [x] ORG
    + [x] MISC
    
2. [wnut17](https://noisy-text.github.io/2017/emerging-rare-entities.html)

    wnut17.train 3394 pieces of data and 1009 pieces of nut17.test data, a total of 6 types of labels：
    
    + [x] Person
    + [x] Location (including GPE, facility)
    + [x] Corporation
    + [x] Consumer good (tangible goods, or well-defined services)
    + [x] Creative work (song, movie, book, and so on)
    + [x] Group (subsuming music band, sports team, and non-corporate organisations)

3. [QUAERO](https://quaerofrenchmed.limsi.fr/)

    EMEA.train EMEA.test MEDLINE.train MEDLINE.test data, a total of 10 kinds of labels：
    
    + [x] ANAT
    + [x] CHEM
    + [x] DEVI
    + [x] DISO
    + [x] GEOG
    + [x] LIVB
    + [x] OBJC
    + [x] PHEN
    + [x] PHYS
    + [x] PROC

4. [MedMentions](https://github.com/chanzuckerberg/MedMentions)

    MedMentions.train MedMentions.test, a total of 10 kinds of labels was extracted from full version：
    
    + [x] ANAT
    + [x] CHEM
    + [x] DEVI
    + [x] DISO
    + [x] GEOG
    + [x] LIVB
    + [x] OBJC
    + [x] PHEN
    + [x] PHYS
    + [x] PROC

### Model structure

BertForTokenClassification model in transformers + ELMO + LASER (u can comment elmo and laser part if dont wanna to use it)

### Model effect

- Conll2003

Model parameters：bert-base-uncased, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

run model_evaluate.py,The model evaluation results are as follows:

```
              precision    recall  f1-score   support

         LOC     0.9444    0.9706    0.9573      1837
        MISC     0.8579    0.8709    0.8644       922
         ORG     0.8993    0.9128    0.9060      1341
         PER     0.9772    0.9794    0.9783      1842

   micro avg     0.9309    0.9448    0.9378      5942
   macro avg     0.9197    0.9334    0.9265      5942
weighted avg     0.9310    0.9448    0.9378      5942
```

[The F1 value of the latest SOTA result 94.3%.](https://github.com/sebastianruder/NLP-progress/blob/master/english/named_entity_recognition.md)

- wnut17

Model parameters：bert-base-uncased, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

run model_evaluate.py,The model evaluation results are as follows:

```
               precision    recall  f1-score   support

  corporation     0.2667    0.3529    0.3038        34
creative-work     0.2500    0.1333    0.1739       105
        group     0.2059    0.1795    0.1918        39
     location     0.5250    0.5676    0.5455        74
       person     0.7711    0.6809    0.7232       470
      product     0.5263    0.1754    0.2632       114

    micro avg     0.6213    0.4964    0.5519       836
    macro avg     0.4242    0.3483    0.3669       836
 weighted avg     0.6036    0.4964    0.5339       836
```

- EMEA+MedMentions+MEDLINE

Model parameters：bert-base-multilingual-cased, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

run model_evaluate.py,The model evaluation results are as follows:

```
	precision	recall	f1-score     support
ANAT	0.4648	0.5762	0.5145       14137
CHEM	0.6231	0.6406	0.6317       31475
DEVI	0.3237	0.2386	0.2747       2355
DISO	0.5899	0.5096	0.5468       35935
GEOG	0.6910	0.6275	0.6578       2470
LIVB	0.6839	0.7169 	0.7000       24633
OBJC	0.4317	0.2836	0.3423       5606
PHEN	0.3117	0.1736	0.2230       5744
PHYS	0.4412	0.4834	0.4613       20072
PROC	0.5119	0.4433	0.4751       30515
micro avg	0.5554	0.5332	0.5440   172942
macro avg	0.5073	0.4693	0.4827	 172942
```

- EMEA+MedMentions+MEDLINE

Model parameters：bert-base-multilingual-cased+LASER, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

run model_evaluate.py,The model evaluation results are as follows:

```
	precision	recall	f1-score	support
ANAT	0.4866	0.5590	0.5203	14137
CHEM	0.5920	0.6853	0.6352	31475
DEVI	0.2581	0.3223	0.2866	2355
DISO	0.4868	0.5792	0.5290	35935
GEOG	0.5958	0.7113	0.6485	2470
LIVB	0.6580	0.7454	0.6990	24633
OBJC	0.3654	0.4417	0.3999	5606
PHEN	0.2647	0.3297	0.2937	5744
PHYS	0.4275	0.5146	0.4670	20072
PROC	0.4128	0.5896	0.4856	30515
micro avg	0.4924	0.6005	0.5411	172942
macro avg	0.4548	0.5478	0.4965	172942
```

- EMEA+MedMentions+MEDLINE

Model parameters：bert-base-multilingual-cased+LASER+ELMO+BiLSTM, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

run model_evaluate.py,The model evaluation results are as follows:

```
	precision	recall	f1-score	support
ANAT	0.5295	0.5361	0.5328	14137
CHEM	0.6327	0.6313	0.6320	31475
DEVI	0.2770	0.2811	0.2811	2355
DISO	0.5216	0.5721	0.5457	35935
GEOG	0.6655	0.6789	0.6721	2470
LIVB	0.6835	0.7487	0.7146	24633
OBJC	0.3777	0.4270	0.4008	5606
PHEN	0.3020	0.3125	0.3072	5744
PHYS	0.4489	0.5272	0.4849	20072
PROC	0.4471	0.5729	0.5022	30515
micro avg	0.5263	0.5843	0.5538	172942
macro avg	0.4885	0.5288	0.5071	172942
```

- EMEA

Model parameters：bert-base-multilingual-cased+LASER+ELMO+BiLSTM, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

run model_evaluate.py,The model evaluation results are as follows:

```
	precision	recall	f1-score	support
ANAT	0.4571	0.5581	0.5026	86
CHEM	0.6611	0.6759	0.6684	762
DEVI	0.1875	0.1184	0.1452	76
DISO	0.4338	0.5685	0.4921	248
GEOG	0.3810	0.5714	0.4571	14
LIVB	0.7559	0.8170	0.7853	235
OBJC	0.1739	0.1633	0.1684	49
PHEN	0.0000	0.0000	0.0000	29
PHYS	0.3188	0.6111	0.4190	72
PROC	0.5606	0.7105	0.6267	228
micro avg	0.5621	0.6265	0.5925	1799
macro avg	0.3930	0.4794	0.4265	1799
```

- MEDLINE

Model parameters：bert-base-multilingual-cased+LASER+ELMO+BiLSTM, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

run model_evaluate.py,The model evaluation results are as follows:

```
	precision	recall	f1-score	support
ANAT	0.3369	0.3926	0.3626	242
CHEM	0.5392	0.6654	0.5957	269
DEVI	0.0000	0.0000	0.0000	30
DISO	0.4724	0.5258	0.4977	717
GEOG	0.6571	0.4600	0.5412	50
LIVB	0.5972	0.6565	0.6255	262
OBJC	0.0000	0.0000	0.0000	32
PHEN	0.0000	0.0000	0.0000	40
PHYS	0.1457	0.2522	0.1847	115
PROC	0.5049	0.4915	0.4981	529
micro avg	0.4614	0.4965	0.4783	2286
macro avg	0.3253	0.3444	0.3305	2286
```

- MedMentions

Model parameters：bert-base-multilingual-cased+LASER+ELMO+BiLSTM, MAX_SEQ_LEN=128, BATCH_SIZE=32, EPOCH=5

run model_evaluate.py,The model evaluation results are as follows:

```
	precision	recall	f1-score	support
	precision	recall	f1-score	support
ANAT	0.5148	0.5477	0.5308 	13809
CHEM	0.6162	0.6529	0.6340 	30444
DEVI	0.2523	0.2917	0.2706	2249
DISO	0.4825	0.5873	0.5298	34970
GEOG	0.6126	0.6941	0.6508	2406
LIVB	0.6527	0.7471	0.6968	24136
OBJC	0.3537	0.4389	0.3917	5525
PHEN	0.3009	0.3064	0.3036	5675
PHYS	0.4797	0.4771	0.4784 	19885
PROC	0.4933	0.4859	0.4896 	29758
micro avg	0.5236	0.5712	0.5464	168857
macro avg	0.4759	0.5229	0.4976	168857
```

### How to run
```
!pip install seqeval transformers pytorch_pretrained_bert

!pip install laserembeddings laserembeddings[zh] laserembeddings[ja]
!python -m laserembeddings download-models
!pip install allennlp
!pip install pytorch-crf
```

```
!python load_data.py
```

```
# Use !python to avoid "called" error
#!python torch_model_train.py
%run torch_model_train.py
```

```
!python torch_model_evaluate.py
#%run torch_model_evaluate.py
```

```
!python torch_model_evaluate.py
#%run torch_model_evaluate.py
```

```
#if u need
!python torch_model_evaluate_custom_data.py dataset_name
```


### Code description

0. Put the bert-base-uncased (or any other) pre-training model of hugging face in the corresponding folder
1. Run load_data.py to generate the category label file label2id.json, and note that the O label is 0;
2. Refer to the requirements.txt document for the required Python third-party modules
3. The data you need to classify is prepared in the format of data/conll2003.train and data/conll2003.test
4. Adjust the model parameters and run torch_model_train.py for model training
5. Run torch_model_evaluate.py for model evaluation
6. Run torch_model_predict.py to predict the new text