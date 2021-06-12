# Data-related configuration
#event_type = "conll2003"
#event_type = "wnut17"
#event_type = "MedMentions"
#event_type = "MEDLINE"
event_type = "EMEA"
#event_type = "EMEA+MedMentions+MEDLINE"

train_file_path = "./data/%s.train" % event_type
test_file_path = "./data/%s.test" % event_type

# Model-related configuration
BERT_MODEL_DIR = "./bert-base-multilingual-cased"
ELMO_PATH = './elmo-pubmed'
ELMO_LAYERS = 2
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-05
NUM_WARMUP_STEPS = 0
EPS = 1e-6 #better not use it