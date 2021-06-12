import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from load_data import bert_encode
from util import event_type, BERT_MODEL_DIR, MAX_LEN, VALID_BATCH_SIZE
from torch_model_train import BERTClass, CustomDataset, idx2tag

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BERTClass()
model.to(dev)
model.load_state_dict(torch.load(BERT_MODEL_DIR[2:] + '_{}_ner.pth'.format(event_type)))
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
            #loss = model.crf.forward(logits, targets, mask.byte())
            logits = logits.detach().cpu().numpy()
            tags = [idx2tag[_] for _ in [list(p) for p in np.argmax(logits, axis=2)][0]]
            # tags_idx = model.crf.decode(logits, mask.byte()) #CRF
            # for t in tags_idx: #CRF
            #     tags = [idx2tag[i] for i in t] #CRF

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
    test_text = "South Africa - 15 - Andre Joubert , 14 - Justin Swart"
    print(get_text_predict(test_text))