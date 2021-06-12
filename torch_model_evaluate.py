# Use the seqeval module to evaluate the results of sequence annotation
from seqeval.metrics import classification_report

from util import test_file_path, BERT_MODEL_DIR, event_type
from torch_model_predict import get_text_predict, dev


if __name__ == '__main__':
    # Read test set data
    with open(test_file_path, "r", encoding="utf-8") as f:
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
    text_file = open(BERT_MODEL_DIR[2:] + '_' + event_type + '_eval_results.txt', "w")
    text_file.write(report_results)
    text_file.close()