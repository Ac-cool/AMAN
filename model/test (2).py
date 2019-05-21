import json
import numpy as np
from keras.utils.np_utils import to_categorical
LABELS = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
def read_snli(path_t):
    path_tt = "./datasets/"+path_t
    texts1 = []
    texts2 = []
    labels = []
    with open(path_tt,"r") as file_:
        for line in file_:
            eg = json.loads(line)
            label = eg['gold_label']
            if label == '-':
                continue
            texts1.append(eg['sentence1'])
            texts2.append(eg['sentence2'])
            #print(eg['sentence1'])
            #print(eg['sentence2'])
            #print(LABELS[label])
            labels.append(LABELS[label])
    return texts1, texts2, to_categorical(np.asarray(labels, dtype='int32'))

if __name__ =="__main__":
    ll = read_snli("train.jsonl")
    texts1, texts2,labels = ll
    print(len(texts1),len(texts2),len(labels))
    print(labels[0])
    print(ll[0][0])
    print(ll[1][0])
    print(ll[2][0])


