#! -*- coding: utf-8 -*-

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField, Lambda
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

maxlen = 256
epochs = 1
batch_size = 32
bert_layers = 12
learing_rate = 1e-4  
crf_lr_multiplier = 1000  

# bert setting
config_path = 'models/BERT_MODELS/uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'models/BERT_MODELS/uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'models/BERT_MODELS/uncased_L-12_H-768_A-12/vocab.txt'

# electra setting
# config_path = 'models/ELECTRA_BASE_DIR/electra_small/hyperparameters.json'
# checkpoint_path = 'models/ELECTRA_BASE_DIR/electra_small/model.ckpt-0'
# dict_path = 'models/ELECTRA_BASE_DIR/electra_small/vocab.txt'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, part_of_speech, IOB_tagging, this_flag = c.split(' ')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# labelling data
train_data = load_data('data/CONLL2003/train.txt')
valid_data = load_data('data/CONLL2003/valid.txt')
test_data = load_data('data/CONLL2003/test.txt')

# build tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# mapping
labels = ['PER', 'LOC', 'ORG', 'MISC']
id2label = dict(enumerate(labels))
label2id = {j: i for i, j in id2label.items()}
num_labels = len(labels) * 2 + 1


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# bert setting
model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
bert_output = model.get_layer(output_layer).output

# overwrite if using bert as embedding
output = Dense(num_labels)(bert_output)

# electra setting
# model = build_transformer_model(
#     config_path,
#     checkpoint_path,
#     model='electra'
# )
# output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
# electra_output = model.get_layer(output_layer).output
# output = Dense(num_labels)(electra_output)

# using bert or electra as embedding
# Add 1 bidirectional LSTM
# output = Bidirectional(LSTM(128, return_sequences=True))(bert_output)
# output = TimeDistributed(Dense(128))(output)


CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)

model = Model(model.input, output)
model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)


class NamedEntityRecognizer(ViterbiDecoder):
    def recognize(self, text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text, tokens)
        token_ids = tokenizer.tokens_to_ids(tokens)
        segment_ids = [0] * len(token_ids)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        nodes = model.predict([token_ids, segment_ids])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], id2label[(label - 1) // 2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1] + 1], l)
                for w, l in entities]


NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])


def evaluate(data, act):
    X, Y, Z = 1e-10, 1e-10, 1e-10

    if act == 'test':
        output_dict = {}

    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)

        if act == 'test':
            output_dict.update(R)

    if act == 'test':
        with open("rst.txt", "w") as outfile:
            for k,v in output_dict.items():
                outfile.write(k + ':' + v + '\n')

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        f1, precision, recall = evaluate(valid_data, 'valid')

        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data, 'test')
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )