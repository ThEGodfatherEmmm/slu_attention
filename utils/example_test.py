import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, train = True):
        if train == True:
            datas = json.load(open(data_path, 'r', encoding='UTF-8'))
            examples = []
            for data in datas:
                for utt in data:
                    ex = cls(utt)
                    examples.append(ex)
            return examples
        else:
            datas = json.load(open(data_path, 'r', encoding='UTF-8'))
            examples = []
            for data in datas:
                for utt in data:
                    ex = cls(utt)
                    examples.append(ex)
            return examples

    def __init__(self, ex: dict):
        super(Example, self).__init__()
        self.ex = ex

        self.utt = ex['asr_1best']
        self.slot = {}
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        self.tags = ['O'] * len(self.utt)
        l = Example.label_vocab
