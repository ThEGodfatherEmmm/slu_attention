#coding=utf8
import sys, os, time, gc
from torch.optim import Adam
import json

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.example_test import Example
from utils.batch import from_example_list
from utils.vocab import PAD
from model.slu_baseline_tagging import SLUTagging
from model.slu_attention import slu_attention

args = init_args(sys.argv[1:])

baseline_model_path = os.path.join(args.modelroot, 'baseline_model.pth')
attention_model_path = os.path.join(args.modelroot, 'attention_model.pth')
file_name = os.path.join(args.dataroot, 'test.json')
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

train_path = os.path.join(args.dataroot, 'train.json')
test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
raw = json.load(open(test_path, 'r', encoding='utf-8-sig'))
Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
test_dataset = Example.load_dataset(test_path, train=False)

args.vocab_size = Example.word_vocab.vocab_size
args.pad_idx = Example.word_vocab[PAD]
args.num_tags = Example.label_vocab.num_tags
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

if args.test_model=='attention':
    model = slu_attention(args).to(device)
    model.load_state_dict(torch.load(attention_model_path)['model'])
else:
    model = SLUTagging(args).to(device)
    model.load_state_dict(torch.load(baseline_model_path)['model'])
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)

model.eval()
dataset = test_dataset
predictions = []
with torch.no_grad():
    for i in range(0, len(dataset), args.test_batch_size):
        cur_dataset = dataset[i: i + args.test_batch_size]
        current_batch = from_example_list(args, cur_dataset, device, train=False)
        pred, label = model.decode_(Example.label_vocab, current_batch)
        predictions.extend(pred)

torch.cuda.empty_cache()
gc.collect()
idx = 0
for data in raw:
    for utt in data:
        # print(predictions)
        utt['pred'] = predictions[idx]
        # print(utt)
        idx += 1
with open(file_name, 'w', encoding='utf-8-sig') as f:
    json.dump(raw, f, ensure_ascii=False, indent=4)

print('Test finished')

