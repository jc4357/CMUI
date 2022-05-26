import argparse
import os
import numpy as np
import random
import torch
from src import Dictionary, Ontology, Data, MIE, evaluate
import torch
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False


setup_seed(4357)


parser = argparse.ArgumentParser(description='MIE')
parser.add_argument('--add-global', type=bool, default=False, help='Add global module or not.')
parser.add_argument('--hidden-size', type=int, default=400, help='Hidden size.')
parser.add_argument('--mlp-layer-num', type=int, default=4, help='Number of layers of mlp.')
parser.add_argument('--aggregate-layer-num', type=int, default=3, help='Number of layers of aggregate.')
parser.add_argument('--keep-p', type=float, default=0.6, help='1 - dropout rate.')

parser.add_argument('--start-lr', type=float, default=1e-3, help='Start learning rate.')
parser.add_argument('--end-lr', type=float, default=1e-4, help='End learning rate.')
parser.add_argument('-e', '--epoch-num', type=int, default=120, help='Epoch num.')
parser.add_argument('-b', '--batch-size', type=int, default=35, help='Batch size.')
parser.add_argument('-tp', '--tbatch-size', type=int, default=175, help='Test batch size.')
parser.add_argument('-g', '--gpu-id', type=str, default=0, help='Gpu id.')
parser.add_argument('-l', '--location', type=str, default='./label_selective_atten_06_select_03/', help='Location to save.')
parser.add_argument('-f', '--filename', type=str, default='model.pth', help='filename to save.')
parser.add_argument('-mPos', '--mPos', type=float, default=2.5, help='mPos.')
parser.add_argument('-mNeg', '--mNeg', type=float, default=1, help='mNeg.')
parser.add_argument('-gamma', '--gamma', type=float, default=0.15, help='gamma.')
parser.add_argument('-alpha', '--alpha', type=float, default=0.6, help='alpha.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

dictionary = Dictionary()
dictionary.load('./data/dictionary.txt')

ontology = Ontology(dictionary)
ontology.add_raw('./data/ontology.json', '状态')
ontology.add_examples('./data/example_dict.json')
data = Data(100, dictionary, ontology)
data.add_raw('train', './data/train.json', 'window')
data.add_raw('test', './data/test.json', 'window')
data.add_raw('dev', './data/dev.json', 'window')

params = {
    "add_global": args.add_global,
    "num_units": args.hidden_size,
    "num_layers": args.mlp_layer_num,
    "keep_p": args.keep_p,
    "mPos":args.mPos,
    "mNeg":args.mNeg,
    "gamma":args.gamma,
    "aggregate_layers":args.aggregate_layer_num,

}
torch.cuda.set_device(args.gpu_id)
model = MIE(data, ontology, params=params)

# model.train(
#     epoch_num=args.epoch_num,
#     batch_size=args.batch_size,
#     tbatch_size=args.tbatch_size,
#     start_lr=args.start_lr,
#     end_lr=args.end_lr,
#     cuda=args.cuda,
#     location=args.location,
#     filename=args.filename,
#     alpha=args.alpha
#     )


model._load(args.batch_size,args.location,args.filename)
model.model = model.model.cuda()
infos = evaluate(model, 'test', 100)
print(infos)
