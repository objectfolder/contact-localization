import argparse
import yaml
from easydict import EasyDict as edict
from Engine import Engine

def parse_args():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_objects", type=int, default=300)
    parser.add_argument("--trajectory_length", type=int, default=8)
    parser.add_argument("--modality_list", nargs='+', default=['touch'])
    parser.add_argument("--model", type=str, default="OF_CNN_MFCC")
    parser.add_argument("--touch_feature_extractor", type=str, default='RESNET')
    parser.add_argument("--pretrain", type=str, default='None')
    parser.add_argument("--config_location", type=str, default="./configs/default.yml")
    parser.add_argument('--eval', action='store_true', default=False, help='if True, only perform testing')
    parser.add_argument('--convergence_eval', action='store_true', default=False, help='if True, evaluate the result when contact points converge to one location')
    # Data Locations
    parser.add_argument("--data_location", type=str, default='../DATA_new')
    parser.add_argument("--split_location", type=str, default='../DATA_new/split.json')
    # Train & Evaluation
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    # Exp
    parser.add_argument("--exp", type=str, default='test', help = 'The directory to save checkpoints and results')
    
    args = parser.parse_args()
    return args

def get_config(args):
    cfg_path = args.config_location
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)
    return edict(config)

def main():
    args = parse_args()
    cfg = get_config(args)
    engine = Engine(args, cfg)
    engine()
    
if __name__ == "__main__":
    main()