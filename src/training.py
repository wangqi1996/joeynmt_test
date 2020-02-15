# coding=utf-8

"""
Training modulem
"""
import argparse

from src.data.data import load_data
from src.util.model_util import set_seed
from src.util.util import load_config


def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # set the random seed
    seed = cfg['training'].get('random_seed', 42)
    set_seed(seed)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = load_data(
        data_cfg=cfg['data']
    )

    # build an encoder-decoder model
    model = build_momdel(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('joeynmt')
    parser.add_argument("config", default="configs/default.yaml",
                        type=str, help='Training configuration file (yaml).')
    args = parser.parse_args()
    train(cfg_file=args.config)
