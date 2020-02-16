# coding=utf-8

"""
Training modulem
"""
import argparse
import os
import queue
import time

import math
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchtext.data import Dataset

from src.data.batch import Batch
from src.data.data import load_data, make_data_iter
from src.model.model import build_model, Model
from src.module.builder import build_gradient_clipper, build_optimizer, build_scheduler
from src.module.loss import XentLoss
from src.util.model_util import set_seed
from src.util.util import load_config, make_model_dir, make_logger, ConfigurationError, load_checkpoint, symlink_update


class TrainManager:
    """
    Manager training loop, validations, learning rate scheduling and early stopping
    """

    def __init__(self, model: Model, config: dict) -> None:
        """
        creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["training"]

        # files for logging and storing
        self.model_dir = make_model_dir(train_config["model_dir"],
                                        overwrite=train_config.get("overwrite", False))
        self.logger = make_logger("{}/train.log".format(self.model_dir))
        self.logging_freq = train_config.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        # model
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self._log_parameters_list()

        # objective
        self.label_smoothing = train_config.get("label_smoothing", 0.0)
        self.loss = XentLoss(pad_index=self.pad_index, smoothing=self.label_smoothing)

        self.normalization = train_config.get("normalization", "batch")
        if self.normalization not in ['batch', 'tokens', 'none']:
            raise ConfigurationError("Invalid normalization option."
                                     "Valid options: "
                                     "'batch', 'tokens', 'none'.")

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=train_config)
        self.optimizer = build_optimizer(config=train_config, parameters=model.parameters())

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.log_valid_sents = train_config.get("priingvalid_sents", [0, 1, 2])
        self.ckpt_queue = queue.Queue(
            maxsize=train_config.get("keep_last_ckpts", 5)
        )
        self.eval_metric = train_config.get('eval_metric', 'bleu')
        if self.eval_metric not in ['bleu',
                                    'chrf',
                                    'token_accuracy',
                                    'sequence_accuracy']:
            raise ConfigurationError("Invalid setting for 'eval_metric', "
                                     "valid options: 'bleu', 'chrf', "
                                     "'token_accuracy', 'sequence_accuracy'.")

        self.early_stopping_metric = train_config.get("early_stopping_metric", "eval_metric")

        # if we schedule after BLEU/chrf,  we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for the metric

        if self.early_stopping_metric in ["ppl", "loss"]:
            self.minimize_metric = True
        elif self.early_stopping_metric == "eval_metric":
            if self.eval_metric in ["bleu", "chrf"]:
                self.minimize_metric = False
            # eval metric that has to get minimized (not yet implemented)
            else:
                self.minimize_metric = True
        else:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'ppl', 'eval_metric'.")

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=config["model"]["encoder"]["hidden_size"]
        )

        # data & batch handling
        self.level = config["data"]["config"]
        if self.level not in ["word", "bpe", "char"]:
            raise ConfigurationError("Invalid segmentation level. "
                                     "Valid options: 'word', 'bpe', 'char'.")

        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_type = train_config.get("batch_type", "sentence")
        self.eval_batch_size = train_config.get("eval_batch_size",
                                                self.batch_size)
        self.eval_batch_type = train_config.get("eval_batch_type",
                                                self.batch_type)

        self.batch_multiplier = train_config.get("batch_multiplier", 1)
        self.current_batch_multiplier = self.batch_multiplier

        # generation
        self.max_output_length = train_config.get("max_output_length", None)

        # CPU / GPU
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()

        # initialize accumulated batch loss (needed for batch_multiplier)
        self.norm_batch_loss_accumulated = 0

        # initialize training statistics
        self.steps = 0

        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0

        # initial values for best scores
        self.best_ckpt_scores = np.inf if self.minimize_metric else -np.inf

        # comparision function for scores
        self.is_best = lambda score: score < self.best_ckpt_scores \
            if self.minimize_metric else score > self.best_ckpt_scores

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            reset_best_ckpt = train_config.get("reset_best_ckpt", False)
            reset_scheduler = train_config.get("reset_scheduler", False)
            reset_optimizer = train_config.get("reset_optimizer", False)
            self.init_from_checkpoint(model_load_path,
                                      reset_best_ckpt=reset_best_ckpt,
                                      reset_scheduler=reset_scheduler,
                                      reset_optimizer=reset_optimizer)

    def _save_checkpoint(self) -> None:
        """
        save the model's current parameters and the training state to
        a checkpoint

        the training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far
        and optimizer and scheduler state

        :return:
        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.step)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_scores": self.best_ckpt_scores,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if
            self.scheduler is not None else None,
        }

        torch.save(state, model_path)

        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning("Wanted to delete old checkpoint %s but "
                                    "file does not exist.", to_delete)
        self.ckpt_queue.put(model_path)

        best_path = "{}/best.ckpt".format(self.model_dir)
        try:
            # create/modify symbolic link for best checkpoint
            symlink_update("{}.ckpt".format(self.steps), best_path)
        except OSError:
            # overwrite best.ckpt
            torch.save(state, best_path)

    def init_from_checkpoint(self, path: str,
                             reset_best_ckpt: bool = False,
                             reset_scheduler: bool = False,
                             reset_optimizer: bool = False):
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also scheduler
        and optimizer states, see "self._save_checkpoint"


        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint, use for
                        domain adaptation with a new dev set or when using a new
                        metric for fine-tune
        :param reset_scheduler: reset the learning rate scheduler, and do not use the one
                                stored in the checkpoint
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint
        :return:
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint['model_state'])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint['optimizer_sate'])
        else:
            self.logger.info("Reset optimizer.")

        if not reset_scheduler:
            if model_checkpoint['scheduler_state'] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(model_checkpoint['scheduler_state'])
        else:
            self.logger.info("Reset scheduler.")

        # restore counts
        self.step = model_checkpoint['steps']
        self.total_tokens = model_checkpoint['total_tokens']

        if not reset_best_ckpt:
            self.best_ckpt_scores = model_checkpoint["best_ckpt_score"]
            self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]
        else:
            self.logger.info("Reset tracking of the best checkpoint.")

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log
        """
        model_parameters = filter(
            lambda p: p.requires_grad, self.model.paramters()
        )
        # 一共有多少个参数
        n_params = sum(
            [np.prod(p.size) for p in model_parameters]
        )
        self.logger.info(" Total params: %d", n_params)

        trainable_params = [n for (n, p) in self.model.named_parameters() if p.requires_grad]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))

        assert trainable_params

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset):
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        train_iter = make_data_iter(train_data,
                                    batch_size=self.batch_size,
                                    batch_type=self.batch_type,
                                    train=True, shuffle=self.shuffle)

        # for last batch in epoch batch_multiplier needs to be adjusted
        # to fit the number of leftover training examples
        leftover_batch_size = len(train_data) % (self.batch_multiplier * self.batch_size)

        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            # reset statistics for each epoch
            start = time.time()
            total_valid_duration = 0
            start_tokens = self.total_tokens
            self.current_batch_multiplier = self.batch_multiplier
            count = self.current_batch_multiplier - 1
            epoch_loss = 0

            for i, batch in enumerate(iter(train_iter)):
                # reactivate training
                self.model.train()
                # create a batch object from torchtext batch
                batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)

                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/
                # increasing-mini-batch-size-without-increasing-
                # memory-6794e10db672

                # Set current_batch_mutliplier to fit
                # number of leftover examples for last batch in epoch
                if self.batch_multiplier > 1 and i == len(train_iter) - math.ceil(
                        leftover_batch_size / self.batch_size):
                    self.current_batch_multiplier = math.ceil(
                        leftover_batch_size / self.batch_size)
                    count = self.current_batch_multiplier - 1


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
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection


if __name__ == '__main__':
    parser = argparse.ArgumentParser('joeynmt')
    parser.add_argument("config", default="configs/default.yaml",
                        type=str, help='Training configuration file (yaml).')
    args = parser.parse_args()
    train(cfg_file=args.config)
