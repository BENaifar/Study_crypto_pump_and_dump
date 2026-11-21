"""Training loop. [orig: TargetCoinPrediction/SeqModel/run_train.py:L94-L172]"""

import logging
import os
import time
from pathlib import Path

import tensorflow.compat.v1 as tf
from sklearn import metrics

from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.modeling import optimization  # noqa: F401  QUIRK(Q34): imported, never used
from pnd.modeling.data_loader import FeatGenerator
from pnd.modeling.models.registry import build_model
from pnd.pipeline.registry import register

tf.disable_v2_behavior()

logger = logging.getLogger(__name__)


def model_fn(tensor_dict, m, do_train: bool):
    """[orig: run_train.py:L94-L116]"""
    model_config = {"is_training": do_train,
                    "dropout_rate": m.dropout_rate if do_train else 0.0,
                    "max_seq_length": m.max_seq_length,
                    "learning_rate": m.learning_rate,
                    "batch_size": m.batch_size}
    return build_model(m.model, tensor_dict, model_config)


def reset_graph(seed) -> None:
    """New default graph, then D5 graph-level seed."""
    tf.reset_default_graph()
    if seed is not None:
        tf.set_random_seed(seed)


def checkpoint_dir(cfg: Config) -> Path:
    return Path(cfg.paths.model.checkpoint_root) / cfg.model.checkpointDir


@register
class TrainStage:
    """``--do_train`` branch of run_train.py."""

    name = "train"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        m = cfg.model
        reset_graph(cfg.seed)

        sample_num = m.sample_num
        # feature generation
        feat_generator = FeatGenerator(cfg.paths.model.train_input, cfg.paths.model.wv_embedding,
                                       m.epoch, m.batch_size, shuffle_seed=cfg.seed)

        # build model
        model = model_fn(feat_generator.tensor_dict, m, do_train=True)

        # define saver
        saver = tf.train.Saver(max_to_keep=50)
        # QUIRK(Q32): save interval from the hard-coded sample_num=132314 [orig: L120, L136]
        save_iter = int(sample_num / m.batch_size)
        ckpt_dir = checkpoint_dir(cfg)
        ckpt_dir.mkdir(parents=True, exist_ok=True)  # D1

        # intermediate results
        pred_probas = []
        labels = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iter = 0
            start = time.time()
            while True:
                # QUIRK(Q32): ANY exception (end of data, partial last batch, parse error) ends training [orig: L170-L172]
                try:
                    _, loss, l, y_ = sess.run([model.optimizer, model.loss, model.label, model.y_hat])

                    pred_probas += list(y_)
                    labels += list(l)

                    if iter % 50 == 0 and iter > 0:
                        end = time.time()
                        fpr, tpr, thresholds = metrics.roc_curve(labels, pred_probas, pos_label=1)
                        auc_value = metrics.auc(fpr, tpr)
                        logger.info("iter=%d, loss=%f, auc=%f, time=%.2fs" % (iter, loss, auc_value, end - start))
                        # clear
                        pred_probas = []
                        labels = []
                        start = time.time()

                    iter += 1

                    if iter % save_iter == 0 and iter > 4:
                        saver.save(sess, os.path.join(str(ckpt_dir), m.model + str(round(iter / save_iter))))

                except Exception as e:
                    print(e)
                    break
        logger.info("training stopped after %d iterations", iter)
