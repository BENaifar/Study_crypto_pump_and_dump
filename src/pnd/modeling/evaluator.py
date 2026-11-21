"""Evaluation loop. [orig: TargetCoinPrediction/SeqModel/run_train.py:L175-L234 + run_eval.py]

D8: the ``run_eval.py`` flag-deletion hack is replaced by this stage (do_train=False, do_eval=True).
"""

import logging
import os

import tensorflow.compat.v1 as tf
from sklearn import metrics

from pnd.common.seeding import set_global_seed
from pnd.config import Config
from pnd.modeling.data_loader import FeatGenerator
from pnd.modeling.metrics import HitRatio_calculation
from pnd.modeling.trainer import checkpoint_dir, model_fn, reset_graph
from pnd.pipeline.registry import register

tf.disable_v2_behavior()

logger = logging.getLogger(__name__)


@register
class EvaluateStage:
    """``--do_eval`` branch: AUC + HitRatio for each checkpoint epoch 15..30."""

    name = "evaluate"

    def run(self, cfg: Config) -> None:
        set_global_seed(cfg.seed)
        m = cfg.model
        reset_graph(cfg.seed)

        feat_generator = FeatGenerator(cfg.paths.model.test_input, cfg.paths.model.wv_embedding, 1, m.batch_size,
                                       shuffle_seed=cfg.seed)
        model = model_fn(feat_generator.tensor_dict, m, do_train=False)
        ckpt_dir = checkpoint_dir(cfg)

        auc_value_list = []
        # QUIRK(Q33): only checkpoints 15..30 are evaluated [orig: L179]
        for epoch in range(m.eval_epoch_start, m.eval_epoch_end_inclusive + 1):

            ckpt = os.path.join(str(ckpt_dir), m.model + str(epoch))
            saver = tf.train.Saver()

            pre_probas = []
            labels = []
            coin_ids = []
            channel_ids = []
            timestamps = []

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                try:
                    saver.restore(sess, ckpt)
                except Exception as e:
                    print(e)
                    break

                iter = 0
                while True:

                    try:
                        channel_id, coin_id, timestamp, y_, label, loss = \
                            sess.run([feat_generator.features["channel"],
                                      feat_generator.features["coin"],
                                      feat_generator.features["timestamp"],
                                      model.y_hat,
                                      model.label,
                                      model.loss])

                        pre_probas += list(y_)
                        labels += list(label)
                        timestamps += list(timestamp)
                        channel_ids += list(channel_id)
                        coin_ids += list(coin_id)

                        iter += 1
                    except Exception:
                        break

            fpr, tpr, thresholds_roc = metrics.roc_curve(y_true=labels, y_score=pre_probas, pos_label=1)
            auc_value = metrics.auc(fpr, tpr)
            auc_value_list.append(auc_value)

            print("=====================================")
            print("epoch=%d, auc=%f" % (epoch, auc_value))
            HR1, HR3, HR5, HR10, HR20, HR30 = HitRatio_calculation(channel_ids, coin_ids, timestamps, labels, pre_probas)
            print("HitRate@1=%.4f" % HR1)
            print("HitRate@3=%.4f" % HR3)
            print("HitRate@5=%.4f" % HR5)
            print("HitRate@10=%.4f" % HR10)
            print("HitRate@20=%.4f" % HR20)
            # QUIRK(Q33): value is HR@50 [orig: L234]
            print("HitRate@30=%.4f" % HR30)
