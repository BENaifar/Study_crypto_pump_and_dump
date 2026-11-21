"""Baseline DNN on the target features only. [orig: TargetCoinPrediction/SeqModel/modeling.py:L5-L50]"""

import logging

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

logger = logging.getLogger(__name__)


class DNN(object):

    def __init__(self, tensor_dict, model_config):
        self.label = tensor_dict["label"]
        self.channel_embedding = tensor_dict["channel_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]
        self.target_features = tensor_dict["target_features"]

        # configuration
        # QUIRK(Q31): hidden3 is declared but unused; dropout_rate is never applied [orig: L14-L19]
        self.model_config = {
            "hidden1": 32,
            "hidden2": 16,
            "hidden3": 32,
            "learning_rate": 0.0005
        }
        self.model_config.update(model_config)
        logger.info("Model Configuaration:")
        logger.info(self.model_config)

    def build(self):
        """
        build the architecture for the base DNN model.
        """
        self.inp = self.target_features
        self.build_fcn_net(self.inp)
        self.loss_op()

    def build_fcn_net(self, inp):
        with tf.name_scope("Fully_connected_layer"):

            dnn1 = tf.layers.dense(inp, self.model_config["hidden1"], activation=tf.nn.relu, name='f1')
            dnn2 = tf.layers.dense(dnn1, self.model_config["hidden2"], activation=tf.nn.relu, name='f2')

            self.logit = tf.squeeze(tf.layers.dense(dnn2, 1, activation=None, name='logit'))

        self.y_hat = tf.sigmoid(self.logit)

        return

    def loss_op(self):
        with tf.name_scope('Metrics'):

            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"]).minimize(self.loss)

        return
