"""SNN with target attention. [orig: TargetCoinPrediction/SeqModel/modeling.py:L136-L197]"""

import tensorflow.compat.v1 as tf

from pnd.modeling.layers import layer_norm, multihead_attention
from pnd.modeling.models.dnn import DNN

tf.disable_v2_behavior()


class SNNTA(DNN):

    def __init__(self, tensor_dict, model_config):
        super(SNNTA, self).__init__(tensor_dict, model_config)

        self.length = tensor_dict["length"]
        self.seq_embedding = tensor_dict["seq_embedding"][:, :, -9:]
        self.seq_coin_embedding = tensor_dict["seq_coin_embedding"]

        # QUIRK(Q31): only the "mlp" strategy is used [orig: L145]
        self.atten_strategy = "mlp"
        # self.atten_strategy = "dot"

    def positional_attention_layer(self):

        self.length = tf.where(tf.less_equal(self.length, self.model_config["max_seq_length"]), self.length,
                               tf.Variable([self.model_config["max_seq_length"] for i in range(self.model_config["batch_size"])]))
        self.sequence_mask = tf.sequence_mask(self.length, maxlen=50, name="sequence_mask")

        query = self.target_features[:, -9:]
        key = self.seq_embedding

        if self.atten_strategy == "mlp":
            query = tf.expand_dims(query, axis=1)
            queries = tf.tile(query, [1, key.shape[1], 1])
            attention_all = tf.concat([queries, key, queries - key, queries * key], axis=-1)

            d_layer_1_all = tf.layers.dense(attention_all, 32, activation=tf.nn.relu, name="f1_att")
            d_layer_2_all = tf.layers.dense(d_layer_1_all, 16, activation=tf.nn.relu, name="f2_att")
            d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name="f3_att")
            d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, key.shape[1]])

            scores = d_layer_3_all
            key_masks = tf.expand_dims(self.sequence_mask, 1)

            paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
            scores = tf.where(key_masks, scores, paddings)
            scores = tf.nn.softmax(scores, axis=2)
            output = tf.squeeze(tf.matmul(scores, key))

        elif self.atten_strategy == "dot":
            attended_embedding = multihead_attention(queries=tf.expand_dims(query, axis=1),
                                                     keys=key,
                                                     values=key,
                                                     num_heads=8,
                                                     key_masks=self.sequence_mask,
                                                     dropout_rate=self.model_config["dropout_rate"],
                                                     is_training=self.model_config["is_training"],
                                                     reuse=False)
            output = tf.squeeze(attended_embedding, axis=1)

        else:
            raise ValueError("Please choose the right attention strategy. Current is " + self.atten_strategy)

        return layer_norm(output, "layer_norm")

    def build(self):
        """
        override the build function
        """
        self.seq_embedding_mean = self.positional_attention_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_mean], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()
