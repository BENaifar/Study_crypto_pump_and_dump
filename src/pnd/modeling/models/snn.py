"""SNN: positional attention over the channel's pump sequence. [orig: TargetCoinPrediction/SeqModel/modeling.py:L52-L134]"""

import tensorflow.compat.v1 as tf

from pnd.modeling.layers import layer_norm
from pnd.modeling.models.dnn import DNN

tf.disable_v2_behavior()


class SNN(DNN):
    def __init__(self, tensor_dict, model_config):
        super(SNN, self).__init__(tensor_dict, model_config)

        self.length = tensor_dict["length"]
        # QUIRK(Q31/Q30): last 9 "features" of the per-character sequence split [orig: L57]
        self.seq_embedding = tensor_dict["seq_embedding"][:, :, -9:]
        self.seq_coin_embedding = tensor_dict["seq_coin_embedding"]
        self.coin_embedding = tensor_dict["coin_embedding"]

    def positional_attention_layer(self):

        feat_num = self.seq_embedding.shape[2]
        # QUIRK(Q31): a tf.Variable is created inside the length clip [orig: L63-L64]
        self.length = tf.where(tf.less_equal(self.length, self.model_config["max_seq_length"]),
                               self.length, tf.Variable([self.model_config["max_seq_length"] for i in range(self.model_config["batch_size"])]))

        # QUIRK(Q31): maxlen hard-coded to 50 [orig: L66]
        self.sequence_mask = tf.sequence_mask(self.length, maxlen=50, name="sequence_mask")

        mask_2d = tf.expand_dims(self.sequence_mask, axis=2)
        masked_seq_embedding = self.seq_embedding * tf.cast(mask_2d, tf.float32)  # convert bool to float

        # use f(x) to speed training
        pos_atten_list = []
        seq_pos_attent_embed_list = []
        num_head = 8

        for i in range(feat_num):

            x = tf.expand_dims(masked_seq_embedding[:, :, i], axis=2)
            inp = layer_norm(x, name="layer_norm_" + str(i))  # add layer norm

            # positional attention
            pos_atten = tf.get_variable("pos_atten_" + str(i), [1, 50, num_head],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))
            pos_atten = tf.layers.dense(pos_atten, 8, activation=tf.nn.relu, name="pos_" + str(i) + "_fx1")
            pos_atten = tf.layers.dense(pos_atten, num_head, activation=None, name="pos_" + str(i) + "_fx2")

            pos_atten = tf.tile(pos_atten, [self.model_config["batch_size"], 1, 1])
            paddings = tf.ones_like(pos_atten) * (-2 ** 32 + 1)

            mask_new = tf.tile(mask_2d, [1, 1, num_head])
            pos_atten = tf.where(tf.logical_not(mask_new), paddings, pos_atten)
            pos_atten = tf.nn.softmax(pos_atten, axis=1)

            seq_pos_attent_embed = tf.reduce_sum(pos_atten * inp, axis=1)
            seq_pos_attent_embed_list.append(seq_pos_attent_embed)
            pos_atten_list.append(tf.expand_dims(pos_atten, axis=2))

        self.pos_atten = tf.concat(pos_atten_list, axis=2)
        self.seq_pos_attent_embeding = tf.concat(seq_pos_attent_embed_list, axis=1)

        # for coin embedding
        coin_mask_2d = tf.expand_dims(self.sequence_mask, axis=2)
        masked_seq_coin_embedding = self.seq_coin_embedding * tf.cast(coin_mask_2d, tf.float32)  # convert bool to float
        coin_inp = layer_norm(masked_seq_coin_embedding, name="layer_norm_coin_embedding")  # add layer norm

        num_head = 8
        pos_atten_coin = tf.get_variable("pos_atten_coin", [1, 50, num_head],
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))

        # QUIRK(Q31): coin attention fx1 uses num_head units (feature attention uses 8) [orig: L111]
        pos_atten_coin = tf.layers.dense(pos_atten_coin, num_head, activation=tf.nn.relu, name="pos_coin_fx1")
        pos_atten_coin = tf.layers.dense(pos_atten_coin, num_head, activation=None, name="pos_coin_fx2")

        pos_atten_coin = tf.tile(pos_atten_coin, [self.model_config["batch_size"], 1, 1])
        paddings = tf.ones_like(pos_atten_coin) * (-2 ** 32 + 1)

        coin_mask_new = tf.tile(coin_mask_2d, [1, 1, num_head])
        pos_atten_coin = tf.where(tf.logical_not(coin_mask_new), paddings, pos_atten_coin)
        pos_atten_coin = tf.nn.softmax(pos_atten_coin, axis=1)

        seq_coin_pos_attent_embeding = tf.reduce_sum(tf.expand_dims(pos_atten_coin, axis=2) * tf.expand_dims(coin_inp, axis=3), axis=1)
        self.seq_coin_pos_attent_embeding = tf.reshape(seq_coin_pos_attent_embeding, [self.model_config["batch_size"], -1])
        output = tf.concat([self.seq_pos_attent_embeding, self.seq_coin_pos_attent_embeding], axis=1)

        return output

    def build(self):
        """
        override the build function
        """
        self.seq_embedding_mean = self.positional_attention_layer()
        self.inp = tf.concat([self.target_features, self.seq_embedding_mean, self.coin_embedding], axis=1)
        self.build_fcn_net(self.inp)
        self.loss_op()
