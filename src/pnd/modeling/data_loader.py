"""tf.data input pipeline + embeddings. [orig: TargetCoinPrediction/SeqModel/data_loader.py]"""

import pickle as pkl

import tensorflow.compat.v1 as tf

from pnd.features.columns import COLUMN_LIST  # noqa: F401  (same list as the original module constant)

tf.disable_v2_behavior()

column_list = COLUMN_LIST


class FeatGenerator(object):

    def __init__(self, input_file, wv_embedding_path, epoch=1, batch_size=256, shuffle_seed=None):
        self.input_file = input_file
        # R3a: embedding path injected (was "../FeatGeneration/feature/wv_embedding.pkl")
        self.wv_embedding_path = str(wv_embedding_path)
        # D5: seed for dataset.shuffle(100); None = original unseeded behaviour
        self.shuffle_seed = shuffle_seed
        # QUIRK(Q30): fixed feature config [orig: L13-L23]
        self.feat_config = {
            "n_channel": 200,
            "d_channel": 8,
            "n_coin": 378,
            "d_coin": 8,
            "n_seq_feat": 147,
            "n_target_feat": 138,
            "max_length": 50,
            "batch_size": batch_size,
            "epoch": epoch,
        }

        self.features = self.feature_generation()
        self.tensor_dict = self.tensor_generation(self.features)

    def parse_split(self, line):
        parse_res = tf.string_split([line], delimiter=",")
        values = parse_res.values
        label = values[0]
        channel = values[1]
        channel_id = values[2]
        coin = values[3]
        coin_id = values[4]
        time_stamp = values[5]
        length = values[6]
        coin_id_seq = values[7]
        feature_seq = values[8]
        feature_target = values[9:]
        return label, channel, channel_id, coin, coin_id, time_stamp, length, coin_id_seq, feature_seq, feature_target

    def parse_sequence(self, sequence):
        """
        split the sequence and convert to dense tensor
        """
        split_sequence = tf.string_split(sequence, delimiter="\t")
        split_sequence = tf.sparse_to_dense(sparse_indices=split_sequence.indices,
                                            output_shape=[self.feat_config["batch_size"],
                                                          self.feat_config["max_length"]],
                                            sparse_values=split_sequence.values, default_value="0")

        return split_sequence

    def parse_feature_sequence(self, sequence):
        split_sequence = tf.string_split(sequence, delimiter="\t")
        split_sequence1 = tf.sparse_to_dense(sparse_indices=split_sequence.indices,
                                             output_shape=[self.feat_config["batch_size"],
                                                           self.feat_config["max_length"]],
                                             sparse_values=split_sequence.values,
                                             default_value="".join(["0" for i in range(self.feat_config["n_seq_feat"])]))

        split_sequence1 = tf.reshape(split_sequence1, shape=[self.feat_config["batch_size"] * self.feat_config["max_length"]])

        # QUIRK(Q30): delimiter "" splits every feature string into single characters [orig: L66]
        split_sequence2 = tf.string_split(split_sequence1, delimiter="")
        split_sequence2 = tf.sparse_to_dense(sparse_indices=split_sequence2.indices,
                                             output_shape=[self.feat_config["batch_size"] * self.feat_config["max_length"],
                                                           self.feat_config["n_seq_feat"]],
                                             sparse_values=split_sequence2.values,
                                             default_value="0")

        split_sequence_final = tf.reshape(split_sequence2, shape=[-1, self.feat_config["max_length"], self.feat_config["n_seq_feat"]])

        return split_sequence_final

    def feature_generation(self):
        """
        Args:
            input_file: a .txt file that includes the training or testing sample
        Returns:
            feature tensor used for training or testing
        """
        dataset = tf.data.TextLineDataset(self.input_file)
        dataset = dataset.map(self.parse_split, num_parallel_calls=2)
        dataset = dataset.shuffle(100, seed=self.shuffle_seed).repeat(self.feat_config["epoch"]).batch(self.feat_config["batch_size"])
        iterator = dataset.make_one_shot_iterator()

        label, channel, channel_id, coin, coin_id, time_stamp, \
            length, coin_id_seq, feature_seq, feature_target = iterator.get_next()

        seq_coin_id = self.parse_sequence(coin_id_seq)
        seq_feature = self.parse_feature_sequence(feature_seq)

        features = {}
        features["channel"] = channel
        features["channel_id"] = tf.string_to_number(channel_id, out_type=tf.int32)
        features["coin"] = coin
        features["coin_id"] = tf.string_to_number(coin_id, out_type=tf.int32)

        features["timestamp"] = time_stamp
        features["label"] = tf.string_to_number(label, out_type=tf.float32)
        features["target_features"] = tf.reshape(tf.string_to_number(feature_target, out_type=tf.float32), [-1, self.feat_config["n_target_feat"]])

        features["seq_coin_id"] = tf.string_to_number(seq_coin_id, out_type=tf.int32)
        features["seq_feature"] = tf.string_to_number(seq_feature, out_type=tf.float32)
        features["length"] = tf.string_to_number(length, out_type=tf.int32)

        return features

    def tensor_generation(self, features):
        with tf.name_scope('Embedding_layer'):

            channel_lookup_table = tf.get_variable("channel_embedding_var", [self.feat_config["n_channel"], self.feat_config["d_channel"]])
            channel_embedding = tf.nn.embedding_lookup(channel_lookup_table, features["channel_id"])

            with open(self.wv_embedding_path, "rb") as f:
                wv_embedding = pkl.load(f)

            # QUIRK(Q30): frozen pre-trained coin embedding (constant, not a Variable) [orig: L120]
            coin_lookup_table = tf.constant(wv_embedding, name="coin_embedding_var", dtype=tf.float32)

            coin_embedding = tf.nn.embedding_lookup(coin_lookup_table, features["coin_id"])
            seq_coin_embedding = tf.nn.embedding_lookup(coin_lookup_table, features["seq_coin_id"])

            # concatenate the tensors
            tensor_dict = {}

            tensor_dict["label"] = features["label"]
            tensor_dict["length"] = features["length"]
            tensor_dict["channel_embedding"] = channel_embedding
            tensor_dict["coin_embedding"] = coin_embedding
            tensor_dict["target_features"] = features["target_features"]
            tensor_dict["seq_embedding"] = features["seq_feature"]
            tensor_dict["seq_coin_embedding"] = seq_coin_embedding

        return tensor_dict
