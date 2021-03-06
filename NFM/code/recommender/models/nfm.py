import tensorflow as tf
import tensorflow.contrib.eager as tfe

from recommender.models.model import Model


class NeuralFM(Model):
    def __init__(
        self,
        ds,
        num_units=64,
        layers=None,
        dropout_prob=None,
        apply_batchnorm=True,
        activation_fn="relu",
        apply_dropout=True,
        l2_regularizer=0.0,
        apply_nfm=True,
    ):
        super(NeuralFM, self).__init__()
        self._num_weights = ds.num_features_one_hot
        self._num_units = num_units
        self._num_features = ds.num_features
        self.apply_nfm = apply_nfm

        if layers and dropout_prob and apply_dropout:
            assert len(layers) + 1 == len(dropout_prob)

        if layers is None:
            layers = [64]

        if dropout_prob is None:
            dropout_prob = [0.5, 0.5]

        self.dropout_prob = dropout_prob

        self.apply_batchnorm = apply_batchnorm
        self.apply_dropout = apply_dropout
        self.activation = activation_fn
        self.dense_layers = [
            tf.keras.layers.Dense(units, activation=self.activation) for units in layers
        ]
        self.final_dense_layer = tf.keras.layers.Dense(1)
        self.fm_layer = tf.keras.layers.Dense(num_units)
        if self.apply_batchnorm:
            self.batch_norm_layer = tf.keras.layers.BatchNormalization()
            self.dense_batch_norm = [
                tf.keras.layers.BatchNormalization() for _ in layers
            ]

        if self.apply_dropout:
            self.fm_dropout = tf.keras.layers.Dropout(self.dropout_prob[-1])
            self.dense_dropout = [
                tf.keras.layers.Dropout(self.dropout_prob[i])
                for i in range(len(dropout_prob) - 1)
            ]

        self.w = tf.keras.layers.Embedding(
            self._num_weights,
            num_units,
            input_length=self._num_features,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01
            ),
            embeddings_regularizer=tf.keras.regularizers.l2(l2_regularizer),
        )
        self.w0 = tf.keras.layers.Embedding(
            self._num_weights,
            1,
            input_length=self._num_features,
            embeddings_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.01
            ),
        )
        self.bias = tfe.Variable(tf.constant(0.0))

    def call(self, one_hot_features, training=False, features=None, **kwargs):
        """
        Args:
            one_hot_features: A dense tensor of shape [batch_size, self._num_features]
                that indicates which features are present in this input.
            training: A boolean indicating if is training or not.
            features: A dense tensor of shape [batch_size, self._num_features] that indicates
                the value of each feature.
        Returns:
            Logits.
        """
        # TODO: add support to other features.

        # FM
        weights = self.w(one_hot_features)  # [batch_size, num_features, num_units]

        sum_nzw = tf.reduce_sum(weights, 1)  # [batch_size, num_units]
        squared_sum_nzw = tf.square(sum_nzw)  # [batch_size, num_units]

        squared_nzw = tf.square(weights)  # [batch_size, num_features, num_units]
        sum_squared_nzw = tf.reduce_sum(squared_nzw, 1)  # [batch_size, num_units]

        fm = 0.5 * (squared_sum_nzw - sum_squared_nzw)
        '''
        a = tf.reduce_sum(self.w0(one_hot_features), 1)
        b = a
        for i in range(self._num_units - 1):
            b = tf.concat([b,a],axis=1)
        fm = tf.add_n([fm, b]) + self.bias
        '''
        if self.apply_nfm:
            fm_first = self.fm_layer(tf.reduce_sum(self.w0(one_hot_features) , 1))
            fm = fm_first + fm + self.bias

        if self.apply_batchnorm:
            fm = self.batch_norm_layer(fm, training=training)

        if self.apply_dropout:
            fm = self.fm_dropout(fm, training=training)

        # Dense layers on top of FM
        for i, layer in enumerate(self.dense_layers):
            fm = layer(fm)
            if self.apply_batchnorm:
                fm = self.dense_batch_norm[i](fm, training=training)
            if self.apply_dropout:
                fm = self.dense_dropout[i](fm, training=training)

        # Aggregate
        fm = self.final_dense_layer(fm)  # [batch_size, 1]
        bilinear = tf.reduce_sum(fm, 1, keep_dims=True)  # [batch_size, 1]
        if not self.apply_nfm:
            weight_bias = tf.reduce_sum(self.w0(one_hot_features), 1)  # [batch_size, 1]
            bilinear = weight_bias + bilinear + self.bias
            
        logits = bilinear

        return logits
