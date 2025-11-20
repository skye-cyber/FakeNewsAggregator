import tensorflow as tf


def model_architecture(vocab_size, embedding_matrix, embedding_dim, max_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False,
            name='embedding'
        ),

        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0,  # Gradient clipping to prevent explosions
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
