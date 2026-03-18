import tensorflow as tf

# 載入資料
train_data = tf.keras.utils.image_dataset_from_directory(
    "PlantDataset",
    image_size=(224,224),
    batch_size=32
)

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(len(train_data.class_names),activation='softmax')
])

# 編譯
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練
model.fit(train_data, epochs=10)
model.save("model.h5")