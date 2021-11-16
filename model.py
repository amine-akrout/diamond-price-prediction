##
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

##
dataframe = pd.read_csv('./data/diamonds.csv')

##
dataframe.head()

##
dataframe.info()

##
dataframe = dataframe.drop(columns=['Unnamed: 0'])

##
train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])

##
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('price')
  df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

##
batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)

##
[(train_features, label_batch)] = train_ds.take(1)
print('Every feature:', list(train_features.keys()))
print('A batch of carats:', train_features['carat'])
print('A batch of targets:', label_batch )

##
def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

##
carat_count_col = train_features['carat']
layer = get_normalization_layer('carat', train_ds)
layer(carat_count_col)

##
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))

##
test_clarity_col = train_features['clarity']
test_clarity_layer = get_category_encoding_layer(name='clarity',
                                              dataset=train_ds,
                                              dtype='string')
test_clarity_layer(test_clarity_col)

## Preprocess selected features to train the model on
batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

##
all_inputs = []
encoded_features = []

# Numerical features.
for header in ['carat', 'depth', 'table','x', 'y', 'z']:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  normalization_layer = get_normalization_layer(header, train_ds)
  encoded_numeric_col = normalization_layer(numeric_col)
  all_inputs.append(numeric_col)
  encoded_features.append(encoded_numeric_col)

# Categorical features.
categorical_cols = ['cut', 'color', 'clarity']

for header in categorical_cols:
  categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
  encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=train_ds,
                                               dtype='string',
                                               max_tokens=5)
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)


##
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(32, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

##
def rmse(y_true, y_pred):
  return tf.sqrt(tf.reduce_mean((y_pred - y_true) ** 2))
##

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=[rmse, "mse"])

##
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

##
history= model.fit(train_ds, epochs=200, validation_data=val_ds)

##
loss, rmse, mse = model.evaluate(test_ds)
print("RMSE", rmse)
##
import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'rmse')
plot_graphs(history, 'loss')

## save model
model.save('diamond_price_predictor')
reloaded_model = tf.keras.models.load_model('diamond_price_predictor', custom_objects={'rmse':rmse})

# Perform inference
sample = {
    'carat': 0.23,
    'cut': 'Ideal',
    'color': 'E',
    'clarity': 'SI2',
    'depth': 61.5,
    'table': 55.0,
    'x': 3.95,
    'y': 3.98,
    'z': 2.43,
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = reloaded_model.predict(input_dict)
predicted = predictions[0]

print("This particular pet had a %.1f percent probability "
    "of getting adopted.", predicted[0])