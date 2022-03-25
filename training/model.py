# Import Libraries
from azureml.core import Run, Dataset
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
# import matplotlib.pyplot as plt
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import argparse

# Get the script arguments (dataset ID)
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='training_dataset_id', help='training dataset')
args = parser.parse_args()

# Get the experiment run context
run = Run.get_context()

# Import Data
# Get the training dataset
print("Loading Data...")
dataframe = run.input_datasets['training_data'].to_pandas_dataframe()

# dataframe = pd.read_csv('./data/diamonds.csv')
dataframe.head()
dataframe.info()

# add volume feature
# dataframe = dataframe.drop(columns=['Unnamed: 0'])
dataframe = dataframe.drop(columns=['Column1'])
dataframe = dataframe[(dataframe[['x','y','z']] != 0).all(axis=1)]
dataframe['volume'] = dataframe['x']*dataframe['y']*dataframe['z']

# split data
train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])

# create input function
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

# normalization layer
def get_normalization_layer(name, dataset):
  normalizer = layers.Normalization(axis=None)
  feature_ds = dataset.map(lambda x, y: x[name])
  normalizer.adapt(feature_ds)
  return normalizer

# category encoding feature
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)
  feature_ds = dataset.map(lambda x, y: x[name])
  index.adapt(feature_ds)
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())
  return lambda feature: encoder(index(feature))


# Preprocess selected features to train the model on
batch_size = 256
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

all_inputs = []
encoded_features = []

# feature engineering
# Numerical features.
for header in ['carat', 'depth', 'table','x', 'y', 'z', 'volume']:
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
                                               dtype='string')
  encoded_categorical_col = encoding_layer(categorical_col)
  all_inputs.append(categorical_col)
  encoded_features.append(encoded_categorical_col)


# Define the model
all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(256, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(8, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(1)(x)

model = tf.keras.Model(all_inputs, output)

# metrics for the model
METRICS = [tf.keras.metrics.RootMeanSquaredError(name='rmse'),
           tf.keras.metrics.MeanAbsolutePercentageError(name='mape'),
]

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mean_squared_error',
              metrics=[METRICS])

# plot the model
# tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# model training
history= model.fit(train_ds, epochs=10, validation_data=val_ds)

# model evaluation
loss, rmse, mape = model.evaluate(test_ds)
print("RMSE", rmse)
print("MAPE", mape)

run.log('RMSE', np.float(rmse))
run.log('MAPE', np.float(mape))

'''
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  run.log_image('Plot', plt)
  #plt.show()

# plot model performance
plot_graphs(history, 'rmse')
plot_graphs(history, 'mape')
plot_graphs(history, 'loss')
'''
# save model
os.makedirs('./outputs/model/', exist_ok=True)
model.save('outputs/model/')

#reloaded_model = tf.keras.models.load_model('diamond_price_predictor')
run.complete()

'''
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
volume = sample['x']*sample['y']*sample['z']
sample['volume'] =  volume

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = reloaded_model.predict(input_dict)
predicted = predictions[0]

print("estimated price :", predicted[0], "+/- 9%")
'''