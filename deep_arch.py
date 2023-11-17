import pandas as pd
import csv
import numpy as np
import json
import tensorflow as tf
import pickle
from tensorflow import keras
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# this function loads data (embeddings) to be trained/test in a unique matrix X
# whose values are then fitted by the deep model
def matching_graph_bert_ids(users, items, ratings, graph_embs, word_embs):

  nu = []
  ni = []
  nr = []

  y_original = np.array(ratings)

  dim_embeddings = len(list(graph_embs.values())[0])

  dim_X_cols = 4
  dim_X_rows = len(users)

  X_rows = 0
  i = 0
  while i < dim_X_rows:

    user_id = users[i]
    item_id = items[i]

    check = int(user_id) in graph_embs and int(user_id) in word_embs and int(item_id) in graph_embs and int(item_id) in word_embs

    if check:
      X_rows += 1

    i += 1

  X = np.empty(shape=(X_rows,dim_X_cols,dim_embeddings))
  y = np.empty(shape=(X_rows))
  print("Loading embeddings to be fitted/tested...")

  i=0
  c=0

  while i < dim_X_rows:

    user_id = users[i]
    item_id = items[i]

    check = int(user_id) in graph_embs and int(user_id) in word_embs and int(item_id) in graph_embs and int(item_id) in word_embs

    if check:

      user_graph_emb = np.array(graph_embs[int(user_id)])
      user_word_emb = np.array(word_embs[int(user_id)])
      item_graph_emb = np.array(graph_embs[int(item_id)])
      item_word_emb = np.array(word_embs[int(item_id)])

      X[c][0] = user_graph_emb
      X[c][1] = item_graph_emb
      X[c][2] = user_word_emb
      X[c][3] = item_word_emb

      y[c] = y_original[i]
      
      nu.append(users[i])
      ni.append(items[i])
      nr.append(ratings[i])

      c += 1

    i += 1

  return X[0:c], y[0:c], dim_embeddings, nu, ni, nr


def read_ratings(filename):

  user=[]
  item=[]
  rating=[]

  with open(filename) as csv_file:

    csv_reader = csv.reader(csv_file, delimiter='\t')

    for row in csv_reader:
        user.append(int(row[0]))
        item.append(int(row[1]))
        rating.append(int(row[2]))

  return user, item, rating

def top_scores(predictions,n):

  top_n_scores = pd.DataFrame()

  for u in list(set(predictions['users'])):
    p = predictions.loc[predictions['users'] == u ]
    top_n_scores = top_n_scores.append(p.head(n))

  return top_n_scores


def model_entity_based(X,y,dim_embeddings,epochs,batch_size):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_1)
  x1_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x1_user)
  x1_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x1_2_user)

  x1_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_1)
  x1_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x1_item)
  x1_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x1_2_item)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user = keras.layers.Dense(256, activation=tf.nn.relu)(input_users_2)
  x2_2_user = keras.layers.Dense(128, activation=tf.nn.relu)(x2_user)
  x2_3_user = keras.layers.Dense(64, activation=tf.nn.relu)(x2_2_user)

  x2_item = keras.layers.Dense(256, activation=tf.nn.relu)(input_items_2)
  x2_2_item = keras.layers.Dense(128, activation=tf.nn.relu)(x2_item)
  x2_3_item = keras.layers.Dense(64, activation=tf.nn.relu)(x2_2_item)
  
  concatenated_1 = keras.layers.Concatenate()([x1_3_user, x2_3_user])
  dense_user = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_1)
  dense_user_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_user)
  
  concatenated_2 = keras.layers.Concatenate()([x1_3_item, x2_3_item])
  dense_item = keras.layers.Dense(64, activation=tf.nn.relu)(concatenated_2)
  dense_item_2 = keras.layers.Dense(32, activation=tf.nn.relu)(dense_item)

  concatenated = keras.layers.Concatenate()([dense_user_2, dense_item_2])
  dense = keras.layers.Dense(32, activation=tf.nn.relu)(concatenated)
  dense2 = keras.layers.Dense(16, activation=tf.nn.relu)(dense)
  dense3 = keras.layers.Dense(8, activation=tf.nn.relu)(dense2)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(dense3)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  
  return model


def model_entity_dropout_selfatt_crossatt(X,y,dim_embeddings,epochs,batch_size, value):

  model = keras.Sequential()

  input_users_1 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_1 = keras.layers.Input(shape=(dim_embeddings,))

  x1_user_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_users_1)
  x1_item_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_items_1)

  x1_user = keras.layers.Dense(512, activation=tf.nn.relu)(x1_user_drop)
  x1_2_user = keras.layers.Dense(256, activation=tf.nn.relu)(x1_user)
  x1_3_user = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2_user)

  x1_item = keras.layers.Dense(512, activation=tf.nn.relu)(x1_item_drop)
  x1_2_item = keras.layers.Dense(256, activation=tf.nn.relu)(x1_item)
  x1_3_item = keras.layers.Dense(128, activation=tf.nn.relu)(x1_2_item)

  input_users_2 = keras.layers.Input(shape=(dim_embeddings,))
  input_items_2 = keras.layers.Input(shape=(dim_embeddings,))

  x2_user_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_users_2)
  x2_item_drop = keras.layers.Dropout(value, input_shape=(dim_embeddings,))(input_items_2)

  x2_user = keras.layers.Dense(512, activation=tf.nn.relu)(x2_user_drop)
  x2_2_user = keras.layers.Dense(256, activation=tf.nn.relu)(x2_user)
  x2_3_user = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2_user)

  x2_item = keras.layers.Dense(512, activation=tf.nn.relu)(x2_item_drop)
  x2_2_item = keras.layers.Dense(256, activation=tf.nn.relu)(x2_item)
  x2_3_item = keras.layers.Dense(128, activation=tf.nn.relu)(x2_2_item)

  # self attenzione 1 - merge graph user e word user
  concat_user = keras.layers.Concatenate()([x1_3_user, x2_3_user])
  attention_w_user = keras.layers.Dense(128, activation='softmax')(concat_user)
  merged_user = attention_w_user * x1_3_user + (1 - attention_w_user) * x2_3_user

  # self attenzione 2 - merge graph item e word item
  concat_item = keras.layers.Concatenate()([x1_3_item, x2_3_item])
  attention_w_item = keras.layers.Dense(128, activation='softmax')(concat_item)
  merged_item = attention_w_item * x1_3_item + (1 - attention_w_item) * x2_3_item

  # cross attenzione - merge dei due merged
  attention_weights = keras.layers.Dot(axes=-1)([merged_user, merged_item])
  attention_weights = keras.layers.Dense(128, activation='softmax')(attention_weights)
  merged = keras.layers.Add()([merged_user * attention_weights, merged_item * (1 - attention_weights)])

  merged2 = keras.layers.Dense(64, activation=tf.nn.relu)(merged)
  merged3 = keras.layers.Dense(32, activation=tf.nn.relu)(merged2)
  merged4 = keras.layers.Dense(16, activation=tf.nn.relu)(merged3)
  merged5 = keras.layers.Dense(8, activation=tf.nn.relu)(merged4)
  out = keras.layers.Dense(1, activation=tf.nn.sigmoid)(merged5)

  model = keras.models.Model(inputs=[input_users_1,input_items_1,input_users_2,input_items_2],outputs=out)
  model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9), metrics=['accuracy'])
  model.fit([X[:,0],X[:,1],X[:,2],X[:,3]], y, epochs=epochs, batch_size=batch_size)
  
  return model

# for this implementation, graph embeddings and word embeddings 
# are dict in the form:
# id -> embedding
# where id is an integer
# embedding is a list of float or numpy.array

source_graph_path = 'path/to/graph_emb.pickle'
source_text_path = 'path/to/word_emb.pickle'
model_path = 'path/to/model.h5'
predictions_path = 'path/to/predictions'


# read training data
users, items, ratings = read_ratings('movielens/train.tsv')

# read graph and word embedding
graph_emb = pickle.load(open(source_graph_path, 'rb'))
word_emb = pickle.load(open(source_text_path, 'rb'))


# il the model already exixts, it's loaded
if os.path.exists(model_path):

  recsys_model = tf.keras.models.load_model(model_path)
  print("Model loaded.")

# otherwise it's trained
else:

  print("Matched data for training...")
  X, y, dim_embeddings, _, _, _ = matching_graph_bert_ids(users, items, ratings, graph_emb, word_emb)
  
  # training the model
  recsys_model = run_layers_entity_dropout_selfatt_crossatt(X,y,dim_embeddings,epochs=30,batch_size=512, value=0.7)

  # saving the model
  recsys_model.save(model_path)

# read test ratings to be predicted
users, items, ratings = read_ratings('movielens/test.tsv')

# embeddings for test
X, y, dim_embeddings, nu, ni, nr = matching_graph_bert_ids(users, items, ratings, graph_emb, word_emb)

# predict   
print("\tPredicting...")
score = recsys_model.predict([X[:,0],X[:,1],X[:,2],X[:,3]])

# write predictions
print("\tComputing predictions...")
score = score.reshape(1, -1)[0,:]
predictions = pd.DataFrame()
predictions['users'] = np.array(nu)
predictions['items'] = np.array(ni)
predictions['scores'] = score

predictions = predictions.sort_values(by=['users', 'scores'],ascending=[True, False])

# create predictions folder if it does not exist
if not os.path.exists(predictions_path):
  os.mkdir(predictions_path)

# write top 5 predictions
top_5_scores = top_scores(predictions,5)
top_5_scores.to_csv(predictions_path + '/top5_predictions.tsv',sep='\t',header=False,index=False)
print("\tTop 5 wrote.")
