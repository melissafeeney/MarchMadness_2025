#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 10:00:45 2025

@author: melissafeeney
"""

# -------------------------
# II. CREATE AND TRAIN MODELS
# -------------------------

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
import pickle 

# Run first for reproducability
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)


# -------------------------
# Data Processing
# -------------------------

final_model_data = pd.read_csv('/content/final_model_data_addl.csv')

# Split into X and Y
X = final_model_data.iloc[:, 4:-1].values

# 1. Split Data
X_train, X_val = train_test_split(X, test_size = 0.2, random_state = 123)

# 2. Preprocessing (Scaling after split)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val) # use the scaler already fit on the training data.
X_scaled = scaler.transform(X) # Scale the whole dataset for feature extraction at the end.

# Save the scaler for future use
scalerfile = 'scaler.save'
pickle.dump(scaler, open(scalerfile, 'wb'))

encoding_dim = 15
input_dim = X.shape[1]


# -------------------------
# Step I: Construct Auto Encoder Model
# -------------------------

# Encoder
encoder_input = tf.keras.layers.Input(shape=(input_dim,))
encoder_layer1 = tf.keras.layers.Dense(256, activation='relu')(encoder_input)
encoder_layer2 = tf.keras.layers.Dense(128, activation='relu')(encoder_layer1)
encoder_layer3 = tf.keras.layers.Dense(64, activation='relu')(encoder_layer2)
encoder_output = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoder_layer3)
encoder = tf.keras.Model(encoder_input, encoder_output)

# Decoder
decoder_input = tf.keras.layers.Input(shape=(encoding_dim,))
decoder_layer1 = tf.keras.layers.Dense(64, activation='relu')(decoder_input)
decoder_layer2 = tf.keras.layers.Dense(128, activation='relu')(decoder_layer1)
decoder_layer3 = tf.keras.layers.Dense(256, activation='relu')(decoder_layer2)
decoder_output = tf.keras.layers.Dense(input_dim, activation='linear')(decoder_layer3)
decoder = tf.keras.Model(decoder_input, decoder_output)

autoencoder_input = tf.keras.layers.Input(shape=(input_dim,))
autoencoder_output = decoder(encoder(autoencoder_input))
autoencoder = tf.keras.Model(autoencoder_input, autoencoder_output)

autoencoder.compile(optimizer = 'adam', loss = 'mse')

# Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = 10,  
    restore_best_weights = True
)

# LR Reduction Callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5,  
    patience = 5,  
    min_lr = 0.00001
)

# Train Autoencoder with Callbacks
autoencoder.fit(
    X_train_scaled,
    X_train_scaled,
    epochs = 50,
    batch_size = 512,
    validation_data = (X_val_scaled, X_val_scaled),
    callbacks = [early_stopping, reduce_lr],
    verbose = 2
)

# Extract Features using model
encoded_features = encoder.predict(X_scaled)

# This is now the new feature set
encoded_df = pd.DataFrame(encoded_features)

# Save auto encoder model as pkl file
with open('autoencoder.pkl', 'wb') as fid:
    pickle.dump(autoencoder, fid)


# Save encoder model as pkl file
with open('encoder.pkl', 'wb') as fid:
    pickle.dump(encoder, fid)
    
# -------------------------
## Step II: Model with Residual connection and Stratified K-Fold Cross Validation
# -------------------------

encoded_df = encoded_df.values
y = final_model_data.iloc[:, -1].values

# Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(encoded_df, y, test_size = 0.2, random_state = 123)

# Create metric instances outside the function
precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()
auc_metric = tf.keras.metrics.AUC()

# Define custom metrics
def f1_macro(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    precision = precision_metric(y_true, y_pred_binary)
    recall = recall_metric(y_true, y_pred_binary)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

def recall(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    return recall_metric(y_true, y_pred_binary)

def precision(y_true, y_pred):
    y_pred_binary = tf.round(y_pred)
    return precision_metric(y_true, y_pred_binary)

def roc_auc(y_true, y_pred):
    return auc_metric(y_true, y_pred)

# Residual model function
def create_residual_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = inputs

    def residual_block(x, filters, stride = 1):
        shortcut = x
        x = tf.keras.layers.Dense(filters, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(filters, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        if shortcut.shape[-1] != filters or stride != 1:
            shortcut = tf.keras.layers.Dense(filters)(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model

# Stratified K-Fold Cross Validation
n_splits = 5
skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 123)

all_accuracies = []
all_f1_scores = []
all_recalls = []
all_precisions = []
all_roc_aucs = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
    print(f"Fold {fold + 1}/{n_splits}")

    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Create Model
    input_dim = X_train_fold.shape[1]
    model = create_residual_model(input_dim)

    # Compile Model with adam
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy', f1_macro, recall, precision, roc_auc])

    # Callbacks for training efficiency
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights = True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 5, min_lr = 0.00001)

    # Train Model
    model.fit(X_train_fold, y_train_fold, 
              epochs = 50, 
              batch_size = 512, 
              validation_data = (X_val_fold, y_val_fold), 
              verbose = 2, 
              callbacks = [early_stopping, reduce_lr])

    # Evaluate Model
    metrics = model.evaluate(X_val_fold, y_val_fold, verbose = 2)

    # Store metrics
    all_accuracies.append(metrics[1])
    all_f1_scores.append(metrics[2])
    all_recalls.append(metrics[3])
    all_precisions.append(metrics[4])
    all_roc_aucs.append(metrics[5])

    # Reset metric states between folds
    precision_metric.reset_state()
    recall_metric.reset_state()
    auc_metric.reset_state()

# Print average metrics from cross-validation
print(f"Average Accuracy: {np.mean(all_accuracies)}")
print(f"Average F1 Score: {np.mean(all_f1_scores)}")
print(f"Average Recall: {np.mean(all_recalls)}")
print(f"Average Precision: {np.mean(all_precisions)}")
print(f"Average ROC AUC: {np.mean(all_roc_aucs)}")

# Evaluate on test set
test_metrics = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_metrics[1]}")
print(f"Test F1 Score: {f1_macro(y_test, model.predict(X_test))}")
print(f"Test Recall: {recall(y_test, model.predict(X_test))}")
print(f"Test Precision: {precision(y_test, model.predict(X_test))}")
print(f"Test ROC AUC: {roc_auc(y_test, model.predict(X_test))}")


# Save model as pkl file
with open('model.pkl', 'wb') as fid:
    pickle.dump(model, fid)