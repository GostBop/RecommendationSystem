

import sys

from recommender import datasets
from recommender import models
import pandas as pd
import tensorflow as tf

# Dataset
ds = datasets.build_dataset("frappe")

# Model
model = models.NeuralFM(ds, num_units=256, layers=[64], apply_nfm=True)

epochs = 100
x,y,z = model.train(
    ds,
    batch_size=1024,
    epochs=epochs,
    loss_function="rmse",
    eval_metrics=["auc"],
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.01),
)
#for x_name in x:
print(x[0]["rmse"])

num = []
train_losses = []
valid_losses = []
test_losses = []
for i in range(epochs):
    num.append(i + 1)
    train_losses.append(float(x[i]["rmse"]))
    valid_losses.append(float(y[i]["rmse"]))
    test_losses.append(float(z[i]["rmse"]))

dataframe = pd.DataFrame({'epoch': num, 'train_losses': train_losses, 'valid_losses': valid_losses, 'test_losses': test_losses})
dataframe.to_csv("nfm_1_loss_hidden_64_1024.csv",index=False,sep=',')
