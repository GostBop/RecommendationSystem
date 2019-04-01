import pandas as pd
import numpy as np
import random
import tensorflow as tf
def data(sample_size):
    uname = ['userId','gender','age','occupation','zip']
    users = pd.read_csv('./ml-1m/users.dat', sep='::', header = None, names=uname, engine='python')
    user1 = users.ix[:,[0, 1, 2]]

    rnames = ['userId','movieId','rating','timestamp']
    ratings_df = pd.read_csv('./ml-1m/ratings.dat', sep='::', names=rnames, engine='python')

    mnames = ['movieId','title','genres']
    movies_df = pd.read_csv('./ml-1m/movies.dat', sep='::', names=mnames, engine='python')
    movies_df['movieRow'] = movies_df.index
    movies_df = movies_df[['movieRow','movieId','title', 'genres']]
    #筛选三列出来
    movies_df.to_csv('./ml-1m/moviesProcessed.csv', index=False, header=True, encoding='utf-8')
    #生成一个新的文件moviesProcessed.csv

    ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
    ratings_df = ratings_df[['userId','movieRow','rating']]
    ratings_df = pd.merge(ratings_df, user1)
    ratings_df.to_csv('./ml-1m/ratingsProcessed.csv', index=False, header=True, encoding='utf-8')
    #导出一个新的文件ratingsProcessed.csv

    userNo = ratings_df['userId'].max() + 1

    movieNo = ratings_df['movieRow'].max() + 1

    ratings_df_length = np.shape(ratings_df)[0]

    x_train = np.zeros((sample_size, movieNo + userNo + 2 + 7))
    x_test = np.zeros((sample_size, movieNo + userNo + 2 + 7))
    y_train = np.zeros((sample_size, 1))
    y_test = np.zeros((sample_size, 1))
    for i in range(sample_size):
        n = random.randint(int(800000 / sample_size), int(1000000 / sample_size))

        x_train[i, ratings_df.ix[(i + 1) * n, 1]] = 1
        x_train[i, movieNo + ratings_df.ix[(i + 1) * n, 0]] = 1
        x_train[i, movieNo + userNo + 2 + int(ratings_df.ix[(i + 1) * n, 4] / 9.1)] = 1
        if ratings_df.ix[(i + 1) * n, 3] == 'F':
            x_train[i, movieNo + userNo] = 1
        ##for ii in range(len(bin(ratings_df.ix[(i + 1) * n + 1, 1])) - 2):
        
        x_test[i, ratings_df.ix[(i + 1) * n + 1, 1]] = 1  
        x_test[i, movieNo + ratings_df.ix[(i + 1) * n + 1, 0]] = 1
        x_test[i, movieNo + userNo + 2 + int(ratings_df.ix[(i + 1) * n + 1, 4] / 9.1)] = 1
        if ratings_df.ix[(i + 1) * n + 1, 3] == 'F':
            x_test[i, movieNo + userNo] = 1

        
        y_train[i] = ratings_df.ix[(i + 1) * n, 2]
        y_test[i] = ratings_df.ix[(i + 1) * n + 1, 2]

    return x_train, y_train, x_test, y_test

