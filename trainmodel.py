import tensorflow as tf
from tensorflow import keras
import sys
import os
import numpy as np
import math
from config import *
from alpha_zero_loss import alpha_zero_loss

model_iteration = sys.argv[1]

files_directory = os.path.join(os.getcwd(),'generated_data\\iteration'+model_iteration)
model_directory = os.path.join(os.getcwd(),'models\\iter'+model_iteration+'.h5')

reconstructed_model = keras.models.load_model(model_directory,custom_objects={ 'alpha_zero_loss': alpha_zero_loss })
print(reconstructed_model.summary())

files_directory = os.path.join(os.getcwd(),'generated_data\\iteration'+model_iteration)
moves_x_list = []
moves_y_list = []
for filename in os.listdir(files_directory):
    if filename[-5] == 'x': #since the files are sorted alpha numerically
        moves_x_list.append(np.load('generated_data\\iteration'+model_iteration+'\\'+filename))
    elif filename[-5] == 'y':
        moves_y_list.append(np.load('generated_data\\iteration'+model_iteration+'\\'+filename))

moves_x = np.concatenate(moves_x_list)
moves_y = np.concatenate(moves_y_list)

perm = np.random.permutation(len(moves_x))
moves_x = moves_x[perm]
moves_y = moves_y[perm]

moves_x_train = moves_x[:math.ceil(0.9*len(moves_x))] # 90:10 split
moves_y_train = moves_y[:math.ceil(0.9*len(moves_y))]

moves_x_test = moves_x[math.ceil(0.9*len(moves_x)):] 
moves_y_test = moves_y[math.ceil(0.9*len(moves_y)):]



reconstructed_model.fit(moves_x_train,  moves_y_train, epochs=EPOCHS)


print(reconstructed_model.evaluate(x=moves_x_test, y=moves_y_test))


reconstructed_model.save('models\\iter'+str(int(model_iteration)+1)+'.h5')