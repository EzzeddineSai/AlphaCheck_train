import gameclasses
from gameclasses import state_compression
import numpy as np
import copy
import sys
import os
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from multiprocessing import Process, Lock, Value
from config import *


def predict(reconstructed_model,state):
    compressed_state = state_compression(state)
    prediction = reconstructed_model.predict(np.expand_dims(compressed_state,axis=0),verbose = 0)[0]
    return prediction[:256], prediction[256]


class node:
    def __init__(self,reconstructed_model,state,c_puct, rec_depth): #initialized means never been visited
        self.rec_depth = rec_depth
        self.reconstructed_model = reconstructed_model
        self.c_puct = c_puct
        self.state = copy.deepcopy(state) #evaluate
        self.p, self.v = predict(reconstructed_model,state)
        
        self.legal_moves = gameclasses.find_legal_moves(state['board'], state['player turn'], state['sequence piece'])
        if len(self.legal_moves) > 0:
            self.legal_moves_indices = []
            for i in range(len(self.legal_moves)):
                move = self.legal_moves[i]

                self.legal_moves_indices.append(gameclasses.move_to_index[move]) #indices in the total move space

            for i in range(len(self.p)): #make illegal moves have 0 probability
                if i not in self.legal_moves_indices:
                    self.p[i] = 0
            self.p = self.p/np.sum(self.p)


            #expand

            self.children = [None for i in range(len(self.legal_moves))] #don't make children until needed
            self.N = np.zeros(len(self.legal_moves)) #for the child with the same index
            self.Q = np.zeros(len(self.legal_moves))
            self.W = np.zeros(len(self.legal_moves))
    
    def select(self):
        global depth
        UB_max = -float("inf")
        UB_max_index = -1
        
        if len(self.legal_moves) == 0:
            return self.v
        
        if self.rec_depth >= REC_DEPTH:
            return self.v

        for i in range(len(self.legal_moves)):
            prob = self.p[self.legal_moves_indices[i]]
            q = self.Q[i]
            U = self.c_puct*prob*np.sqrt(1+sum(self.N))/(1+self.N[i])
            if U+q > UB_max:
                UB_max = U+q
                UB_max_index = i
        if self.N[UB_max_index] == 0: #never visited
            aux_game = gameclasses.GameState(board=self.state['board'],player_turn=self.state['player turn'],sequence_piece=self.state['sequence piece'])
            
            aux_game.transition(self.legal_moves[UB_max_index])
            new_state = aux_game.return_game_state()
            self.children[UB_max_index] = node(self.reconstructed_model,new_state,self.c_puct, self.rec_depth+1)
            
            v = self.children[UB_max_index].v
            #propagate
        else:
            
            v = self.children[UB_max_index].select()
        
        self.W[UB_max_index] += v
        self.N[UB_max_index] += 1
        self.Q[UB_max_index] = self.W[UB_max_index]/self.N[UB_max_index]
        
        return v

def play(reconstructed_model,search_iter,c_puct,tau):
    moves_played_x = []
    moves_played_y = []
    main_game = gameclasses.GameState()
    game = gameclasses.GameState()
    default_state = game.return_game_state()
    root = node(reconstructed_model,default_state, c_puct, 0)
    winner_color = -2
    for i in range(MAX_GAME_LEN): #max game depth
        for j in range(search_iter):
            root.select()
            
        if len(root.legal_moves) == 0:
            winner_color = -1*root.state['player turn']
            break
        if np.sum(root.N**(1/tau)) == 0:
            winner_color = 0
            break

        pi = (root.N**(1/tau))/(np.sum(root.N**(1/tau)))
        
        
        expanded_pi = [0 for i in range(256)]
        for j in range(len(root.legal_moves)):
            expanded_pi[root.legal_moves_indices[j]] = pi[j]
        moves_played_x.append(state_compression(root.state))
        moves_played_y.append(expanded_pi)
        
        move_chosen_index = np.random.choice([k for k in range(len(root.legal_moves))],p=pi)
        
        move_chosen = gameclasses.general_moves[root.legal_moves_indices[move_chosen_index]]
        main_game.transition(move_chosen)

        
        root = root.children[move_chosen_index]
        
    for i in range(len(moves_played_x)):
        move_color = moves_played_x[i][-1] #player turn
        if winner_color == 0:
            moves_played_y[i].append(DRAW_LOSS)
        elif winner_color == move_color:
            moves_played_y[i].append(1)
        else:
            moves_played_y[i].append(-1)
    return moves_played_x, moves_played_y
    

def generate_data(game_no, lock, model_iteration, no_games, search_iter, c_puct, tau):
    from alpha_zero_loss import alpha_zero_loss
    
    #import the model separatly for each process (this is neccessary for this set up)
    reconstructed_model = keras.models.load_model('models\iter'+model_iteration+'.h5',custom_objects={ 'alpha_zero_loss': alpha_zero_loss })

    t = time.time()

    for i in range(no_games):
        
        moves_x, moves_y = play(reconstructed_model,search_iter,c_puct,tau) 
        moves_x = np.array(moves_x)
        np.random.shuffle(moves_x)
        moves_y = np.array(moves_y)
        np.random.shuffle(moves_y)
        with lock:
            #only 1 process can write at a time
            print("GAME", game_no.value, "complete", "time", time.time()-t)
            t = time.time()
            np.save('generated_data\\iteration'+model_iteration+'\\'+str(game_no.value)+'_x', moves_x)
            np.save('generated_data\\iteration'+model_iteration+'\\'+str(game_no.value)+'_y', moves_y)
            game_no.value += 1

def create_model():
    from alpha_zero_loss import alpha_zero_loss
    input_layer = keras.Input(shape=(33,))
    hidden_layer_1 = layers.Dense(512, activation='tanh')(input_layer)
    hidden_layer_2 = layers.Dense(512, activation='tanh')(hidden_layer_1)
    hidden_layer_3 = layers.Dense(512, activation='tanh')(hidden_layer_2)
    hidden_layer_4 = layers.Dense(1024, activation='tanh')(hidden_layer_3)
    output_1 = layers.Lambda(lambda x: x/tf.reduce_sum(x))(layers.Dense(256, activation='sigmoid')(hidden_layer_4))
    output_2 = layers.Dense(1, activation='tanh')(hidden_layer_4)
    combined_output = layers.concatenate([output_1,output_2])
    
    model = keras.Model(input_layer,combined_output)
    model.compile(optimizer='adam', loss=alpha_zero_loss)

    model.save('models\\iter1.h5')

if __name__ == "__main__":

    

    search_iter = SEARCH_ITER
    c_puct = C_PUCT
    tau = TAU
    no_processes = NO_PROCESSES
    no_games = NO_GAMES #games per process

    model_iteration = sys.argv[1]
    if model_iteration == '0':
        create_model()
    else:

        os.makedirs('generated_data\\iteration'+model_iteration)
        lock = Lock()
        game_no = Value('i',0) #integer with val 0
        processes = []
        for i in range(no_processes):
            processes.append(Process(target=generate_data,args=(game_no, lock, model_iteration, no_games, search_iter, c_puct, tau)))
        for i in range(no_processes):
            processes[i].start()
        for i in range(no_processes):
            processes[i].join()