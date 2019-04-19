import time
import json
import keras
import numpy as np
import pandas as pd
import pathlib
from keras import optimizers
from keras.layers import Input, CuDNNLSTM, Dense
from keras.models import Sequential, Model, load_model

class MultiVarSeq2Seq(object):
    def __init__(self, 
                 nfeat,
                 leadtime_sz,
                 forecast_sz,
                 enc_lstm_units,
                 dec_lstm_units,
                 lr=0.001,
                 loss='logcosh'):
        
        self.leadtime_sz = leadtime_sz
        self.forecast_sz = forecast_sz
        self.nfeat = nfeat
        self.lr = lr
        self.loss = loss
        
        # model params
        
        self.enc_lstm_units = enc_lstm_units
        self.dec_lstm_units = dec_lstm_units
        
        self.layers = {}
        self.model = None
        self.enc_model = None
        self.dec_model = None
        
    def build(self):
        
        # ========================================================================
        # make the training model
        # ========================================================================
        
        # define our input shape (to take any sized input)
        enc_input_shape = (None, self.nfeat)
        
        # initialize the basic input layers
        self.layers['enc_input'] = Input(shape=enc_input_shape)
        self.layers['enc_recurrent'] = CuDNNLSTM(self.enc_lstm_units,
                                                 kernel_initializer="he_normal",
                                                 input_shape=enc_input_shape,
                                                 return_state=True)
        
        # the input is passed through the recurrent layer
        # and the output and hiddent states are returned
        _, enc_h, enc_c = self.layers['enc_recurrent'](self.layers['enc_input'])
        
        # bundle the hidden states together
        enc_states = [enc_h, enc_c]
        
        # setup the decoder
        dec_input_shape = (None, self.nfeat)
        
        # initialize our recurrent output layers
        self.layers['dec_input'] = Input(shape=dec_input_shape)
        self.layers['dec_recurrent'] = CuDNNLSTM(self.dec_lstm_units,
                                                 kernel_initializer="he_normal",
                                                 return_sequences=True,
                                                 return_state=True)
        
        # the decoder input is passed through the 
        # decoder layers, and returns an output
        X_out, _, _ = self.layers['dec_recurrent'](self.layers['dec_input'],
                                                   initial_state=enc_states
                                                  )
        
        # need one last dense layer to ensure that
        # there are enough features to output
        self.layers['dec_dense'] = Dense(self.nfeat)
        
        # define the output
        dec_outputs = self.layers['dec_dense'](X_out)
        
        # define our model
        self.model = Model(
            [self.layers['enc_input'],
             self.layers['dec_input']],
            dec_outputs
        )
        
        self.model.compile(optimizer=optimizers.Adam(lr=self.lr),
                           loss=self.loss)
        self.model.summary()
        
        # ========================================================================
        # make the inference model
        # ========================================================================
        
        # the encoder is definied as taking in input
        # and outputing encoder states
        self.enc_model = Model(self.layers['enc_input'],
                               enc_states)
        
        # setting up the decoder model
        # creating placeholder inputs for the hidden states passed
        # by the encoder
        self.layers['dec_state_h'] = Input(shape=(self.enc_lstm_units,))
        self.layers['dec_state_c'] = Input(shape=(self.enc_lstm_units,))
        
        # bundle the input states together
        dec_state_inputs = [
            self.layers['dec_state_h'],
            self.layers['dec_state_c']
        ]
        
        # the decoder outputs are defined as decoder inputs passed through
        # the decoder recurrent layers
        dec_outputs, dec_h, dec_c = self.layers['dec_recurrent'](self.layers['dec_input'], 
                                                                 initial_state=dec_state_inputs)

        dec_output_states = [dec_h, dec_c]
        dec_outputs = self.layers['dec_dense'](dec_outputs)
        
        self.dec_model = Model(
            [self.layers['dec_input']] + dec_state_inputs, 
            [dec_outputs] + dec_output_states
        )
        
        self.enc_model.summary()
        self.dec_model.summary()
        
    def fit(self, X_trn, Y_trn, X_val, Y_val, batch_sz=128, epochs=6, verbose=1):
        start_time = time.time()
        
        # fit the model
        hist = self.model.fit([X_trn, Y_trn[:, :-1, :]],
                              Y_trn[:,1:,:],
                              batch_size=batch_sz,
                              epochs=epochs,
                              verbose=verbose,
                              validation_data=(
                                  [X_val, Y_val[:, :-1, :]],
                                  Y_val[:, 1:, :]
                              )
                             )
        print("training total time:{0}".format(time.time() - start_time))
        
    def predict(self, input_seq):
        
        batch_sz, win_sz, nfeat = input_seq.shape
        
        # encode the input as state vectors
        state_vals = self.enc_model.predict(input_seq)
        
        # generate empty target sequence
        target_seq = np.zeros((batch_sz, 1, self.nfeat))
        
        # populate the first position of the target seq with
        # the start number, will use the last item of the input
        # sequence
        target_seq[:, 0, :] = input_seq[:, -1, :]
        
        # initialize so that we will do (forecast_index, batch_sz, nfeat)
        # will populate with loop below
        decoded_timeseries = np.zeros((batch_sz, self.forecast_sz, nfeat))
        
        # now we will loop, there will be 1 loop
        # for every step of forecasting required

        for j in range(self.forecast_sz):
            # take in an input + states value
            # and predict the next step
            output, h, c = self.dec_model.predict([target_seq] + state_vals)
            
            # we store the t+1 in the time series
            decoded_timeseries[:, j, :] = output[:, -1, :]
            
            # the t+1 becomes the new input
            target_seq[:, 0, :] = output[:, -1, :]
            
            # states are updated
            state_vals = [h, c]
            
            # looped again
            
        return decoded_timeseries
    
    
    def save(self, model_file="./model.h5", config_file='./model_config.json'):
        if (self.enc_model is None) | (self.dec_model is None):
            print("model has not been trained, please train")
            return None
        
        model_config = {
            "nfeat": self.nfeat,
            "forecast_sz": self.forecast_sz,
            "leadtime_sz": self.leadtime_sz,
            "enc_lstm_units": self.enc_lstm_units,
            "dec_lstm_units": self.dec_lstm_units
        }
        with open(config_file, "w") as f:
            json.dump(model_config, f)
        
        self.model.save(model_file)
        print("saved {0}".format(model_file))
        
        
    def _from_saved_model(self, saved_model):
        # =====================================================
        # recreating the encoder
        # =====================================================
        enc_input = saved_model.input[0] # input 1
        _, enc_h, enc_c = saved_model.layers[2].output # lstm 1
        enc_states = [enc_h, enc_c]
        self.enc_model = Model(enc_input, enc_states)

        # =====================================================
        # recreating the decoder
        # =====================================================
        dec_inputs = saved_model.input[1] # input 2
        dec_state_h = Input(shape=(self.enc_lstm_units,))
        dec_state_c = Input(shape=(self.enc_lstm_units,))
        dec_state_inputs = [dec_state_h, dec_state_c]
        
        dec_recurrent = saved_model.layers[3] # lstm 2
        dec_outputs, dec_h, dec_c = dec_recurrent(dec_inputs,
                                                  initial_state=dec_state_inputs)
        
        dec_output_states = [dec_h, dec_c]
        dec_dense = saved_model.layers[4]
        dec_outputs = dec_dense(dec_outputs)
        self.dec_model = Model(
            [dec_inputs] + dec_state_inputs,
            [dec_outputs] + dec_output_states
        )
        
        
def load_saved_seq_model(filename='./model.h5',
                         config='./model_config.json'):
    model = load_model(filename)
    model.summary()
    with open(config) as f:
        config = json.load(f)
    
    m = MultiVarSeq2Seq(nfeat=config['nfeat'], 
                        leadtime_sz=config['leadtime_sz'],
                        forecast_sz=config['forecast_sz'],
                        enc_lstm_units=config['enc_lstm_units'],
                        dec_lstm_units=config['dec_lstm_units'],
                       )
    m.model = model
    m._from_saved_model(model)
    return m