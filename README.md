# MultiVarSeq2Seq

Multivariate sequence to sequence forecasting. This model is designed to take in multivariate continuous time series and simultaneously forecast the same continous variables. 

For example: 

- 1 year of Google Stock Data
- 1 year of FB stock data
- 1 year of Snap stock data 

3 features, we can take in 3 months of data, and predict the next 5 days of stocks.

## Why this code?

Most tutorials out there don't have the proper multi-variate inference code. This is complicated since its usually multi-step in and multi-step out. This has already been implemented in this code.


## How to use:

```python

from model import MultiVarSeq2Seq, load_saved_seq_model

# initialize the model
m = MultiVarSeq2Seq(nfeat=6, 
                    leadtime_sz=80,
                    forecast_sz=20,
                    enc_lstm_units=8,
                    dec_lstm_units=8)
                    
# build the model
m.build()

# fit to traing data, will also score against validation data
m.fit(X_trn, Y_trn, X_val, Y_val)

# make a prediction
p1 = m.predict(X_trn[:2])
```
