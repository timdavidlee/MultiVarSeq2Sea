from model import MultiVarSeq2Seq, load_saved_seq_model
import numpy as np

def main():
    batch_sz = 10000
    window_sz = 100
    nfeat = 6

    simulated_data = np.random.rand(batch_sz, window_sz, nfeat)
    X = simulated_data[:,:80,:]
    Y = simulated_data[:,80:,:]

    print(X.shape, Y.shape)
    X_trn = X[:9000, :, :]
    X_val = X[9000:, :, :]

    Y_trn = Y[:9000, :, :]
    Y_val = Y[9000:, :, :]

    m = MultiVarSeq2Seq(nfeat=6, leadtime_sz=80, forecast_sz=20, enc_lstm_units=8, dec_lstm_units=8)
    m.build()
    m.fit(X_trn, Y_trn, X_val, Y_val)

    m.save()

    saved_m = load_saved_seq_model()

    a = m.enc_model.layers[1].get_weights()
    b = saved_m.enc_model.layers[1].get_weights()

    for j, k in zip(a,b):
        print(np.array_equal(j,k))


    a = m.dec_model.layers[1].get_weights()
    b = saved_m.dec_model.layers[1].get_weights()

    for j, k in zip(a,b):
        print(np.array_equal(j,k))


    p1 = m.predict(X_trn[:2])
    p2 = saved_m.predict(X_trn[:2])

    assert p1.shape == p2.shape

    print(np.array_equal(p1, p2))

if __name__ == "__main__":
    main()