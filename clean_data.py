def getmnistclean():
    import numpy as np

    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=np.reshape(x_train,(x_train.shape[0],-1))
    print(x_train.shape)
    
    x_np=[]
    y_np=[]
    k=0
    for i in range(len(y_train)):
        x_np.append(x_train[i])
        y_np.append(y_train[i])
    x_np=np.array(x_np)
    y_np=np.array(y_np)
    x_nptrain=x_np[:10000]
    y_nptrain=y_np[:10000]
    print(x_nptrain.shape,y_nptrain.shape)
    x_nptest=x_np[10000:12000]
    y_nptest=y_np[10000:12000]
    print(x_nptest.shape,y_nptest.shape)

    return x_nptrain, y_nptrain, x_nptest, y_nptest

def getmnistpoisoned( level=0.1):

    import numpy as np
    import random
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=np.reshape(x_train,(x_train.shape[0],-1))
    print(x_train.shape)
    
    

    x_tp=[]
    y_tp=[]


     #flipping attack script
    
    index_p=random.randint(0,9)
    change=random.randint(0,9)
    while (change==index_p):
        change=random.randint(0,9)
        print(index_p,change)
    for i in range (len(x_train)):
        x_tp.append(x_train[i])
        if (y_train[i]==index_p):
            y_tp.append(change)

        else:
            y_tp.append(y_train[i])
    
  
  
    

    print("working")
    # np.asarray(x).astype('float32')
    x_tp=np.array(x_tp)
    y_tp=np.array(y_tp)
    print(x_tp.shape,y_tp.shape)

    x_tptrain=x_tp[:10000]
    y_tptrain=y_tp[:10000]
    print(x_tptrain.shape,y_tptrain.shape)
    x_tptest=x_tp[10000:12000]
    y_tptest=y_tp[10000:12000]
    print(x_tptest.shape,y_tptest.shape)

    return x_tptrain, y_tptrain, x_tptest, y_tptest


