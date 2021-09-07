import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras
from tensorflow.keras import regularizers
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#loading the dataset ##should be in the form of X_train, y_train, X_valid,y_valid
import clean_data
from client import Client
 
# weights=np.random.rand(784,512)
# bias=np.random.rand(512)
# weights2=np.random.rand(512,512)
# bias2=np.random.rand(512)
# weights3=np.random.rand(512,512)
# bias3=np.random.rand(512)
# weights4=np.random.rand(512,128)
# bias4=np.random.rand(128)
# weights5=np.random.rand(128,10)
# bias5=np.random.rand(10)

intializer = keras.initializers.GlorotUniform(seed=42)

def mnist_model():
    

    model=keras.models.Sequential([
        keras.layers.Flatten(input_shape=[784,]),
        keras.layers.Dense(256,activation='tanh'),
        keras.layers.Dense(128,activation='tanh'),
        keras.layers.Dense(10,activation='softmax')
        ])
    
    return model

def model_average(client_weights):
    average_weight_list=[]
    for index1 in range(len(client_weights[0])):
        layer_weights=[]
        for index2 in range(len(client_weights)):
            weights=client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight=np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list
            


def create_model():
    model = mnist_model()
   
    
    weight = model.get_weights()

    return weight

    
#initializing the client automatically

def train_server(training_rounds,epoch,batch,learning_rate,level):

    x_nptrain, y_nptrain, x_nptest, y_nptest = clean_data.getmnistclean()
    x_tptrain, y_tptrain, x_tptest, y_tptest = clean_data.getmnistpoisoned(level= level)

    accuracy_list=[]
    accuracy_list1=[]

    client_weight_for_sending=[]
    client_weight_for_sending1=[]

   # success_rates=[]

    for index1 in range(1,training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged=[]
        client_weights_tobe_averaged1=[]

        for index in range(10):
            print('-------Client-------', index)
            if index1==1:
                if index==4 or index==1 or index==2:
                    print('Sharing Initial Global Model with Common Weight Initialization')
                    initial_weight=create_model()
                    client=Client(x_nptrain,y_nptrain,epoch,learning_rate,initial_weight,batch)
                    weight=client.train()
                    client_weights_tobe_averaged.append(weight)

                    initial_weight1=create_model()
                    client1=Client(x_tptrain,y_tptrain,epoch,learning_rate,initial_weight1,batch)
                    weight1=client1.train()
                    client_weights_tobe_averaged1.append(weight1)
                    
                else:
                    print('Sharing Initial Global Model with Common Weight Initialization')
                    initial_weight=create_model()
                    client=Client(x_nptrain,y_nptrain,epoch,learning_rate,initial_weight,batch)
                    weight=client.train()
                    client_weights_tobe_averaged.append(weight)
                    client_weights_tobe_averaged1.append(weight)
            else:
                if index==4:
                    client1=Client(x_tptrain,y_tptrain,epoch,learning_rate,client_weight_for_sending1[index1-2],batch)
                    weight1=client1.train()
                    client_weights_tobe_averaged1.append(weight1)

                    client=Client(x_nptrain,y_nptrain,epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                    weight=client.train()
                    client_weights_tobe_averaged.append(weight)
                else:
                    client=Client(x_nptrain,y_nptrain,epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                    weight=client.train()
                    client_weights_tobe_averaged.append(weight)

                    client1=Client(x_nptrain,y_nptrain,epoch,learning_rate,client_weight_for_sending1[index1-2],batch)
                    weight1=client1.train()
                    client_weights_tobe_averaged1.append(weight1)
    
        #calculating the avearge weight from all the clients (benign scenario)
        client_average_weight=model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)

        #calculating the avearge weight from all the clients (adversarial scenario)
        client_average_weight1=model_average(client_weights_tobe_averaged1)
        client_weight_for_sending1.append(client_average_weight1)

        #validating the model with avearge weight (benign scenario)
        model=mnist_model()
        model.set_weights(client_average_weight)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=learning_rate),metrics=['accuracy'])
        result=model.evaluate(x_nptest, y_nptest, batch_size = batch)
        accuracy=result[1]
        print('#######-----Acccuracy without poison for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list.append(accuracy)
        
        #validating the model with avearge weight (adversarial scenario)
        model1=mnist_model()
        model1.set_weights(client_average_weight1)
        model1.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=learning_rate),metrics=['accuracy'])
        result1=model1.evaluate(x_nptest, y_nptest, batch_size = batch)
        accuracy1=result1[1]
        print('#######-----Acccuracy with poison for round ', index1, 'is ', accuracy1, ' ------########')
        accuracy_list1.append(accuracy1)


    return accuracy_list, accuracy_list1



print("==============Federated learning with complete poisoning==============")
training_accuracy_list100, training_accuracy_list_adv100 = train_server(25,1,64,0.01,0.9)
print("Train accuracy without adversary:", training_accuracy_list100)
print("Train accuracy with adversary:", training_accuracy_list_adv100)
#print("Success rate: ", sc_rate100)
print("Result accuracy without adversary:", training_accuracy_list100[-1])
print("Result accuracy with adversary:", training_accuracy_list_adv100[-1])


with open('tp_100_benign_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list100)
f.close()

with open('tp_100_mal_acc.npy', 'wb') as f:
    np.save(f, training_accuracy_list_adv100)
f.close()





