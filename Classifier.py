
# from matplotlib import pyplot as plt
import numpy as np
import Neural_Net_Minus as NNM

airplane_data = np.load('./Data/airplane.npy')/255
basketball_data = np.load('./Data/basketball.npy')/255
bird_data = np.load('./Data/bird.npy')/255
door_data = np.load('./Data/door.npy')/255
hexagon_data = np.load('./Data/hexagon.npy')/255

#format answers
airplane = [1, 0, 0, 0, 0]
basketball = [0, 1, 0, 0, 0]
bird = [0, 0, 1, 0, 0]
door = [0, 0, 0, 1, 0]
hexagon = [0, 0, 0, 0, 1]

# setting parameters
train_set_size = 10000
train_epochs = 10000
testing_set_size = 100
testing_amount = 1000

# prep data
training_set = []
testing_set = []
for x in range(train_set_size//5):
    # all files have more than 10000 data pts
    i = np.random.randint(0, 10000)
    training_set.append([airplane_data[i], airplane])
    training_set.append([basketball_data[i], basketball])
    training_set.append([bird_data[i], bird])
    training_set.append([door_data[i], door])
    training_set.append([hexagon_data[i], hexagon])

for x in range(testing_set_size//5):
    i = np.random.randint(0, 10000)
    testing_set.append([airplane_data[i], airplane])
    testing_set.append([basketball_data[i], basketball])
    testing_set.append([bird_data[i], bird])
    testing_set.append([door_data[i], door])
    testing_set.append([hexagon_data[i], hexagon])
    

# passing desired shape of neural network
nn = NNM.neural_network([784, 100, 100, 5], 0.03);

for i in range(train_epochs):
    # picking random data to train
    idx = np.random.randint(0,train_set_size)
    nn.train(training_set[idx][0], training_set[idx][1])

# for i in range(len(testing_set)):
correct = 0
for i in range(testing_amount):
    idx = np.random.randint(0, testing_set_size)
    outputs = nn.feed_fwd(testing_set[idx][0])
    guessIdx = np.argmax(outputs, axis = 0)[0]
    if testing_set[idx][1][guessIdx] == 1:
        correct += 1;
    
print(correct/testing_amount)
    
    
        
