import random
from deep_learning import Perceptron, Layer, Model
import matplotlib.pyplot as plt
import statistics
import torch
import cProfile

"""
Copyright (c) 2025 Nils Häußler. All Rights Reserved.

This work may not be copied, reproduced, distributed, displayed, performed, modified, adapted, published, transmitted, or used to create derivative works in any form or by any means without the express prior written permission of the copyright owner. No rights are granted to any user except for the right to view the work for personal reference or rights explicitly granted by law. All other rights are strictly reserved to the author.
"""
model = Model(input_size = 3)

input_layer = Layer(layer_type="hidden",num_neurons=4)
model.add_layer(input_layer)

hidden_layer1 = Layer(layer_type="hidden",num_neurons=3)
model.add_layer(hidden_layer1)

output_layer = Layer(layer_type="output",num_neurons=3)
model.add_layer(output_layer)

print(model)

training_data = [
    [[0,0,0],[0,0,0]], #one-hot-encoding of number of 1s
    
    [[1,0,0],[1,0,0]],
    [[0,1,0],[1,0,0]],
    [[0,0,1],[1,0,0]],
    
    [[1,1,0],[0,1,0]],
    [[1,0,1],[0,1,0]],
    [[0,1,1],[0,1,0]],
    
    [[1,1,1],[0,0,1]]
]

batch_loss = []


count = 0

x = []
y = []


def train_run():
    global count
    global x
    global y
    global batch_loss
    
    for i in range(0,100000):#100000
        count += 1
        

        train_choice = random.choice(training_data)
        
        res = model.train(*train_choice)
            
        batch_loss.append(res)
        
        if count % 10000 == 0:
            for choice in training_data:
                result = model.feed_forward(choice[0])
                
                text_res = "---------------\n"
                text_res += "Iteration: {}\n".format(count)
                text_res += "Input: {}\n".format(choice[0])
                text_res += "Result: {}\n".format(result)
                text_res += "Expected result: {}\n".format(choice[1])
                text_res += "---------------\n"
                
                print(text_res)
        
        if len(batch_loss)>100:
            del batch_loss[0]
            
        if count > 100:
            x.append(count)
            y.append(statistics.mean(batch_loss))


res = cProfile.run('train_run()')

plt.plot(x,y)

plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Error rate")

plt.show()