import random
from deep_learning import Perceptron, Layer, Model
import matplotlib.pyplot as plt
import statistics


model = Model(input_size = 2, random_seed=3)

input_layer = Layer(layer_type="hidden")
model.add_layer(input_layer)

output_layer = Layer(layer_type="output",num_neurons=2)
model.add_layer(output_layer)

print(model)

training_data = [
    [[0,0],[0,0]], #XOR-training-data
    [[1,1],[0,0]],
    [[1,0],[1,0]],
    [[0,1],[0,1]]
]

batch_loss = []


count = 0

x = []
y = []


for i in range(0,10000):
    count += 1
    
    
    train_choice = random.choice(training_data)
    res = model.train(*train_choice)
    
    
    batch_loss.append(res)
    
    if count % 1000 == 0:
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

plt.plot(x,y)

plt.xlabel("Iteration")
plt.ylabel("Loss (MSE)")
plt.title("Error rate")

plt.show()
