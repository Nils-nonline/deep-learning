import random
from deep_learning import Perceptron, Layer, Model
import matplotlib.pyplot as plt
import statistics


a = Perceptron(num_weights=3,aktivation_fun="sigmoid",loss_func="linear")


x = []
y = []

training_data = [
    [[0,0],[0]],
    [[1,1],[1]],
    [[1,0],[0]],
    [[0,1],[0]]
]
count = 0

while count <= 10000:
    count += 1
    train_choice = random.choice(training_data)
    res = a.train_simple(train_choice[0],train_choice[1])
    
    x.append(count)
    y.append(res)
    
plt.plot(x,y)

plt.xlabel("Iteration")
plt.ylabel("Loss (Linear)")
plt.title("Error rate")

plt.show()
    
