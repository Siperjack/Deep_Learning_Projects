import networkclass
import functions
import numpy as np
import imageGenerator
import matplotlib.pyplot as plt

n = 20 #pixels in picture generated
m = 1000 #total sample size5
epochs = 10
C = 50 #remeber to set over m/10


picture_list = imageGenerator.GenerateTestSet(n, m, center = False, noicerate = 0.1)

training_list = picture_list[0:(len(picture_list)*8)//10]
validation_list = picture_list[(len(picture_list)*8)//10:(len(picture_list)*9)//10]
testing_list = picture_list[(len(picture_list)*9)//10:]
NNs = []
NNs.append(networkclass.network(n**2, "specification1.txt"))
NNs.append(networkclass.network(n**2, "specification1.txt"))
NNs.append(networkclass.network(n**2, "specification1.txt"))
NNs.append(networkclass.network(n**2, "specification1.txt"))
NNs.append(networkclass.network(n**2, "specification2.txt"))
NNs.append(networkclass.network(n**2, "specification2.txt"))
NNs.append(networkclass.network(n**2, "specification2.txt"))
NNs.append(networkclass.network(n**2, "specification2.txt"))
#NNs.append(networkclass.network(n**2, "specification3.txt"))
for NN in NNs:
    NN.train_network_with_SGD(training_list, C, validation_list, epochs)


fig, ax = plt.subplots(4,4, figsize = (7,7))
for i in range(4):
    for j in range(4):
        ax[i,j].imshow(picture_list[i + 4*j].image)

for count, NN in enumerate(NNs):
    fig, ax = plt.subplots(2, figsize = (14,14))
    ax[0].title.set_text(f"spec: {count + 1} Training/Validation loss")
    ax[0].plot(np.arange(len(NN.training_performance)),NN.training_performance, color = 'tab:blue', label = f"Training")
    ax[0].plot(np.arange(len(NN.validation_performance)),NN.validation_performance, color = 'tab:orange', label = f"Validation")
    ax[0].legend()
    ax[1].title.set_text(f"spec: {count + 1} Training/Validation hitrate")
    ax[1].plot(np.arange(len(NN.training_hitrate)),NN.training_hitrate, color = 'tab:blue', label = "Training")
    ax[1].plot(np.arange(len(NN.validation_hitrate)),NN.validation_hitrate, color = 'tab:orange', label = f"Validation")
    ax[1].legend()


numb_to_figure = ["Circle", "Cross", "Square", "lines"]
for count, NN in enumerate(NNs):
    loss_sum, predict, rightwrong_list = NN.test(testing_list)
    print(f"Hitrate of spec {count + 1} on the testset is: {predict}")
    for i in range(2):
        if i >= len(rightwrong_list[1]):
            print(f"Exists less than {1} missed predictions in testset!")
            break
        plt.figure(100 + count)
        pred_figure = numb_to_figure[rightwrong_list[1][i][1]]
        actual_figure = numb_to_figure[int(np.argmax(rightwrong_list[1][i][0].sol))]
        plt.title(f"Prediction: {pred_figure}, Actually: {actual_figure}")
        plt.imshow(rightwrong_list[1][i][0].image)


