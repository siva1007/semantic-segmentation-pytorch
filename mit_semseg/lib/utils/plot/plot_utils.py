import matplotlib.pyplot as plt
import json

## To Do
## A standard class for all plots

def plot(data):
    epochs = [int(d) for d in data]
    epochs = sorted(epochs)
    
    IoU = []
    accuracy = []
    
    for epoch in epochs:
        IoU.append(data[str(epoch)]['Mean IoU'] * 100)
        accuracy.append(data[str(epoch)]['Accuracy']) 

    plt.plot(epochs, IoU, 'r') # plotting t, a separately 
    plt.plot(epochs, accuracy, 'b') # plotting t, b separately 
    plt.legend(["Mean IoU", "Accuracy"], loc ="lower right") 
    plt.xlabel('epochs', fontsize=18)
    plt.show()

def plot_evaluation(files_list):
    ## TO DO
    ## Filter the files by timestamp.
    ## Pick the latest one for duplicates
    data = []
    for f in files_list:
        with open(f, 'r') as json_file:
            d = json.load(json_file)    
            data.append(d)

    plot(data)




if __name__ == '__main__':
    output_file = './ckpt/ade20k-resnet101-upernet/result/evaluation_start-5_interval-5_Mon_Jan_11_06:02:55_2021.json'
    with open(output_file, 'r') as outfile:
           data =  json.load(outfile)

    plot(data)
