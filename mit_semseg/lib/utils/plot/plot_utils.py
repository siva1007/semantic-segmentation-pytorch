import matplotlib.pyplot as plt
import json
from os import listdir
from os.path import isfile, join

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

def train_val_plot(train_data, val_data, parameter, prefix_path, multiplier = 1):
    epochs = [int(d) for d in train_data]
    for key in val_data:
        if int(key) not in epochs:
            epochs.append(int(key))

    epochs = sorted(epochs)
    train = []
    val = []

    for epoch in epochs:
        t = train_data.get(str(epoch), {parameter : None})[parameter]
        train.append(t * multiplier if t is not None else None)
        v = val_data.get(str(epoch), {parameter : None})[parameter]
        val.append(v * multiplier if v is not None else None)



    plt.plot(epochs, train, 'r') # plotting t, a separately 
    plt.plot(epochs, val, 'b') # plotting t, b separately 
    plt.legend(["Train " + parameter, "Val " + parameter], loc ="lower right") 
    plt.xlabel('epochs', fontsize=18)
    # plt.show()
    filename = 'train_val_' + parameter + '_plot.png' 
    plt.savefig(join(prefix_path, filename), dpi=100)
    plt.close()




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


def extract_dict_json_data(all_files, filter_by_keyword, path_prefix):
    files = [f for f in all_files if filter_by_keyword in f]
    data = {}
    for output_file in files:
        output_file = join(path_prefix, output_file)
        with open(output_file, 'r') as outfile:
            d =  json.load(outfile)
            for k,v in d.items():
                data[k] = v

    return data



if __name__ == '__main__':

    RESULTS='./ckpt1/ade20k-resnet101-upernet/result'
    all_files = [f for f in listdir(RESULTS) if isfile(join(RESULTS, f))]
    train_files = [f for f in all_files if 'train_evaluation' in f]
    val_files = [f for f in all_files if 'val_evaluation' in f]

    train_data = extract_dict_json_data(all_files, 'train_evaluation', RESULTS)
    val_data = extract_dict_json_data(all_files, 'val_evaluation', RESULTS)

    # output_file = './ckpt/ade20k-resnet101-upernet/result/evaluation_start-5_interval-5_Mon_Jan_11_06:02:55_2021.json'
    # with open(output_file, 'r') as outfile:
    #        data =  json.load(outfile)

    train_val_plot(train_data, val_data, "Mean IoU", RESULTS, multiplier=100)
    train_val_plot(train_data, val_data, "Accuracy", RESULTS)
