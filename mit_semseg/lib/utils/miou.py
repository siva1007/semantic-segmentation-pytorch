import pandas as pd
import json

def write_to_csv(run_name, filename, epochs, output_file):

    with open(filename, 'r') as outfile:
           data =  json.load(outfile)

    #class_names = list(data[list(data.keys())[0]]['Class Result'].keys())
    # to ensure order is maintained
    class_names = ["class {}".format(i) for i in range(150)]

    
    columns = ['run_name', 'epoch']
    columns.extend(class_names)
    columns.append('Mean IoU')
    rows = []

    for epoch in epochs:
        row = []
        row.append(run_name)
        epoch = str(epoch)
        row.append(epoch)
        epoch_data = data[epoch]
        
        #To ensure order is maintained
        for i in range(150):
            class_name = "class {}".format(i)
            row.append(epoch_data['Class Result'][class_name])

        row.append(epoch_data["Mean IoU"])
        rows.append(row)


    df = pd.DataFrame(rows, columns=columns)
    
    df.to_csv(output_file)



if __name__ == '__main__':
    write_to_csv("ade20k-resnet101-upernet",
         "./ckpt1/ade20k-resnet101-upernet/result/val_evaluation_start-5_interval-5_2021-02-17_23-24-41.json",
         [65, 80],
         "./ckpt1/ade20k-resnet101-upernet/result/out.csv"
    )