import pandas as pd
import os
import glob

def create_csv_file(directory,class_map):
    list_of_filenames = glob.glob(directory + '**/*.nii.gz')
    image_paths = []
    labels = []
    for filename in list_of_filenames:
        dir, file_name = os.path.split(filename)
        _ , class_ = os.path.split(dir)
        image_paths.append(filename)
        labels.append(class_map[class_])

    df = pd.Dataframe({'image_path':image_paths,'label':labels})
    return df

current_dir = os.getcwd()
validation_dir = os.path.join(current_dir,'Validation')
test_dir = os.path.join(current_dir,'Test')
train_dir = os.path.join(current_dir,'Train')

class_map = {'PD':1, 'HC':0}

df_train = create_csv_file(train_dir,class_map)
df_val = create_csv_file(validation_dir,class_map)
df_test = create_csv_file(test_dir,class_map)

df_train.to_csv('train.csv',index=False)
df_val.to_csv('validation.csv',index=False)
df_test.to_csv('test.csv',index=False)
