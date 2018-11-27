import os
import glob
import shutil
import random

root_dir = os.getcwd()
data_dir = os.path.join(root_dir,'PPMI')
validation_dir = ps.path.join(root_dir,'Validation')
test_dir = os.path.join(root_dir,'Test')
train_dir = ps.path.join(root_dir,'Train')
final_data_dir = os.path.join(root_dir,'PPMI')

for root,dirs,files in os.walk(data_dir):
    classes = dirs
    break

os.mkdir(validation_dir)
os.mkdir(os.path.join(validation_dir,classes[0]))
os.mkdir(os.path.join(validation_dir,classes[1]))
os.mkdir(test_dir)
os.mkdir(os.path.join(test_dir,classes[0]))
os.mkdir(os.path.join(test_dir,classes[1]))

list_of_filenames = glob.glob(data_dir + '**/*.nii.gz')
list_of_filenames = random.shuffle(list_of_filenames)
num = len(list_of_filenames)
num_test = num_val =  num//10
test_filename = list_of_filenames[:num_test]
val_filenames = list_of_filenames[-num_test:]

for filename in test_filename:
    dir, file_name = os.path.split(filename)
    _ , class_ = os.path.split(dir)
    new_path = os.path.join(test_dir,class,file_name)
    shutil.move(filename,new_path)

for filename in val_filename:
    dir, file_name = os.path.split(filename)
    _ , class_ = os.path.split(dir)
    new_path = os.path.join(validation_dir,class,file_name)
    shutil.move(filename,new_path)

os.rename(data_dir,train_dir)
os.mkdir(final_data_dir)
shutil.move(train_dir,final_data_dir)
shutil.move(test_dir,final_data_dir)
shutil.move(validation_dir,final_data_dir)
