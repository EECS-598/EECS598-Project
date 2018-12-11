import os
import glob
import shutil
import random

def create_directory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print('DIRECTORY CREATED: ', dir)

root_dir = os.getcwd()
data_dir = os.path.join(root_dir,'PPMI')
# validation_dir = ps.path.join(root_dir,'Validation')
test_dir = os.path.join(root_dir,'Test')
train_dir = os.path.join(root_dir,'Train')
final_data_dir = os.path.join(root_dir,'PPMI')

for root,dirs,files in os.walk(data_dir):
    classes = dirs
    break

print('CLASSES ARE: ', classes)

# os.mkdir(validation_dir)
# os.mkdir(os.path.join(validation_dir,classes[0]))
# os.mkdir(os.path.join(validation_dir,classes[1]))
create_directory(test_dir)
create_directory(os.path.join(test_dir,classes[0]))
create_directory(os.path.join(test_dir,classes[1]))

list_of_filenames_PD = glob.glob(os.path.join(data_dir,'**','*.nii.gz'))
list_of_filenames_HC = glob.glob(os.path.join(data_dir,'**','*.nii'))
list_of_all_filenames = list_of_filenames_PD + list_of_filenames_HC
total = len(list_of_all_filenames)
print('TOTAL FILENAMES ARE: ', total)

num_test = 0

for i,filename in enumerate(list_of_all_filenames):
    if i%10 == 0:
        num_test += 1
        dir, file_name = os.path.split(filename)
        _ , class_ = os.path.split(dir)
        new_path = os.path.join(test_dir,class_,file_name)
        shutil.move(filename,new_path)

print('TOTAL TEST FILES ARE: ', num_test)

# for filename in val_filename:
#     dir, file_name = os.path.split(filename)
#     _ , class_ = os.path.split(dir)
#     new_path = os.path.join(validation_dir,class,file_name)
#     shutil.move(filename,new_path)

os.rename(data_dir,train_dir)
os.mkdir(final_data_dir)
shutil.move(train_dir,final_data_dir)
shutil.move(test_dir,final_data_dir)
# shutil.move(validation_dir,final_data_dir)
