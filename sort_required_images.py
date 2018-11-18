"""
This file will move/copy all the images that have T1 weighting and have slice thickness < 2mm.
For Parkinson's, the directory structure should be as follows:
Root directory
    - all the code and .csv files
    - PD_BET (where all the .nii.gz skull-stripped images of PD are as given by nikunj)
    - HC_BET (list of dirs corresponding to subjectId as given by pavani)
        - subjectId
            - where all the .nii.gz skull-stripped images of HC are following the original directory structure
        - subjectId
        - and so on
I wrote two separate functions as I had told pavani to preserve the original directory structure for faster traversal
meanwhile,in the Parkinson's data, all the images are in one folder
To save space there is an option to move the file instead of copy, just set copy=False
"""

import pandas as pd
import glob
import os
import shutil

def is_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return os.path.abspath(directory)

def sort_PD(root_directory,copy=True):
    sorted_directory = is_directory('PD_BET_2_T1')
    PD_metadata = pd.read_csv('PD_metadata.csv')
    PD_dir = os.path.join(root_directory,'PD_BET')
    print("Now sorting Parkinson's Disease images")
    for identifier in PD_metadata['seriesIdentifier']:
        key = 'S' + str(identifier)
        for file in glob.glob(os.path.join(PD_dir,'*'+key+'*'+'.gz')):
            try:
                if copy:
                    shutil.copy(file,sorted_directory)
                else:
                    shutil.move(file,sorted_directory)
            except shutil.Error:
                continue
    print("Done sorting the required Parkinson's Disease images")

def sort_HC(root_directory,copy=True):
    sorted_directory = is_directory('HC_BET_2_T1')
    HC_metadata = pd.read_csv('HC_metadata.csv')
    HC_metadata = HC_metadata.sort_values('subjectIdentifier')
    HC_dir = os.path.join(root_directory,'HC_BET')
    print('Now sorting Healthy Control images')
    for index, row in HC_metadata.iterrows():
        subjectId = row['subjectIdentifier']
        identifier = row['seriesIdentifier']
        temp_path = os.path.join(HC_dir,subjectId)
        key = 'S' + str(identifier)
        for file in glob.glob(os.path.join(temp_path,'*'+key+'*'+'.gz')):
            try:
                if copy:
                    shutil.copy(file,sorted_directory)
                else:
                    shutil.move(file,sorted_directory)
            except shutil.Error:
                print('Error processing: ', file)
                continue
    print('Done sorting the required Healthy Control images')

def main():
        root_directory = os.getcwd()
        sort_PD(root_directory,copy=False)
        sort_HC(root_directory)

if __name__ == '__main__':
    main()
