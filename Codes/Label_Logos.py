#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:47:21 2024

@author: TheJeff
"""

#%%
###############################################################################
############################ Script description ###############################
###############################################################################






#%%
###############################################################################
################################# Libraries ###################################
###############################################################################


import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os


#%%
###############################################################################
################################# Functions ###################################
###############################################################################
def print_image(img):

    plt.figure()
    plt.imshow(img)
    plt.title(f"Img - {img.shape}")
    plt.xticks([])
    plt.yticks([])
    plt.show()


#%%
###############################################################################
############################# Labeling process ################################
###############################################################################


objects_path = "../Data/Objects_Database/"


objects_file = "../Data/Objects_Database/Selected_Objects_Information.csv"
objects_df = pd.read_csv(objects_file)


label = "logo"


# objects_df[f"{label}_label"] = [0]*objects_df.shape[0]
print(objects_df.columns)
print(objects_df)

object_data_df = objects_df.loc[(objects_df["label"]==label) & (objects_df[f"{label}_label"]==0), :]


print("All objects: ", objects_df.loc[(objects_df["label"]==label)].shape)
print("Not labeled objects: ",object_data_df.shape)


idx_list = []
img_label_list = []


for idx in object_data_df.index:
    
    ## Create image path ##
    image_path = os.path.join(objects_path, label, object_data_df.loc[idx,"docid"])
    ## load image ##
    image = cv2.imread(image_path)
    ## print image ##
    print_image(image)
    
    print("\n\nType 0 to stop the labeling\n-> type the object label:\n")
    l1 = int(input())
    
    
    if (l1 == 0):
        #### Stop labeling ####
        objects_df.loc[idx_list, f"{label}_label"] = img_label_list
        objects_df.to_csv(objects_file, index=False)
        
        print("\nProcess finished.")
        break
    
        
    else:
        #### Labeling #####
        ## Save label ##
        img_label_list.append(l1)
        ## Save index ##
        idx_list.append(idx)
        
        
        
    if(len(img_label_list)==object_data_df.shape[0]):
        #### Stop labeling ####
        objects_df.loc[idx_list, f"{label}_label"] = img_label_list
        objects_df.to_csv(objects_file, index=False)
        
        print("\nAll Labeled.")


print(objects_df)



