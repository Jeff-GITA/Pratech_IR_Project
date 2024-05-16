#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 14:47:02 2024

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


import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from imutils import paths
import os
import shutil
# import functions as f




#%%
###############################################################################
############################ Annotation Classes ###############################
###############################################################################


class annotations_class:
    """
    This class is used to perform and save the annotations in images.
    - Input     -> image_path: is the path of the folder where the images to annotate are located.
                -> annotations_path: In the path where the annotated information is going to save.
    
    
    """
    def __init__(self, image_path, annotations_path):
        
        #### Paths and files ####
        ## save tha image path ##
        self.image_path = image_path
        ## save the image name to builld the annotations name ##
        self.image_name = image_path.split("/")[-1].split(".")[0]
        ## Save the annotations path ##
        self.annotations_path = annotations_path
        
    
    def annotator(self):
        
        print("\n#######################################")
        print(f"{self.image_name} - Annotations:")
        print("#######################################\n")
        
        ## Load image ##
        self.img = cv2.imread(self.image_path)
        ## Make a copy to draw the annotations ##
        self.img_ann = self.img.copy()
        
        ## annotation parameters (bounding box) ###
        self.annotations_list = []
        self.x_init = 0
        self.y_init = 0
        self.x_end = 0
        self.y_end = 0
        

        ## Show one image ##
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
        # Resize the window to the desired dimensions ##
        cv2.resizeWindow("image", 550, 650) ## small screen
        # cv2.resizeWindow("image", 800, 900) ## large screen
        cv2.imshow("image", self.img_ann) 
        
        ## Set the mause events ##
        cv2.setMouseCallback("image", self.mouse_events)
        
        ## Destroy the image ##
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def mouse_events(self, event, x, y, flags, params):
        
        ## When the click is pressed ##
        if(event == cv2.EVENT_LBUTTONDOWN):
            # print(f"Down (x,y): ({x},{y})")
            ## Take the initial dimentions ##
            self.x_init = x
            self.y_init = y
            
        elif(event == cv2.EVENT_LBUTTONUP):
            
            ## When the clisck is released ##
            self.x_end = x
            self.y_end = y
            ## compute wheight and height to save the annotations ##
            w = self.x_end - self.x_init
            h = self.y_end - self.y_init
            
            print(f"Down (x,y): ({self.x_init},{self.y_init})")
            print(f"Up (x,y): ({self.x_end},{self.y_end})")
            
            ## Drawing the selected bounding box ##
            cv2.rectangle(self.img_ann, (self.x_init, self.y_init), (self.x_end, self.y_end), (0,255,0), 5)
            ## plot the updated image ##
            cv2.imshow("image", self.img_ann)
            
            ## save the selected annotations ##
            self.annotations_list.append([self.x_init, self.y_init, w, h])
            print("Annotations list lenght: ",len(self.annotations_list))
            
    def print_image(self, img):
        
        #####################################
        #### Function to plot the images ####
        #####################################
        
        plt.figure()
        plt.imshow(img)
        plt.title(f"Img - {img.shape}")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        
    def create_dir(self, dir_path):
        
        ## Chech if dir exist ##
        if(os.path.isdir(dir_path)):
            ## Remove ##
            self.remove_dir(dir_path)
        
        ## Create dir ##
        os.mkdir(dir_path)
    
    def remove_dir(self, dir_path):
        shutil.rmtree(dir_path)
    
    
    def labeling(self):
        """
        This function is used to label the previos annotations 
        
        """
        
        print("\n#######################################")
        print(f"{self.image_name} - Labeling:")
        print("#######################################\n")
        
        ## Create path to save ##
        save_path = os.path.join(self.annotations_path, self.image_name)
        ## Create dir ##
        self.create_dir(save_path)
       
        ## Get list of annotations ##
        annotated_list = np.array(self.annotations_list)
        # print(len(annotated_list[0]))
        
        ## Create dictionary to save the annotations ##
        annotation_dict = {}
        annotation_dict["image_name"] = self.image_name
        annotation_dict["docid"] = []
        annotation_dict["label"] = []
        annotation_dict["x"] = []
        annotation_dict["y"] = []
        annotation_dict["w"] = []
        annotation_dict["h"] = []
        
        
        ## List to save the asigned labels ##
        labels = []
        for i in range(len(annotated_list)):
            
            ## Name of the selected object ##
            object_name = self.image_name+f"_obj{i}.png"
            print()
            print(object_name)
            
            ## take the annotations bounding boxes ##
            x, y ,w, h = annotated_list[i]
            
            ## This could lead to a wring boundig box ##
            if (w<=0 | h<=0):
                continue
            
            ## take the selected object ##
            object_img = self.img[y : y+h, x : x+w, :]
            ## print the object ##
            self.print_image(object_img)
            
            
            print("Input a label: ")
            lb = input()
            
            ## save the input label ##
            labels.append(lb)
            ## create annotated object path ##
            annotated_image_file = os.path.join(save_path, object_name)
            ## save the annotated object image ##
            cv2.imwrite(annotated_image_file, object_img)
            
            ## Save data ##
            annotation_dict["docid"].append(object_name)
            annotation_dict["label"].append(lb)
            annotation_dict["x"].append(x)
            annotation_dict["y"].append(y)
            annotation_dict["w"].append(w)
            annotation_dict["h"].append(h)
            # # break
        
        print("\n#######################################")
        print("Labeled finished")
        
        
        print("\saving labels...")
        
        print("#######################################")
        
        
        #### Save all the information ####
        
        ## create the file to save the annotations ##
        annotations_file = os.path.join(save_path, "annotations_"+self.image_name+".csv")
        ## Convert the dict into dataframe ##
        annotation_df = pd.DataFrame.from_dict(annotation_dict)
        ## Organize the index ##
        annotation_df.reset_index(drop=True, inplace=True)
        ## save the info in csv file ##
        annotation_df.to_csv(annotations_file, index=False)
        
        


#%%
###############################################################################
############################ Annotation process ###############################
###############################################################################

#### Parameters ####
## Images path ##
# images_path = "data/docs_exm/"
images_path = "../Datasets/Tobacco 800 Dataset/tobacco800/"
## Save annotations info ##
annotated_path = "../Data/Annotated_Images/"


## List images from path ##
images_list = list(paths.list_images(images_path))


print("\n#######################")
print("Amount of data:")
print("All data:",len(images_list))

## Remove annotated images from list ##
annot_images_list = os.listdir(annotated_path)
print("Ann data:",len(annot_images_list))

for img_f in annot_images_list:
    file = os.path.join(images_path, img_f+".png")
    images_list.remove(file)

print("To Ann data:",len(images_list))
print("#######################")


#### Loop ####
n_i = 0

for n_i in range(len(images_list)):
    
    
    print("\n\nDo you want to continue labeling? (y/n):")
    stop = input()
    if (stop.lower() == "y"):
        print("\ncontinue...")
        
    else:
        print("\nProcess finished.")
        break
    
    ## Create annotator insatance ##
    ann_class = annotations_class(images_list[n_i], annotated_path)
    
    ## Make annotations ##
    annotator_flag = "y"
    while (annotator_flag.lower() == "y"):    
        ann_class.annotator()
        print("\nRepeat annotaions? (y/n):")
        annotator_flag = input()
    
    ## Print annotated image ##
    ann_class.print_image(ann_class.img_ann)
    
    ## Label annotations ##
    label_flag = "y"
    while (label_flag.lower() == "y"):
        ann_class.labeling()
        print("\nRepeat labeling? (y/n):")
        label_flag = input()
        
        
