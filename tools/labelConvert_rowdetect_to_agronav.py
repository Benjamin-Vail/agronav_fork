import json
import tkinter as tk
from tkinter import filedialog
from tkinter.font import nametofont
import numpy as np
import os
import pandas as pd
import cv2
from sklearn.linear_model import LinearRegression
import argparse

def linear_regression(xy_pairs):
    x = np.array(xy_pairs[0]).reshape(-1, 1)
    y = np.array(xy_pairs[1])

    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_

def find_ego_idx(key_points, fileName, parent_folder):
    dist_x = []
    idxs = []
    l_val = 0
    r_val = 0
    for idx, key_point in key_points.iterrows():

        p_y = np.polyfit(key_point['x'], key_point['y'], N_DEGREE)
        x_min = np.polyval(p_y, 0)
        dist_x.append(x_min-0.5)
        idxs.append(idx)
    dist_x = np.array(dist_x)
    idxs = np.array(idxs)
    idxs_l = idxs[dist_x > 0]#idxs[dist_x > 0]
    xs_l = dist_x[dist_x > 0]#dist_x[dist_x > 0]
    
    idxs_r = idxs[dist_x <= 0]#idxs[dist_x <= 0]
    xs_r = dist_x[dist_x <= 0]#dist_x[dist_x <= 0]

    print("---------------")
    print(idxs)
    print(dist_x)
    print((xs_l,xs_r))
    print("---------------")

    if(len(xs_r) == 0):
        with open(parent_folder + "/deleted_files.txt", 'a') as txtFile:
            txtFile.write("\n" + str(fileName))
            txtFile.close()

        return 100, 100
    elif(len(xs_l) == 0):
        with open(parent_folder + "/deleted_files.txt", 'a') as txtFile:
            txtFile.write("\n" + str(fileName))
            txtFile.close()
        return 100, 100
    else:
        return idxs_l[np.argmax(xs_l)], idxs_r[np.argmin(xs_r)]

def main():
    
    parser = argparse.ArgumentParser(description='Converts Json labels to .txt with ego trimming and format: N y0start x0start y0end x0end ... yNstart xNstart yNend xNend')
    # arguments from command line
    parser.add_argument('--labels_input', required=True, help="The path to the input labels")
    parser.add_argument('--labels_output', required=True, help='The path to output the converted labels to')
    args = parser.parse_args()

    input__folder_path = args.labels_input
    output_folder_path = args.labels_output
    progress = 0
    total_files = len(os.listdir(input__folder_path))

    for file in os.listdir(input__folder_path): 
        if(file.endswith(".json")): 
            file_path = input__folder_path + "/" + file 
            parent_folder = os.path.join(input__folder_path, os.pardir)
            image_path = parent_folder + "/imgs/" + file.replace(".json", ".jpg")  
            coords = []
            print("img path: " + image_path)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR) 
            print(file)
            print(str(progress) + " / " + str(total_files))

            with open(file_path) as f:
                data = json.load(f)
                data_df = pd.json_normalize(data, 'labels')
                f.close()


            idxs_l, idxs_r = find_ego_idx(data_df, file, parent_folder)

            if(idxs_l == 100 and idxs_r == 100):
                print("bad file")
                os.remove(file_path)
                os.remove(image_path)
                
                pass

            else:
                height, width = img.shape[:2]

                for i in range(0,len(data_df.iloc[idxs_l]['x'])):
                    cv2.circle(img, (int(data_df.iloc[idxs_l]['x'][i]),int(data_df.iloc[idxs_l]['y'][i])), radius=5, color=(255, 0, 0), thickness=-1)

                for i in range(0,len(data_df.iloc[idxs_r]['x'])):
                    cv2.circle(img, (int(data_df.iloc[idxs_r]['x'][i]),int(data_df.iloc[idxs_r]['y'][i])), radius=5, color=(0, 0, 255), thickness=-1)

                L_slope, L_intercept = linear_regression((data_df.iloc[idxs_l]['x'], data_df.iloc[idxs_l]['y']))
                R_slope, R_intercept = linear_regression((data_df.iloc[idxs_r]['x'], data_df.iloc[idxs_r]['y']))

                L_start = (int((height-L_intercept)/L_slope), height)
                L_end = (int((0-L_intercept)/L_slope), 0)
                cv2.line(img, L_start, L_end, (255,0,0), 5)

                R_start = (int((0-R_intercept)/R_slope), 0)
                R_end = (int((height-R_intercept)/R_slope), height)
                cv2.line(img, R_start, R_end, (0,0,255), 5)

                # should be y,x not x,y
                coords = (L_start[1], L_start[0], L_end[1], L_end[0], R_start[1], R_start[0], R_end[1], R_end[0])
                                
                # Display the image
                label_path = output_folder_path + '/' + file.replace('.json', '.txt')

                with open(label_path, 'x') as f:
                    f.write(f"{2}")
                    for coord in coords:
                        f.write(f" {coord}")
                    f.close()

        progress +=1

if __name__ == "__main__":
    N_DEGREE = 2
    main()