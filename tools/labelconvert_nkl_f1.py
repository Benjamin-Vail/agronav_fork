import json
import tkinter as tk
from tkinter import filedialog
from tkinter.font import nametofont
import numpy as np
import os
import pandas as pd
import cv2
from sklearn.linear_model import LinearRegression

def linear_regression(xy_pairs):
    x = np.array(xy_pairs[0]).reshape(-1, 1)
    y = np.array(xy_pairs[1])
    # x = np.array([pair[0] for pair in xy_pairs]).reshape(-1, 1)
    # y = np.array([pair[1] for pair in xy_pairs])

    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_

def find_ego_idx(key_points, fileName, parent_folder):
    dist_x = []
    idxs = []
    l_val = 0
    r_val = 0
    for idx, key_point in key_points.iterrows():

        print("key_point: ",key_point)
        print("idx: ",idx)
        # p_y = np.polyfit(key_point['x'], key_point['y'], N_DEGREE)
        # x_min = np.polyval(p_y, 0)
        # dist_x.append(x_min-0.5)
        # idxs.append(idx)
        

    # dist_x = np.array(dist_x)
    # idxs = np.array(idxs)
    # idxs_l = idxs[dist_x > 0]#idxs[dist_x > 0]
    # xs_l = dist_x[dist_x > 0]#dist_x[dist_x > 0]
    
    # idxs_r = idxs[dist_x <= 0]#idxs[dist_x <= 0]
    # xs_r = dist_x[dist_x <= 0]#dist_x[dist_x <= 0]

    # print("---------------")
    # print(idxs)
    # print(dist_x)
    # print((xs_l,xs_r))
    # print("---------------")

    # if(len(xs_r) == 0):
    #     # with open("/home/deleted_files.txt", 'w') as txtFile:
    #     #         txtFile.write(str(fileName))
    #     #         txtFile.close()

    #     with open(parent_folder + "/deleted_files.txt", 'a') as txtFile:
    #         txtFile.write("\n" + str(fileName))
    #         txtFile.close()

    #     return 100, 100
    # elif(len(xs_l) == 0):
    #     # with open("/home/deleted_files.txt", 'w') as txtFile:
    #     #         txtFile.write(str(fileName))
    #     #         txtFile.close()

    #     with open(parent_folder + "/deleted_files.txt", 'a') as txtFile:
    #         txtFile.write("\n" + str(fileName))
    #         txtFile.close()

    #     return 100, 100
    # else:
    #     return idxs_l[np.argmax(xs_l)], idxs_r[np.argmin(xs_r)]

def main():
    root = tk.Tk()
    default_font = nametofont("TkDefaultFont")
    default_font.configure(size=20)

    menu_font = nametofont("TkMenuFont")
    menu_font.configure(size=20)

    icon_font = nametofont("TkIconFont")
    icon_font.configure(size=20)

    text_font = nametofont("TkTextFont")
    text_font.configure(size=20)

    root.withdraw()

    input__folder_path = filedialog.askdirectory(title="Select the input folder", initialdir="/mnt/data/BenV/row-detection-agronav/train/labels-orig")
    output_folder_path = filedialog.askdirectory(title="Select the output folder", initialdir="/mnt/data/BenV/row-detection-agronav/train/labels")
    progress = 0
    total_files = len(os.listdir(input__folder_path))

    for file in os.listdir(input__folder_path): 
        if(file.endswith(".json")): 
            file_path = input__folder_path + "/" + file 
            parent_folder = os.path.join(input__folder_path, os.pardir)
            image_path = parent_folder + "/images/" + file.replace(".json", ".jpg")  
            coords = []
            print("img path: " + image_path)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR) 
            print(file)
            print(str(progress) + " / " + str(total_files))

            with open(file_path) as f:
                data = json.load(f)
                data_df = pd.json_normalize(data, 'labels')
                f.close()
                
            coords_all_rows = []
            height, width = img.shape[:2]
            for idx, key_point in data_df.iterrows():
                L_slope, L_intercept = linear_regression((key_point['x'], key_point['y']))
                
                L_start = (int((height-L_intercept)/L_slope), height)
                L_end = (int((0-L_intercept)/L_slope), 0)

                # should be (y1, x1, y2, x2)
                coords = (L_start[1], L_start[0], L_end[1], L_end[0])
                coords_all_rows.append(coords)
                
            num_rows = len(coords_all_rows)
            label_path = output_folder_path + '/' + file.replace('.json', '.txt')
            with open(label_path, 'x') as f:
                f.write(f"{int(num_rows)}")
                for coord in coords_all_rows:
                    f.write(" " + " ".join(map(str, coord)))
                    # f.write(f" {coord}")
                f.close()

            # idxs_l, idxs_r = find_ego_idx(data_df, file, parent_folder)

            # if(idxs_l == 100 and idxs_r == 100):
            #     print("bad file")
            #     os.remove(file_path)
            #     os.remove(image_path)
                
            #     pass


            

            # for i in range(0,len(data_df.iloc[idxs_l]['x'])):
            #     cv2.circle(img, (int(data_df.iloc[idxs_l]['x'][i]),int(data_df.iloc[idxs_l]['y'][i])), radius=5, color=(255, 0, 0), thickness=-1)

            # for i in range(0,len(data_df.iloc[idxs_r]['x'])):
            #     cv2.circle(img, (int(data_df.iloc[idxs_r]['x'][i]),int(data_df.iloc[idxs_r]['y'][i])), radius=5, color=(0, 0, 255), thickness=-1)

            # L_slope, L_intercept = linear_regression((data_df.iloc[idxs_l]['x'], data_df.iloc[idxs_l]['y']))
            # R_slope, R_intercept = linear_regression((data_df.iloc[idxs_r]['x'], data_df.iloc[idxs_r]['y']))

            # L_start = (int((height-L_intercept)/L_slope), height)
            # L_end = (int((0-L_intercept)/L_slope), 0)
            # cv2.line(img, L_start, L_end, (255,0,0), 5)

            # R_start = (int((0-R_intercept)/R_slope), 0)
            # R_end = (int((height-R_intercept)/R_slope), height)
            # cv2.line(img, R_start, R_end, (0,0,255), 5)

            # coords = (L_start[0], L_start[1], L_end[0], L_end[1], R_start[0], R_start[1], R_end[0], R_end[1])
                            
            # # Display the image
            # label_path = output_folder_path + '/' + file.replace('.json', '.txt')

            # with open(label_path, 'x') as f:
            #     f.write(f"{2}")
            #     for coord in coords:
            #         f.write(f" {coord}")
            #     f.close()

                # image_out_path = label_path.replace('.txt', '.jpg')
                # cv2.imwrite(image_out_path, img)
                # cv2.imshow('Image with Lines', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        progress +=1

if __name__ == "__main__":
    N_DEGREE = 2
    main()

            

# def draw_line(y, x, angle, image, color=(0,0,255), num_directions=24):
#     '''
#     Draw a line with point y, x, angle in image with color.
#     '''
#     cv2.circle(image, (x, y), 2, color, 2)
#     H, W = image.shape[:2]
#     angle = int2arc(angle, num_directions)
#     point1, point2 = get_boundary_point(y, x, angle, H, W)
#     cv2.line(image, point1, point2, color, 2)
#     return image

