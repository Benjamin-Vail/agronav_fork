import cv2

img_path = "C:/Users/Benv86/OneDrive - Kansas State University/GitLab/AgroNav/preds-converted/imgs/726.jpg"
pred_label_path = "C:/Users/Benv86/OneDrive - Kansas State University/GitLab/AgroNav/preds-converted/labels-converted/726.txt"
gt_label_path = "C:/Users/Benv86/OneDrive - Kansas State University/GitLab/AgroNav/test-converted/test_labels_converted/726.txt"

img = cv2.imread(img_path)
pred_txt = open(pred_label_path)
gt_txt = open(gt_label_path)
gt_coords = gt_txt.readlines()
pred_coords = pred_txt.readlines()
print(gt_coords)

gt_rows = gt_coords[0].split(' ')[0]
for i in range(int(gt_rows)):
    coords = gt_coords[0].split(' ')[i*4+1:i*4+5]
    print((coords[1], coords[0]))
    cv2.line(img, (int(coords[1]), int(coords[0])), (int(coords[3]), int(coords[2])), (255,0,0), 5)
    
pred_rows = pred_coords[0].split(' ')[0]
for i in range(int(pred_rows)):
    coords = pred_coords[0].split(' ')[i*4+1:i*4+5]
    print((coords[1], coords[0]))
    cv2.line(img, (int(coords[1]), int(coords[0])), (int(coords[3]), int(coords[2])), (0,255,0), 5)


cv2.imshow('Image with Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

    