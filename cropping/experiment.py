# iou analyses


from tensorcrop import *
from utils.annolist.AnnotationLib import AnnoRect
from utils.train_utils import rescale_boxes
import csv
from utils.rect import Rect


def rescale_boxes2(current_shape, x1, x2, y1, y2, target_width, target_height,):
    x_scale = target_width / float(current_shape[0])
    y_scale = target_height / float(current_shape[1])
    assert x1 < x2  # one of the annotations has negative width!
    x1 *= x_scale
    x2 *= x_scale
    assert y1 < y2  # one of the annotations has negative height!
    y1 *= y_scale
    y2 *= y_scale
    return Rect(x1, y1, x2 - x1, y2 - y1, 0.0)


data_dir = 'data/'
lbl_names = ['Type_1', 'Type_2', 'Type_3']
output_dir = 'testcrop/'


cropper = TensorCrop(data_dir, output_dir)
train_flat = getAllTrainingFlat(data_dir, lbl_names)
true_box_list = getTrueBoundingBoxes(data_dir)


images = true_box_list[:, 0]
rects = true_box_list[:, 1]
true_rects = []
for tr in rects:
    print tr['x1'], tr['y1'], tr['x2'] - tr['x1'], tr['y2'] - tr['y1']
    true_rects.append(Rect(tr['x1'], tr['y1'], tr['x2'] - tr['x1'], tr['y2'] - tr['y1'], 0.0))


pred = cropper.tb_predict(train_flat)
start = 0


with open("newtest.csv", "wb") as f:
    writer = csv.writer(f)
    for index, (i, img, rect) in enumerate(pred):

        info = []
        info.append('image: ' + i)
        shape = img.size
        if rect == []:
            info.append('iou: ' + str(-1))
        else:
            print rect[0].x1, rect[0].y1, rect[0].x2 - rect[0].x1, rect[0].y2 - rect[0].y1
            pred_rect = Rect(rect[0].x1, rect[0].y1, rect[0].x2 -
                             rect[0].x1, rect[0].y2 - rect[0].y1, 0.0)
            #         print rect[0]
            # draw image original
            #         print ("draw orginal image")
            #         img2 = img.copy()
            #         drawImageAnno(img,true_anno[index+start],sizew=0,sizeh=0)
            #         # draw image original
            #         print ("draw resized image")
            r_true = true_rects[index + start]
            true_rects[index + start] = rescale_boxes2(shape,
                                                       r_true.cx,
                                                       r_true.cx + r_true.width,
                                                       r_true.cy,
                                                       r_true.cy + r_true.height,
                                                       480, 640)
    #         drawImageAnno(img2,true_anno[index+start])
    #         # draw predict
    #         print ("draw predicted image")
    #         drawImage(pred_images[index])
            info.append('iou: ' + str(true_rects[index + start].iou(pred_rect)))

        writer.writerow(info)
