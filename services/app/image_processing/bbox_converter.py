# convert YOLO bbox coordinates
def bbox_convert(image, bboxes):
    image_h, image_w, _ = image.shape
    out_boxes, out_classes, num_boxes = bboxes
    bbox_coords = []
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) == 3 or int(out_classes[0][i]) == 4: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        bbox_list = []
        x1 = coor[1]
        y1 = coor[0]
        x3 = coor[3]
        y3 = coor[2]
        x2 = x3
        y2 = y1
        x4 = x1
        y4 = y3
        bbox_list.append(int(x1))
        bbox_list.append(int(y1))
        bbox_list.append(int(x2))
        bbox_list.append(int(y2))
        bbox_list.append(int(x3))
        bbox_list.append(int(y3))
        bbox_list.append(int(x4))
        bbox_list.append(int(y4))
        bbox_list.append(int(out_classes[0][i]))
        bbox_coords.append(bbox_list)
    return bbox_coords
