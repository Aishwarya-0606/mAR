import glob
import json
import os
import sys
import shutil
import argparse

parser = argparse.ArgumentParser(description='Set the value of MINIMUM OVERLAP value for IOU such as 0.7')
parser.add_argument('--min_overlap', '-min_ov', default=0.7, type=float, help='MINIMUM OVERLAP value for IOU')
args = parser.parse_args()
if not (0.0 <args.min_overlap<= 1):
    print("--min_overlap value should be between 0 and 1")
sys.exit(1)
print("MINIMUM OVERLAP value for IOU: ", args.min_overlap)

MINOVERLAP = args.min_overlap
GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth')
DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results')

# optional
IMG_PATH = os.path.join(os.getcwd(), 'input', 'images-optional')
show_animation = False
no_animation = False
if os.path.exists(IMG_PATH):
    for dirpath, dirnames, files in os.walk(IMG_PATH):
        if not files:
            print("No image files")
            no_animation = True
else:
    no_animation = True

if not no_animation:
    show_animation = True

# Create directory .temp_files
TEMP_FILE_PATH = ".temp_files"
if not os.path.exists(TEMP_FILE_PATH):
    os.makedirs(TEMP_FILE_PATH)
output_files_path = "output"
if os.path.exists(output_files_path):
    shutil.rmtree(output_files_path)
os.makedirs(output_files_path)


def error(msg):
    print(msg)
    sys.exit(1)


def file_lines_to_list(file_path):
    with open(file_path, 'r') as file:
        file_content = file.readlines()
    file_content = [line.strip() for line in file_content]
    return file_content


def get_GT_data():
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("No ground truth files found")
    ground_truth_files_list.sort()
    gt_files = []
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split('.txt', 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        # check if corresponding detection-results file exists
        temp_dr_path = os.path.join(DR_PATH, (file_id + '.txt'))
        if not os.path.exists(temp_dr_path):
            error_msg = "Error. File not found {}\n".format(temp_dr_path)
            error_msg += "Same text file should be present in detection-result folder"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        for line in lines_list:
            try:
                xmin, ymin, _, _, xmax, ymax, _, _, original_text = line.split(',', 8)
            except ValueError:
                error_msg = "Error: File " + txt_file + " is in wrong format.\n"
                error_msg += " Expected: xmin, ymin, x, y, xmax, ymax, x, y, original_text\n"
                error_msg += " Received: " + line
                error(error_msg)
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            bounding_boxes.append({"bbox": bbox, "used": False})
        new_temp_file = TEMP_FILE_PATH + os.sep + file_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)
    return gt_files


def calculate_iou(bb_dr, bb_gt):
    # coordinates of the intersecting rectangle
    rect_cor = [max(bb_dr[0], bb_gt[0]), max(bb_dr[1], bb_gt[1]),
                min(bb_dr[2], bb_gt[2]), min(bb_dr[3], bb_gt[3])]
    iw = rect_cor[2] - rect_cor[0] + 1
    ih = rect_cor[3] - rect_cor[1] + 1
    if iw> 0 and ih> 0:
        # compute overlap (IoU) = area of intersection(ia) / area of union(ua)
        ia = iw * ih
        ua = (bb_dr[2] - bb_dr[0] + 1) * (bb_dr[3] - bb_dr[1] + 1) + \
             (bb_gt[2] - bb_gt[0] + 1) * (bb_gt[3] - bb_gt[1] + 1) - ia
        iou = ia / ua
        return iou
    return -1


def get_true_positives():
    dr_file_list = glob.glob(DR_PATH + '/*.txt')
    dr_file_list.sort()
    count_true_positives = {}
    for txt_file in dr_file_list:
        file_id = txt_file.split('.txt', 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        count_true_positives[file_id] = 0
        temp_gt_path = os.path.join(GT_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_gt_path):
            error_msg = "Error. File not found {}\n".format(temp_gt_path)
            error_msg += "Same text file should be present in ground-truth folder"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            try:
                xmin, ymin, _, _, xmax, ymax, _, _, original_text = line.split(',', 8)
            except ValueError:
                error_msg = "Error: File " + txt_file + " is in wrong format.\n"
                error_msg += " Expected: xmin, ymin, x, y, xmax, ymax, x, y, original_text\n"
                error_msg += " Received: " + line
                error(error_msg)
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            # load detected object bounding-box
            bb_dr = [float(x) for x in bbox.split()]
            # get the corresponding gt file data
            gt_file = TEMP_FILE_PATH + os.sep + file_id + "_ground_truth.json"
            ground_truth_data = json.load(open(gt_file))
            iou_max = -1
            gt_match = -1
            for obj in ground_truth_data:
                bb_gt = [float(x) for x in obj['bbox'].split()]
                iou = calculate_iou(bb_dr, bb_gt)
                if iou>iou_max:
                    iou_max = iou
                    gt_match = obj

            min_overlap = MINOVERLAP
            if iou_max>= min_overlap:
                if not bool(gt_match['used']):
                    gt_match['used'] = True
                    count_true_positives[file_id] += 1
                    with open(gt_file, 'w') as f:
                        f.write(json.dumps(ground_truth_data))
        print("True positive for {txt_file} is {tp}".format(txt_file=file_id, tp=count_true_positives[file_id]))
    shutil.rmtree(TEMP_FILE_PATH)
    return count_true_positives


def calculate_recall():
    get_GT_data()
    true_positive_val = get_true_positives()
    print("\nfile_name ----> recall value")
    total_actual_true = 0
    total_true_positive = 0
    for file_name, tp in true_positive_val.items():
        # print(file_name, " ----> ", tp)
        gt_file_path = GT_PATH + os.sep + file_name + ".txt"
        actual_true_val = len(file_lines_to_list(gt_file_path))
        total_actual_true += actual_true_val
        total_true_positive += tp
        recall = tp / actual_true_val
        print(file_name, " ----> ", recall)
    print("Average Recall value: {}".format(total_true_positive/total_actual_true))


if __name__ == '__main__':
    calculate_recall()
