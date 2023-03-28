import os
import pandas as pd
import re
import cv2
import numpy as np
from tqdm import tqdm

import requests as req
from PIL import Image
from io import BytesIO


IMAGE_CACHE_DIR = 'image_cache'


def load_image(image_path):
    if image_path.startswith('http'):
        image_cache_path = os.path.join(IMAGE_CACHE_DIR, image_path.split('/')[-1])
        if os.path.exists(image_cache_path):
            image = cv2.imread(image_cache_path)
        else:
            response = req.get(image_path)
            image = Image.open(BytesIO(response.content))
            image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_cache_path, image)
    else:
        image_path = eval(repr(image_path).replace('\\', '/'))
        image = cv2.imread(image_path)
    return image

def read_html_nodes(html_path):
    with open(html_path, 'r') as html_file:
        table_nodes = re.split(r'<h4></h4>\n<table>\n', html_file.read())

    table_dict = {}
    for table_node in table_nodes[1:]:
        user_line, lib_line = table_node.split('\n')[1:3]
        user_id = user_line[4:-5]
        lib_id = lib_line[4:-5]
        table_id = f'{user_id}-{lib_id}'
        if table_id not in table_dict:
            table_dict[table_id] = '<h4></h4>\n<table>\n' + table_node
    return table_dict

def read_html_nodes(html_path):
    with open(html_path, 'r') as html_file:
        table_nodes = re.split(r'<h4></h4>\n<table>\n', html_file.read())

    table_dict = {}
    for table_node in table_nodes[1:]:
        table_node_lines = table_node.split('\n')
        user_line, lib_line = table_node_lines[5:7]
        user_id = user_line[4:-5]
        lib_id = lib_line[4:-5]
        table_id = f'{user_id},{lib_id}'

        user_path_line, lib_path_line = table_node_lines[13:15]
        user_path = user_path_line.split(' ')[1].split('=')[1]
        lib_path = lib_path_line.split(' ')[1].split('=')[1]
        if table_id not in table_dict:
            table_dict[table_id] = [user_path, lib_path]
    return table_dict


def sort_html_with_xlsx(xlsx_path, xml_dir='', test_idx=-1):
    assert os.path.exists(xml_dir)
    data = pd.read_excel(xlsx_path)
    print(f'read {xlsx_path}, has {data.shape[0]} rows, {data.shape[1]} columns.')
    test_groups = data.groupby('test')
    for key, data_frame in test_groups:
        if test_idx != -1 and key != test_idx: continue
        html_path = os.path.join(xml_dir, f'test_{key}.html')
        if not os.path.exists(html_path): continue
        table_dict = read_html_nodes(html_path)
        table_list = []
        for i in range(data_frame.shape[0]):
            user_id = data_frame.iat[i, 0]
            lib_id = data_frame.iat[i, 1]
            table_id = f'{user_id},{lib_id}'
            if table_id in table_dict:
                table_list.append(table_dict[table_id])
        sorted_html_path = os.path.join(xml_dir, f'test_{key}_sorted.html')
        with open(sorted_html_path, 'w') as f:
            f.write(''.join(table_list))


def resize_image(image, size=(960, 1080)):
    h, w = image.shape[:2]
    scale_h = h / size[1]
    scale_w = w / size[0]
    if scale_h > 1 or scale_w > 1:
        outsize = (int(w / scale_w), int(h / scale_w)) if scale_w > scale_h else (int(w / scale_h), int(h / scale_h))
        image = cv2.resize(image, outsize)
    return image

def read_csv_results(csv_path):
    with open(csv_path, 'r') as f:
        result_lines = f.readlines()
    results = []
    for line in result_lines:
        user_id ,lib_id, key, label = line.rstrip('\n').split(',')
        results.append([user_id ,lib_id, key, int(label)])
    return results


def mark(window_size=(2000, 1000)):
    COLORMAPS = [
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255)
    ]
    cv2.namedWindow('mark', cv2.WINDOW_NORMAL)
    table_dict = read_html_nodes('test.html')
    results = read_csv_results('test.csv')

    result_dict = {}
    test_length = len(results)
    idx = 0
    split_width = 4
    while 1:
        user_id ,lib_id, test_idx, cur_label = results[idx]
        table_id = f'{user_id},{lib_id}'
        if table_id not in table_dict:
            idx += 1
            continue
        user_path, lib_path = table_dict[table_id]
        if table_id in result_dict: cur_label = result_dict[table_id]

        user_image = load_image(user_path)
        lib_image = load_image(lib_path)
        user_height, user_width = user_image.shape[:2]
        lib_height, lib_width = lib_image.shape[:2]
        w_scale = (user_width + lib_width) / (window_size[0] - split_width)
        if w_scale <= 1: w_scale = 1

        user_image = resize_image(user_image, (int(user_width / w_scale), int(user_height / w_scale)))
        lib_image = resize_image(lib_image, (int(lib_width / w_scale), int(lib_height / w_scale)))

        user_h, user_w = user_image.shape[:2]
        lib_h, lib_w = lib_image.shape[:2]

        mark_image_height = max(user_h, lib_h)
        mark_image_width = window_size[0]
        mark_image = np.ones((mark_image_height, mark_image_width, 3), np.uint8)


        user_left = max((mark_image_width - user_w - lib_w - split_width) // 2, 0)
        user_right = user_left + user_w
        user_top = max((mark_image_height - user_h) // 2, 0)
        mark_image[user_top:user_top+user_h, user_left:user_right] = user_image

        lib_left = (mark_image_width - user_right - lib_w) // 2 + user_right
        lib_top = max((mark_image_height - lib_h) // 2, 0)
        mark_image[lib_top:lib_top+lib_h, lib_left:lib_left+lib_w] = lib_image

        if cur_label != -1:
            cv2.putText(mark_image, str(cur_label),
                        (mark_image_width // 2 - 10, mark_image_height // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        5, COLORMAPS[cur_label], 3)
        while 1:
            cv2.imshow('mark', mark_image)
            key = cv2.waitKey(0)
            if key in (27, 48, 49, 50, 97, 100, 111):
                break

        if key in (48, 49, 50):
            result_dict[table_id] = key - 48
            idx += 1
        elif key == 97: idx -= 1
        elif key == 100: idx += 1
        elif key == 27: break
        elif key == 111:
            results = save_results(results, result_dict, 'test.csv')
        idx = max(0, idx)
        idx = min(test_length-1, idx)
    save_results(results, result_dict, 'test.csv')


def save_results(results, result_dict, result_path):
    lines = []
    for i in range(len(results)):
        user_id ,lib_id, test_key, label = results[i]
        table_id = f'{user_id},{lib_id}'
        if table_id not in result_dict:
            lines.append(f'{user_id},{lib_id},{test_key},{label}')
            continue
        new_label = result_dict[table_id]
        results[i][-1] = new_label
        lines.append(f'{user_id},{lib_id},{test_key},{new_label}')
    with open(result_path, 'w') as f:
        f.write('\n'.join(lines))
    return results




def compare_with_xlsx(xlsx_path_1, xlsx_path_2):
    data1 = pd.read_excel(xlsx_path_1)
    print(f'read {xlsx_path_1}, has {data1.shape[0]} rows, {data1.shape[1]} columns.')
    data2 = pd.read_excel(xlsx_path_2)
    print(f'read {xlsx_path_2}, has {data2.shape[0]} rows, {data2.shape[1]} columns.')


    for i in range(1, 1151):
        user_id = data1.iat[i, 0]
        lib_id = data1.iat[i, 1]
        table_id = f'{user_id},{lib_id}'

        user_id_ = data2.iat[i, 0]
        lib_id_ = data2.iat[i, 1]
        table_id_ = f'{user_id_},{lib_id_}'
        if table_id != table_id_:
            print(i)
            print(table_id_)
            break


def split_xlsx(ori_xlsx_path):
    data_all = pd.read_excel(ori_xlsx_path)
    test_groups = data_all.groupby('test')
    for key, data_frame in test_groups:
        sub_csv_path = f'test_{key}.csv'
        total_lines = []
        for i in range(data_frame.shape[0]):
            user_id = data_frame.iat[i, 0]
            lib_id = data_frame.iat[i, 1]
            label = int(data_frame.iat[i, 3]) if not pd.isna(data_frame.iat[i, 3]) else -1
            total_lines.append(f'{user_id},{lib_id},{key},{label}')
        with open(sub_csv_path, 'w') as f:
            f.write('\n'.join(total_lines))

def download_image_cache():
    table_dict = read_html_nodes('test.html')
    results = read_csv_results('test.csv')
    for user_id ,lib_id, test_idx, cur_label in tqdm(results):
        table_id = f'{user_id},{lib_id}'
        if table_id not in table_dict:
            continue
        user_path, lib_path = table_dict[table_id]
        load_image(user_path)
        load_image(lib_path)

if __name__ == '__main__':
    if not os.path.exists(IMAGE_CACHE_DIR):
        os.mkdir(IMAGE_CACHE_DIR)

    download_image_cache()
    # split_xlsx('test.xlsx')
    # mark()
    # compare_with_xlsx('jiu.xlsx', 'test_1.xlsx')
    # while 1:
    #     cv2.imshow('', np.zeros((100, 100)))
    #     print(cv2.waitKey())




