#-*- coding:utf-8 -*-
# CTPN
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ctpn_pytorch.ctpn import config
from ctpn_pytorch.ctpn.ctpn import CTPN_Model
from ctpn_pytorch.ctpn.utils import gen_anchor, transform_bbox, clip_bbox, filter_bbox, nms, TextProposalConnectorOriented

# CRNN
import time
from torch.autograd import Variable
import crnn.lib.utils.utils as utils
import crnn.lib.models.crnn as crnn
import crnn.lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse

# 加载这个用以在图像上输出中文
from PIL import Image, ImageDraw, ImageFont

################ CTPN ############
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = './ctpn_pytorch/weights/ctpn.pth'
ctpn_model = CTPN_Model()
ctpn_model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
print('CTPN loaded !!!')

def get_text_boxes(image, display = True, prob_thresh = 0.5):
    ctpn_model.to(device)
    ctpn_model.eval()
    h, w= image.shape[:2]
    rescale_fac = max(h, w) / 1000
    if rescale_fac > 1.0:
        h = int(h / rescale_fac)
        w = int(w / rescale_fac)
        image = cv2.resize(image, (w,h))
        h, w = image.shape[:2]
    # image_c = image.copy()
    image = image.astype(np.float32) - config.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        cls, regr = ctpn_model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = transform_bbox(anchor, regr)
        bbox = clip_bbox(bbox, [h, w])

        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        keep_index = filter_bbox(select_anchor, 16)

        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])
        
        # if display:
        #     for i in text:
        #         s = str(round(i[-1] * 100, 2)) + '%'
        #         i = [int(j) for j in i]
        #         # 原来的代码是用线来画矩形
        #         # cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
        #         # cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
        #         # cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
        #         # cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 100, 255), 2)
        #         # cv2.putText(image_c, s, (i[0]+13, i[1]+13), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        #         # 直接画矩形
        #         cv2.rectangle(image_c, (i[0], i[1]), (i[6], i[7]), (0, 255, 0), 2)

        return text

def video_detect(source):
    cap = cv2.VideoCapture(source)
    while True:
        print("GET")
        ret, img = cap.read()
        text, out_img = get_text_boxes(img)
        print(text)
        cv2.imshow("capture", out_img)
        ret = cv2.waitKey(100)
        if ret == 27: # ASCII=27的字符终止 ESC
            break
            cv2.destroyAllWindows()

def ctpn_get_box(input_img):
    # print(input_img.shape)
    text = get_text_boxes(input_img)
    # 这里是对得到的框进行排序
    data = np.array(text)
    index = np.lexsort([data[:,0], data[:,1]])   
    boxes = data[index,:]
    text = list(boxes) 

    # # boxes的计数
    # count = 0
    # roi_list = []
    # for i in text:
    #     count += 1
    #     i = [int(j) for j in i]
    #     # 直接画矩形
    #     cv2.rectangle(input_img, (i[0], i[1]), (i[6], i[7]), (0, 255, 0), 2)
    #     # cv2.putText(input_img, str(count), (i[0]+13, i[1]+13), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    #     # 得到每个box区域, padding=5
    #     pad = 5
    #     img_roi = input_img[i[1]-pad:i[7]+pad, i[0]-pad:i[6]+pad]
    #     roi_list.append(img_roi)
    #     # cv2.imwrite('./output/result_{0}.jpg'.format(count), img_roi)
    return text  #, roi_list

########### CRNN ################
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='./crnn/lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='./ctpn_pytorch/images/test_1.png', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='./crnn/output/crnn.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, device):

    model.to(device)
    # img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    # print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    # print(sim_pred)
    return sim_pred


def image_ocr(input_img, model, config_crnn):

    # CTPN 得到文字框
    boxes = ctpn_get_box(input_img)    

    # 这里建立一个空白的图像以便输出ocr文字
    h, w, c = input_img.shape
    draw_text = np.ones((h, w, c), dtype='uint8') * 255
    img = Image.fromarray(draw_text)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./font/msyh.ttc", 20, encoding="utf-8")

    for i in boxes:
        i = [int(j) for j in i]
        # 直接画矩形
        cv2.rectangle(input_img, (i[0]-5, i[1]), (i[6]+20, i[7]), (0, 255, 0), 2)
        # 得到每个box区域, padding=5
        pad = 5
        img_roi = input_img[i[1]:i[7], i[0]-pad:i[6]+20]
        ocr_text = recognition(config_crnn, img_roi, model, device)
        print(ocr_text)
        draw.text((i[0], i[1]), ocr_text, fill=(0, 0, 0), font=font)
   
    show_img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # 这里是把输入与输出合成一张图像输出
    if h < w:
        dst = np.zeros((h*2, w, c), dtype='uint8')
        dst[:h, :w] = input_img
        dst[h:, :] = show_img
    elif h > w:
        dst = np.zeros((h, w*2, c), dtype='uint8')
        dst[:h, :w] = input_img
        dst[:, w:] = show_img
    
    # 输出图像，这样不会把原来的结果也覆盖了
    save_dir = './output/result/'
    images = os.listdir(save_dir)
    cv2.imwrite('./output/result/result_{0}.png'.format(len(images)), dst)


def video_ocr(source, model, config_crnn):

    cap = cv2.VideoCapture(source)
    while True:
        ret, img = cap.read()
        boxes = ctpn_get_box(img)
        # count = 0
        for i in boxes:
            # count += 1
            i = [int(j) for j in i]   
            # 直接画矩形
            cv2.rectangle(img, (i[0], i[1]), (i[6]+20, i[7]), (0, 255, 0), 2)       
            pad = 0
            img_roi = img[i[1]-pad:i[7]+pad, i[0]-pad:i[6]+pad]
            ocr_text = recognition(config_crnn, img_roi, model, device)
            print(ocr_text)
        cv2.imshow("capture", img)
        ret = cv2.waitKey(100)
        if ret == 27: # ASCII=27的字符终止 ESC
            break
            cv2.destroyAllWindows()

if __name__ == '__main__':

    # CRNN model load
    config_crnn, args = parse_arg()

    model = crnn.get_crnn(config_crnn)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    img_path = args.image_path
    # img_path = './ctpn_pytorch/images/img_1125.jpg'
    input_img = cv2.imread(img_path)

    # 对图像进行OCR识别，英文效果还是不错，中文稍差
    image_ocr(input_img, model, config_crnn)

    # 视频实时识别，识别率不是很好，可能是因为画面存在抖动
    # video_ocr(0, model, config_crnn)