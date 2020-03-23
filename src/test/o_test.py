import torch
import net.checknet as nets
import tool.utils as utils
from PIL import Image
import time
#import cv2

if __name__ == '__main__':
    #net
    pnet = nets.P_Net(istraining=False)
    rnet = nets.R_Net(istraining=False)
    onet = nets.O_Net(istraining=False, island=True)
    if torch.cuda.is_available():
        pnet = pnet.cuda()
        rnet = rnet.cuda()
        onet = onet.cuda()
    pnet.eval()
    rnet.eval()
    onet.eval()
    pnet.load_state_dict(torch.load(r'D:\Python\face_recognise\MTCNN\params\p_params.pkl')) # 导入训练参数
    rnet.load_state_dict(torch.load(r'D:\Python\face_recognise\MTCNN\params\r_params.pkl'))  # 导入训练参数
    onet.load_state_dict(torch.load(r'D:\Python\face_recognise\MTCNN\params\o_params.pkl'))  # 导入训练参数

    # 输入图片
    img = Image.open("333333.jpg")
    total_start_t = time.time()
    # P网络
    start_time = time.time()
    pboxs = utils.PnetDetect(pnet, img, imgshow=False)
    pnet_end_time = time.time()
    # R网络
    rboxs = utils.RnetDetect(rnet, img, pboxs, imgshow=False)
    rnet_end_time = time.time()
    # O网络
    oboxs = utils.OnetDetect(onet, img, rboxs, imgshow=True, show_conf=False)

    total_end_t = time.time()
    print("total_time:", total_end_t - total_start_t
          , "pnet_time:", pnet_end_time - start_time
          , "rnet_time:", rnet_end_time - pnet_end_time
          , "onet_time:", total_end_t - rnet_end_time
          )

