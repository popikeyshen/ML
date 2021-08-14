import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

from analyze import*

def init():
    img_size = (608, 608) 
    source, weights, half =  opt.source, opt.weights, opt.half


    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)


    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)


    # Eval mode
    model.to(device).eval()


    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()



    #dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    return  device, model,  names, colors



### letterbox tested confidence
### without 0.43
### with    0.43
### letterbox is usless if size is allready resized


def letterbox(img, new_shape=(608, 608), color=(128, 128, 128), auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):

    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)


    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def convert(img):
        # Padded resize
        #img = letterbox(img, new_shape=608)[0]
        #print("letterbox",img.shape)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img, dtype=np.float32 )  # uint8 to fp16/fp32  #  np.float32 / np.float16
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        return img


def detect(save_txt=False, save_img=False):

    device, model,  names, colors = init()

    cap1 = cv2.VideoCapture("12.mp4")
    #cap1 = cv2.VideoCapture("/media/athena/2Tb/test/20m11.mp4")
    #cap1 = cv2.VideoCapture("/media/athena/2Tb/test/M4_dist.mp4")
    #cap1 = cv2.VideoCapture("/media/athena/2Tb/test/25_1.MP4")
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 100)

    # Run inference
    t0 = time.time()
    while 1:
        t = time.time()

        ret, img = cap1.read()
        #img = cv2.imread("4.jpg")

        im0 = cv2.resize(img, (608,  608), interpolation=cv2.INTER_LINEAR)

        img = convert(im0)

        # Convert to torch
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
               img = img.unsqueeze(0)





        ### convert done

        # detect
        pred = model(img)[0]
        if opt.half:
            pred = pred.float()
            print("we use float")


        analyze(model,img)


        # Process detections
        #for i, det in enumerate(pred):  # detections per image
        #print(pred)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)



        # Process detections
        for i, det in enumerate(pred):  # detections per image

            #print(det)

            s = '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])


            print('%sDone. (%.3fs)' % (s, time.time() - t))


            cv2.imshow("out", im0)
            cv2.waitKey(0) 




    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--cfg', type=str, default='a1.cfg', help='*.cfg path')
    parser.add_argument('--cfg', type=str, default='a1.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='a1.weights', help='path to weights file')
    parser.add_argument('--names', type=str, default='37.names', help='*.names path')
    #parser.add_argument('--weights', type=str, default='a1.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='12.mp4', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.0, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()

    print(opt)

    with torch.no_grad():
        detect()
