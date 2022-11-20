
import os
import cv2
import glob
import time
import math
import argparse
import numpy as np
from PIL import Image, ImageDraw
import __init_paths
from face_enhancement_ori import FaceEnhancement
from face_colorization import FaceColorization
from face_inpainting import FaceInpainting
from segmentation2face import Segmentation2Face
import time
def brush_stroke_mask(img, color=(255,255,255)):
    min_num_vertex = 8
    max_num_vertex = 28
    mean_angle = 2*math.pi / 5
    angle_range = 2*math.pi / 15
    min_width = 12
    max_width = 80
    def generate_mask(H, W, img=None):
        average_radius = math.sqrt(H*H+W*W) / 8
        mask = Image.new('RGB', (W, H), 0)
        if img is not None: mask = img #Image.fromarray(img)

        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))

            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(
                    np.random.normal(loc=average_radius, scale=average_radius//2),
                    0, 2*average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))

            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=color, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width//2,
                              v[1] - width//2,
                              v[0] + width//2,
                              v[1] + width//2),
                             fill=color)

        return mask

    width, height = img.size
    mask = generate_mask(height, width, img)
    return mask

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='GPEN-BFR-512', help='GPEN model')
    parser.add_argument('--task', type=str, default='FaceEnhancement', help='task of GPEN model')
    parser.add_argument('--key', type=str, default=None, help='key of GPEN model')
    parser.add_argument('--in_size', type=int, default=512, help='in resolution of GPEN')
    parser.add_argument('--out_size', type=int, default=None, help='out resolution of GPEN')
    parser.add_argument('--channel_multiplier', type=int, default=2, help='channel multiplier of GPEN')
    parser.add_argument('--narrow', type=float, default=1, help='channel narrow scale')
    parser.add_argument('--use_sr', action='store_true', help='use sr or not')
    parser.add_argument('--use_cuda', default=True,  help='use cuda or not')
    parser.add_argument('--save_face', default=True, help='save face or not')
    parser.add_argument('--aligned', action='store_true', help='input are aligned faces or not')
    parser.add_argument('--sr_model', type=str, default='realesrnet', help='SR model')
    parser.add_argument('--sr_scale', type=int, default=2, help='SR scale')
    parser.add_argument('--indir', type=str, default='examples/TEMP', help='input folder')
    parser.add_argument('--outdir', type=str, default='results/TEMP_out', help='input folder')

    args = parser.parse_args()

    #model = {'name':'GPEN-BFR-512', 'size':512, 'channel_multiplier':2, 'narrow':1}
    #model = {'name':'GPEN-BFR-256', 'size':256, 'channel_multiplier':1, 'narrow':0.5}

    os.makedirs(args.outdir, exist_ok=True)

    if args.task == 'FaceEnhancement':
        processer = FaceEnhancement(in_size=args.in_size, model=args.model, use_sr=args.use_sr, channel_multiplier=args.channel_multiplier, narrow=args.narrow, key=args.key, device='cuda' if args.use_cuda else 'cpu')
    elif args.task == 'FaceColorization':
        processer = FaceColorization(in_size=args.in_size, model=args.model, device='cuda' if args.use_cuda else 'cpu')
    elif args.task == 'FaceInpainting':
        processer = FaceInpainting(in_size=args.in_size, model=args.model, device='cuda' if args.use_cuda else 'cpu')
    elif args.task == 'Segmentation2Face':
        processer = Segmentation2Face(in_size=args.in_size, model=args.model, is_norm=False, device='cuda' if args.use_cuda else 'cpu')


    files = sorted(glob.glob(os.path.join(args.indir, '*.*g')))

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    num = 0
    while  cv2.waitKey(33) < 0:
        ret, img = capture.read()

        img = cv2.resize(img, dsize=(0,0), fx=1/4, fy=1/4)
        img = cv2.resize(img, dsize=(0, 0), fx=4, fy=4)

        start = time.time()
        img_out, orig_faces, enhanced_faces = processer.process(img, aligned=args.aligned)
        end = time.time()

        cv2.imshow('camera', np.concatenate((img, img_out), 1))
        num = num + 1


    capture.release()
    cv2.destroyAllWindows()