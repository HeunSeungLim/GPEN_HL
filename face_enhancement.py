'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import cv2
import time
import numpy as np
import __init_paths
from face_detect.retinaface_detection import RetinaFaceDetection
from face_parse.face_parsing import FaceParse
from face_model.face_gan import FaceGAN
from sr_model.real_esrnet import RealESRNet
from align_faces import warp_and_crop_face, get_reference_facial_points


from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet


def load_model():
    model = MobileNetV2_unet(None).cuda()
    state_dict = torch.load('./checkpoints/model.pt', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


class FaceEnhancement(object):
    def __init__(self, base_dir='./', in_size=512, out_size=None, model=None, use_sr=True, sr_model=None, sr_scale=2, channel_multiplier=2, narrow=1, key=None, device='cuda'):
        self.facedetector = RetinaFaceDetection(base_dir, device)
        self.facegan = FaceGAN(base_dir, in_size, out_size, model, channel_multiplier, narrow, key, device=device)
        # self.srmodel =  RealESRNet(base_dir, sr_model, sr_scale, device=device)
        self.faceparser = FaceParse(base_dir, device=device)
        self.use_sr = use_sr
        self.in_size = in_size
        self.out_size = in_size if out_size is None else out_size
        self.threshold = 0.9

        # the mask for pasting restored faces back
        self.mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(self.mask, (26, 26), (486, 486), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)
        self.mask = cv2.GaussianBlur(self.mask, (101, 101), 11)

        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")

        # get the reference 5 landmarks position in the crop settings
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = get_reference_facial_points(
                (self.in_size, self.in_size), inner_padding_factor, outer_padding, default_square)

    def mask_postprocess(self, mask, thres=20):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        mask = cv2.GaussianBlur(mask, (101, 101), 11)
        return mask.astype(np.float32)

    def process(self, img, aligned=False):
        orig_faces, enhanced_faces = [], []
        if aligned:
            ef = self.facegan.process(img)
            orig_faces.append(img)
            enhanced_faces.append(ef)

            if self.use_sr:
                ef = self.srmodel.process(ef)

            return ef, orig_faces, enhanced_faces

        if self.use_sr:
            img_sr = self.srmodel.process(img)
            if img_sr is not None:
                img = cv2.resize(img, img_sr.shape[:2][::-1])
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        self.face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device='cuda')


        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        # get face landmarks for each face
        self.face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
        # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
        # align and warp each face
        self.face_helper.align_warp_face()

        facebs, landms = self.facedetector.detect(img)
        
        height, width = img.shape[:2]
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)

        maskmask = np.zeros((height, width), dtype=np.float32)
        # full_img = np.zeros(img.shape, dtype=np.uint8)

        for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
            if faceb[4]<self.threshold: continue
            fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])

            facial5points = np.reshape(facial5points, (2, 5))

            of, tfm_inv = warp_and_crop_face(img, self.face_helper.all_landmarks_5[0], reference_pts=self.reference_5pts, crop_size=(self.in_size, self.in_size))
            of2, tfm_inv2 = warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.in_size, self.in_size))
            # enhance the face
            ef = self.facegan.process(of)

            orig_faces.append(of)
            enhanced_faces.append(ef)
            
            #tmp_mask = self.mask
            temp = np.uint8(np.clip((ef.astype(np.float32) / 255.) ** (1), 0, 1) * 255)

            cv2.imwrite('face.png', temp)



            tmp_mask = self.mask_postprocess(self.faceparser.process(temp)[0]/255.)

            # tmp_mask = self.mask_postprocess(self.faceparser.process(temp)[0] / 255.)

            cv2.imwrite('ef.png', np.uint8(ef))
            tmp_mask = cv2.resize(tmp_mask, (self.in_size, self.in_size))
            tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

            mask_model = load_model()
            temp_ef = cv2.resize(img, dsize=(224,224))
            temp_ef = temp_ef.astype(np.float32)/255.

            temp_ef = torch.from_numpy(temp_ef.transpose(2, 0, 1)).unsqueeze(0).cuda()
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            imgage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(imgage)
            torch_img = transform(pil_img)
            torch_img = torch_img.unsqueeze(0)
            torch_img = torch_img.cuda()

            logits = mask_model(torch_img)
            # mask = np.argmax(logits.data.cpu().numpy(), axis=1)
            cv2.imwrite('mask.png', np.uint8(
                np.argmax(logits.data.cpu().numpy().astype(np.float32), axis=1).transpose(1, 2, 0) * 255))
            maskmask = np.uint8(
                np.argmax(logits.data.cpu().numpy().astype(np.float32), axis=1).transpose(1, 2, 0) * 255)
            # cv2.imwrite('mask.png', np.uint8(logits.squeeze().data.cpu().numpy().transpose(1,2,0)))

            if min(fh, fw)<100: # gaussian filter for small faces
                ef = cv2.filter2D(ef, -1, self.kernel)
            
            if self.in_size!=self.out_size:
                ef = cv2.resize(ef, (self.in_size, self.in_size))
            tmp_img = cv2.warpAffine(ef, tfm_inv, (width, height), flags=3)

            mask = tmp_mask - full_mask
            full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
            full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]

        full_mask = full_mask[:, :, np.newaxis]
        maskmask = cv2.resize(maskmask, dsize=(width, height))
        maskmask = maskmask[..., np.newaxis]
        maskmask = np.concatenate((maskmask, maskmask, maskmask), axis=2)
        maskmask = maskmask * full_mask
        # if self.use_sr and img_sr is not None:
        #     img = cv2.convertScaleAbs(img_sr*(1-full_mask) + full_img*full_mask)
        # else:
        # if maskmask
        if np.max(maskmask) == 0:
            img = cv2.convertScaleAbs(img * (1 - full_mask) + full_img * full_mask)
        else:

            img = cv2.convertScaleAbs(img*(1-maskmask.astype(np.float32)/255) + full_img*maskmask.astype(np.float32)/255)

        return img, orig_faces, enhanced_faces
        
        
