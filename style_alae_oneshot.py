# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.utils.data

import torch.optim
from torch import mm
from torch.optim.optimizer import Optimizer, required

import torch.nn.functional as F

from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
import lreq
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from idinvert_pytorch.models.perceptual_model_copy import PerceptualModel
import pytorch_ssim
from LBFGS import LBFGS, FullBatchLBFGS
from custom_adam import LREQAdam

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
lreq.use_implicit_lreq.set(True)

rnd = np.random.RandomState(5)


indices = [0, 1, 2, 3, 4, 10, 11, 17, 19]

labels = ["gender",
          "smile",
          "attractive",
          "wavy-hair",
          "young",
          "big lips",
          "big nose",
          "chubby",
          "glasses",
          ]

recon_loss = []
percep_loss = []
net_loss = []

def append_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im


def sample(cfg, logger):
    torch.cuda.set_device(0)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
    model.cuda(0)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()

    decoder_optimizer = LREQAdam([
        {'params': decoder.parameters()},
        {'params': mapping_fl.parameters()}
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    model.train()
    layer_count = cfg.MODEL.LAYER_COUNT

    F_Percep = PerceptualModel(min_val=-1.0, max_val=1.0)

    ssim_loss = pytorch_ssim.SSIM()

    attribute_values = [0.0 for i in indices]
    W = [torch.tensor(np.load("principal_directions/direction_%d.npy" % i), dtype=torch.float32) for i in indices]
    W_copy = W.copy()

    def encode(x):
        Z, _ = model.encode(x, layer_count - 1, 1)
        Z = Z.repeat(1, model.mapping_fl.num_layers, 1)
        return Z

    def decode(x):
        layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        return model.decoder(x, layer_count - 1, 1, noise=True)

    def preprocess(image):
        image = image.astype(np.float32)
        image = image / 255.0 * (2.0) + (-1.0)
        image = image.astype(np.float32).transpose(2, 0, 1)
        return image
    
    def preprocess2(image):
        image = image.astype(np.float32)
        image = cv2.resize(image, (1024,1024), cv2.INTER_AREA)
        image = image.astype(np.float32).transpose(2, 0, 1)
        return image

    def postprocess(x_rec):
        with torch.no_grad():
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()

    def _get_tensor_value(tensor):
        return tensor.cpu().detach().numpy()

    def get_init_code(image):
        img = image
        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]

        needed_resolution = model.decoder.layer_to_resolution[-1]
        while x.shape[2] > needed_resolution:
            x = F.avg_pool2d(x, 2, 2)
        if x.shape[2] != needed_resolution:
            x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))

        img_src = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()

        latents_original = encode(x[None, ...].cuda())
        latents = latents_original[0, 0].clone()

        z = _get_tensor_value(latents_original)
        return z.astype(np.float32)

    def mask_and_replace(target, context, center_x=512, center_y=512, crop_x=512, crop_y=512,):
        image_shape = (1024, 1024, 3)
        mask = np.zeros(image_shape, dtype=np.float32)
        xx = center_x - crop_x // 2
        yy = center_y - crop_y // 2
        mask[yy:yy + crop_y, xx:xx + crop_x, :] = 1.0

        target = target.astype(np.float32) / 255.0 * (2.0) + (-1.0)
        context = context.astype(np.float32) / 255.0 * (2.0) + (-1.0)

        x = target * mask + context * (1 - mask)

        return (x+1.0) / 2.0 * (255.0)

    def invert(image, np_img, ph1_iter=0, ph2_iter=0, rec=1.0, per=5e-5, reg=2.0, lr=1e-2):
        reconstruction_loss_weight=rec
        perceptual_loss_weight=per
        regularization_loss_weight=reg
        x = image[np.newaxis]
        x = torch.from_numpy(x.astype(np.float32))
        x = x.to('cuda')
        x.requires_grad = False

        init_z = get_init_code(np_img)
        z = torch.Tensor(init_z)
        z.requires_grad = False

        learning_rate=lr

        viz_results = []
        viz_results.append(np_img)

        x_init_inv = decode(z)
        viz_results.append(postprocess(x_init_inv))

        pbar = tqdm(range(1, ph1_iter+ph2_iter+1), leave=True)

        # Optimizing decoder
        model.train()
        
        grad_vals = []

        for step in pbar:
            loss = 0.0
            x_rec = decode(z)
            model.train()
            log_message = ""

            # Reconstruction loss.
            if reconstruction_loss_weight:
                loss_pix = torch.mean((x - x_rec) ** 2)
                loss = loss + loss_pix * reconstruction_loss_weight
                log_message = f'PixLoss: {_get_tensor_value(loss_pix):.3f}'

            # Modified Perceptual loss.
            if perceptual_loss_weight:
                x_feat_1,x_feat_2,x_feat_3,x_feat_4 = F_Percep.net(x)
                x_rec_feat_1,x_rec_feat_2,x_rec_feat_3,x_rec_feat_4 = F_Percep.net(x_rec)

                loss_feat_1 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_1, target = x_feat_1, reduction='mean')
                loss_feat_2 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_2, target = x_feat_2, reduction='mean')
                loss_feat_3 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_3, target = x_feat_3, reduction='mean')
                loss_feat_4 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_4, target = x_feat_4, reduction='mean')
                
                loss_feat = loss_feat_1+loss_feat_2+loss_feat_3+loss_feat_4

                loss = loss + loss_feat * perceptual_loss_weight
                log_message += f', PercepLoss: {_get_tensor_value(loss_feat):.3f}'

            # Regularization loss.
            if regularization_loss_weight:
                z_rec = encode(x_rec[None, ...][0].cuda())
                loss_reg = torch.mean((z - z_rec) ** 2)
                loss = loss + loss_reg * regularization_loss_weight
                log_message += f', ReguLoss: {_get_tensor_value(loss_reg):.3f}'

            log_message += f', Loss: {_get_tensor_value(loss):.3f}'
            pbar.set_description_str(log_message)
            
            decoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()

            if step == ph1_iter+ph2_iter:
                viz_results.append(postprocess(x_rec))

        return z, viz_results

    def invert_random_latent(image, np_img, ph1_iter=0, ph2_iter=0, rec=1.0, per=5e-5, reg=2.0, lr=1e-2):

        reconstruction_loss_weight=rec
        perceptual_loss_weight=per
        regularization_loss_weight=reg
        x = image[np.newaxis]
        x = torch.from_numpy(x.astype(np.float32))
        x = x.to('cuda')
        x.requires_grad = False

        random_latent = np.array([rnd.uniform(0, 1, (18,512))])
        z = torch.Tensor(random_latent)
        z.requires_grad = False

        learning_rate=lr

        viz_results = []
        viz_results.append(np_img)

        x_init_inv = decode(z)
        viz_results.append(postprocess(x_init_inv))

        pbar = tqdm(range(1, ph1_iter+ph2_iter+1), leave=True)

        # Optimizing decoder
        model.train()
        
        grad_vals = []

        for step in pbar:
            loss = 0.0
            x_rec = decode(z)
            model.train()
            log_message = ""

            # Reconstruction loss.
            if reconstruction_loss_weight:
                loss_pix = torch.mean((x - x_rec) ** 2)
                loss = loss + loss_pix * reconstruction_loss_weight
                log_message = f'PixLoss: {_get_tensor_value(loss_pix):.3f}'

            # Modified Perceptual loss.
            if perceptual_loss_weight:
                x_feat_1,x_feat_2,x_feat_3,x_feat_4 = F_Percep.net(x)
                x_rec_feat_1,x_rec_feat_2,x_rec_feat_3,x_rec_feat_4 = F_Percep.net(x_rec)

                loss_feat_1 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_1, target = x_feat_1, reduction='mean')
                loss_feat_2 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_2, target = x_feat_2, reduction='mean')
                loss_feat_3 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_3, target = x_feat_3, reduction='mean')
                loss_feat_4 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_4, target = x_feat_4, reduction='mean')
                
                loss_feat = loss_feat_1+loss_feat_2+loss_feat_3+loss_feat_4

                loss = loss + loss_feat * perceptual_loss_weight
                log_message += f', PercepLoss: {_get_tensor_value(loss_feat):.3f}'

            # Regularization loss.
            if regularization_loss_weight:
                z_rec = encode(x_rec[None, ...][0].cuda())
                loss_reg = torch.mean((z - z_rec) ** 2)
                loss = loss + loss_reg * regularization_loss_weight
                log_message += f', ReguLoss: {_get_tensor_value(loss_reg):.3f}'

            log_message += f', Loss: {_get_tensor_value(loss):.3f}'
            pbar.set_description_str(log_message)

            
            decoder_optimizer.zero_grad()
            loss.backward()
            decoder_optimizer.step()

            if step == ph1_iter+ph2_iter:
                viz_results.append(postprocess(x_rec))
        
        return z, viz_results

    def invert_mod(image, np_img, context_img, ph1_iter=0, ph2_iter=0, rec=1.0, per=5e-5, reg=2.0, lr=1e-2):
        reconstruction_loss_weight=rec
        perceptual_loss_weight=per
        regularization_loss_weight=reg

        x = image[np.newaxis]
        x = torch.from_numpy(x.astype(np.float32))
        x = x.to('cuda')
        x.requires_grad = False

        context_x = context_img[np.newaxis]
        context_x = torch.from_numpy(context_x.astype(np.float32))
        context_x = context_x.to('cuda')
        context_x.requires_grad = False

        init_z = get_init_code(np_img)
        z = torch.Tensor(init_z)
        z.requires_grad = True

        learning_rate=lr

        optimizer = torch.optim.Adam([z], lr=learning_rate)

        viz_results = []
        viz_results.append(np_img)

        x_init_inv = decode(z)
        viz_results.append(postprocess(x_init_inv))

        pbar = tqdm(range(1, ph1_iter+ph2_iter+1), leave=True)

        # Optimizing decoder
        model.train()
        
        grad_vals = []

        for step in pbar:
            loss = 0.0
            x_rec = decode(z)
            model.train()

            # Reconstruction loss.
            loss_pix = torch.mean((context_x - x_rec) ** 2)
            loss = loss + loss_pix * reconstruction_loss_weight
            log_message = f'PixLoss: {_get_tensor_value(loss_pix):.3f}'

            # SSIM loss
            # loss_ss = ssim_loss(x, x_rec)
            # loss = loss + loss_ss * 1.0
            # log_message = f'SSIMLoss: {_get_tensor_value(loss_ss):.3f}'

            # Perceptual loss.
            # if perceptual_loss_weight:
            #     x_feat = F_Percep.net(x)
            #     x_rec_feat = F_Percep.net(x_rec)
            #     print(x_feat.shape)
            #     print(x_rec_feat.shape)
            #     exit(0)
            #     loss_feat = torch.mean((x_feat - x_rec_feat) ** 2)
            #     loss = loss + loss_feat * perceptual_loss_weight
            #     log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

            # Modified Perceptual loss.
            if perceptual_loss_weight:
                x_feat_1,x_feat_2,x_feat_3,x_feat_4 = F_Percep.net(context_x)
                x_rec_feat_1,x_rec_feat_2,x_rec_feat_3,x_rec_feat_4 = F_Percep.net(x_rec)

                loss_feat_1 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_1, target = x_feat_1, reduction='mean')
                loss_feat_2 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_2, target = x_feat_2, reduction='mean')
                loss_feat_3 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_3, target = x_feat_3, reduction='mean')
                loss_feat_4 = torch.nn.functional.smooth_l1_loss(input = x_rec_feat_4, target = x_feat_4, reduction='mean')
                
                loss_feat = loss_feat_1+loss_feat_2+loss_feat_3+loss_feat_4

                loss = loss + loss_feat * perceptual_loss_weight
                log_message += f', PercepLoss: {_get_tensor_value(loss_feat):.3f}'

            # Regularization loss.
            if regularization_loss_weight:
                z_rec = encode(x_rec[None, ...][0].cuda())
                loss_reg = torch.mean((z - z_rec) ** 2)
                loss = loss + loss_reg * regularization_loss_weight
                log_message += f', ReguLoss: {_get_tensor_value(loss_reg):.3f}'

            log_message += f', Loss: {_get_tensor_value(loss):.3f}'
            pbar.set_description_str(log_message)

            optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            decoder_optimizer.step()

            if step == ph1_iter+ph2_iter:
                viz_results.append(postprocess(x_rec))
            
            if step%20 == 0:
                viz_results.append(postprocess(x_rec))
        
        return z, viz_results

    def loadNext():
        img = np.asarray(Image.open(path + '/' + paths[0]))
        if len(paths) == 0:
            paths.extend(paths_backup)

        if img.shape[2] == 4:
            img = img[:, :, :3]
        im = img.transpose((2, 0, 1))
        x = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
        if x.shape[0] == 4:
            x = x[:3]

        needed_resolution = model.decoder.layer_to_resolution[-1]
        while x.shape[2] > needed_resolution:
            x = F.avg_pool2d(x, 2, 2)
        if x.shape[2] != needed_resolution:
            x = F.adaptive_avg_pool2d(x, (needed_resolution, needed_resolution))

        img_src = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()

        latents_original = encode(x[None, ...].cuda())
        latents = latents_original[0, 0].clone()
        latents -= model.dlatent_avg.buff.data[0]
        latents_prev = latents.clone()

        for v, w in zip(attribute_values, W):
            v = (latents * w).sum()

        for v, w in zip(attribute_values, W):
            latents = latents - v * w
        
        return latents, latents_original, img_src

    def loadRandom():
        latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
        lat = torch.tensor(latents).float().cuda()
        dlat = mapping_fl(lat)
        layer_idx = torch.arange(2 * layer_count)[np.newaxis, :, np.newaxis]
        ones = torch.ones(layer_idx.shape, dtype=torch.float32)
        coefs = torch.where(layer_idx < model.truncation_cutoff, ones, ones)
        dlat = torch.lerp(model.dlatent_avg.buff.data, dlat, coefs)
        x = decode(dlat)[0]
        img_src = ((x * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()
        latents_original = dlat
        latents = latents_original[0, 0].clone()
        latents -= model.dlatent_avg.buff.data[0]

        for v, w in zip(attribute_values, W):
            v.value = (latents * w).sum()

        for v, w in zip(attribute_values, W):
            latents = latents - v.value * w

        return latents, latents_original, img_src

    def update_image(w, latents_original):
        with torch.no_grad():
            w = w + model.dlatent_avg.buff.data[0]
            w = w[None, None, ...].repeat(1, model.mapping_fl.num_layers, 1)

            layer_idx = torch.arange(model.mapping_fl.num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = (7 + 1) * 2
            mixing_cutoff = cur_layers
            styles = torch.where(layer_idx < mixing_cutoff, w, latents_original)

            x_rec = decode(styles)
            resultsample = ((x_rec * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

    def generate(number_of_imgs):
        font = ImageFont.truetype("MS Reference Sans Serif.ttf", 80)
        for k in range(number_of_imgs):
            mod_images = []
            permute = [2,1,0]
            latents, latents_original, img_src = loadNext()
            orig = Image.fromarray(img_src.astype('uint8'), 'RGB')
            orig = ImageOps.expand(orig,border=100,fill='black')
            draw1 = ImageDraw.Draw(orig)
            draw1.text((520, 5),"Input",(0,255,0),font=font)
            mod_images.append(orig)
            for val in range(-20,21,5):
                attribute_values[0] = float(val)
                latents, latents_original, img_src = loadNext()
                attribute_values[0] = float(0.0)
                im = update_image(latents, latents_original)
                im_numpy = im.numpy()
                if(val == 0):
                    font1 = ImageFont.truetype("MS Reference Sans Serif.ttf", 70)
                    orig = Image.fromarray(im_numpy.astype('uint8'), 'RGB')
                    orig = ImageOps.expand(orig,border=100,fill='black')
                    draw1 = ImageDraw.Draw(orig)
                    draw1.text((120, 0),"<--(-ve)|(Gender)|(+ve)-->",(0,255,0),font=font1)
                    draw1.text((350, 70),"Reconstruction",(0,255,0),font=font1)
                    
                    mod_images.append(orig)
                else:
                    mod_images.append(Image.fromarray(im_numpy.astype('uint8'), 'RGB'))
            
            im_name = paths[0]
            paths.pop(0)
            final_img = mod_images[0]

            combo_1 = append_images(mod_images, direction='horizontal')
            combo_1.save("experiments/gender_variation_ffhq_weights/"+im_name)
            for i in range(1,9):
                final_img = np.concatenate((final_img, mod_images[i]),axis=1)
            cv2.imwrite("experiments/age_variation/"+im_name, final_img)
            cv2.imwrite("experiments/identity_experiments_ffhq_data/original/"+im_name, mod_images[0])
            cv2.imwrite("experiments/samples_varying_attr/"+im_name[:-4]+"_1.png", final_img)

    def change_attributes(latents_original,orig_inverted_image,experiment_no,mod_attr,neg_limit=-20,pos_limit=21,step=10):
        font = ImageFont.truetype("MS Reference Sans Serif.ttf", 80)
        latents = latents_original[0, 0].clone()
        latents -= model.dlatent_avg.buff.data[0]

        mod_images = []

        for val in range(neg_limit,pos_limit,step):
            latents = latents_original[0, 0].clone()
            latents -= model.dlatent_avg.buff.data[0]

            attribute_values[mod_attr] = float(val)
            for v, w in zip(attribute_values, W):
                v = (latents * w).sum()
            for v, w in zip(attribute_values, W):
                latents = latents - v * w

            attribute_values[mod_attr] = float(0.0)
            im = update_image(latents, latents_original)
            im_numpy = im.numpy()
            if(val == 0):
                    orig = Image.fromarray(im_numpy.astype('uint8'), 'RGB')
                    mod_images.append(orig)
            else:
                mod_images.append(Image.fromarray(im_numpy.astype('uint8'), 'RGB'))

        return mod_images

    ffhq_image_list = [
        "diverse_ffhq_samples/00043.png",
        "diverse_ffhq_samples/25750.png",
        "diverse_ffhq_samples/57531.png",
        "diverse_ffhq_samples/57535.png"
    ]
    celeba_image_list = [
        "dataset_samples/faces/realign1024x1024/00043.png",
        "dataset_samples/faces/realign1024x1024/00018.png",
        "dataset_samples/faces/realign1024x1024/00026.png",
        "dataset_samples/faces/realign1024x1024/00104.png"
    ]
    eifel_image_list = [
        "eifel_images/eifel.jpg",
        "eifel_images/eifel2.jpg"
    ]
    car_image_list = [
        "car_images/car1.png",
        "car_images/car2.png"
    ]
    our_img_list = [
        "our_face_images/p.png",
        "our_face_images/r.png",
        "our_face_images/s.png"
    ]
    asian_celebs_list = [
        "asian_celebs/t2.png",
        "asian_celebs/t5.png",
        "asian_celebs/t7.png"
    ]

    akshay_kumar = "/mnt/hdd1/ravikiran/ALAE/qualitative_samples/akshay-kumar_01.png"
    irfan_khan = "/mnt/hdd1/ravikiran/ALAE/qualitative_samples/irfan-khan_01.png"
    amber_heard = "/mnt/hdd1/ravikiran/ALAE/qualitative_samples/amber-heard_01.png"
    emma_stone = "/mnt/hdd1/ravikiran/ALAE/qualitative_samples/emma-stone_01.png"
    yami_gautam = "/mnt/hdd1/ravikiran/ALAE/qualitative_samples/yami-gautam_01.png"

    custom_images_folder_path = "/mnt/hdd1/ravikiran/ALAE/quantitative_samples_1000/" # 1000 Quantitative
    # custom_images_folder_path = "/mnt/hdd1/ravikiran/ALAE/qualitative_samples/" # Qualitative
    custom_images_container = [custom_images_folder_path+i for i in os.listdir(custom_images_folder_path)]
    ffhq_image_list = [
        "diverse_ffhq_samples/00145.png",
        "diverse_ffhq_samples/00425.png",
        "diverse_ffhq_samples/25542.png",
        "diverse_ffhq_samples/25847.png"
    ]
    celeba_image_list = [
        "dataset_samples/faces/realign1024x1024/00029.png",
        "dataset_samples/faces/realign1024x1024/00053.png",
        "dataset_samples/faces/realign1024x1024/00136.png",
        "dataset_samples/faces/realign1024x1024/00100.png"
    ]

    def orig_run(experiment_no,ph1_iter=30,ph2_iter=10,rec=1.0,per=0.05,reg=1.0,lr=1e-2):
        save_dir = "/mnt/hdd1/ravikiran/ALAE/alae_celeba_non_face_inversions/car1.jpg"

        test_img = Image.open(car_image_list[0])
        # test_img = test_img.resize((256,256))
        test_img = test_img.resize((1024,1024))
        test_img = np.asarray(test_img)

        # test_img = np.asarray(Image.open(path)) # Use this for ffhq weights
        a,b = invert(image=preprocess(test_img),np_img=test_img,ph1_iter=ph1_iter,ph2_iter=ph2_iter,rec=rec,per=per,reg=reg,lr=lr)
        
        im1 = Image.fromarray(b[0].astype('uint8'), 'RGB')
        im2 = Image.fromarray(b[1].astype('uint8'), 'RGB')
        im3 = Image.fromarray(b[-1].astype('uint8'), 'RGB')

        out = append_images([im1,im2,im3],direction='horizontal')
        out.save(save_dir)

    def run(experiment_no,ph1_iter=30,ph2_iter=10,rec=1.0,per=0.05,reg=1.0,lr=1e-2):

        ffhq_inversions = []
        celeba_inversions = []

        for path in custom_images_container:

            save_dir = "/mnt/hdd1/ravikiran/ALAE/alae_quantitative_ffhq_weights/ours/"
            # save_dir = "/mnt/hdd1/ravikiran/ALAE/samples_17Jul_gender/"
            # save_folder = path.split("/")[-1].split("_")[0]
            save_folder = path.split("/")[-1].split(".")[0]

            print(save_folder,"##",count)

            final_save_folder_smile = save_dir+"smile/"+save_folder
            final_save_folder_age = save_dir+"age/"+save_folder
            final_save_folder_gender = save_dir+"gender/"+save_folder
            final_save_folder_wavyhair = save_dir+"wavyhair/"+save_folder
            final_save_folder_biglips = save_dir+"biglips/"+save_folder
            final_save_folder_bignose = save_dir+"bignose/"+save_folder
            final_save_folder_attractive = save_dir+"attractive/"+save_folder
            final_save_folder_chubby = save_dir+"chubby/"+save_folder
            final_save_folder_glasses = save_dir+"glasses/"+save_folder

            if not os.path.exists(final_save_folder_smile):
                os.makedirs(final_save_folder_smile)
            if not os.path.exists(final_save_folder_age):
                os.makedirs(final_save_folder_age)
            if not os.path.exists(final_save_folder_gender):
                os.makedirs(final_save_folder_gender)
            if not os.path.exists(final_save_folder_wavyhair):
                os.makedirs(final_save_folder_wavyhair)
            if not os.path.exists(final_save_folder_biglips):
                os.makedirs(final_save_folder_biglips)
            if not os.path.exists(final_save_folder_bignose):
                os.makedirs(final_save_folder_bignose)
            if not os.path.exists(final_save_folder_attractive):
                os.makedirs(final_save_folder_attractive)
            if not os.path.exists(final_save_folder_chubby):
                os.makedirs(final_save_folder_chubby)
            if not os.path.exists(final_save_folder_glasses):
                os.makedirs(final_save_folder_glasses)
            
            #######################################################
            # Use these 3 lines for celeba-hq256 weights
            # test_img = Image.open(path)
            # test_img = test_img.resize((256,256))
            # test_img = test_img.resize((1024,1024))
            # test_img = np.asarray(test_img)
            #######################################################

            # reloading weights for each sample
            extra_checkpoint_data = checkpointer.load()

            try:
                test_img = np.asarray(Image.open(path)) # Use this for ffhq weights
                a,b = invert(image=preprocess(test_img),np_img=test_img,ph1_iter=ph1_iter,ph2_iter=ph2_iter,rec=rec,per=per,reg=reg,lr=lr)
                # a,b = invert_random_latent(image=preprocess(test_img),np_img=test_img,ph1_iter=ph1_iter,ph2_iter=ph2_iter,rec=rec,per=per,reg=reg,lr=lr)
                # a,b = invert(image=preprocess(test_img),np_img=preprocess2(test_img),ph1_iter=ph1_iter,ph2_iter=ph2_iter,rec=rec,per=per,reg=reg,lr=lr)
                # continue
                im1 = Image.fromarray(b[0].astype('uint8'), 'RGB')
                im2 = Image.fromarray(b[1].astype('uint8'), 'RGB')
                im3 = Image.fromarray(b[-1].astype('uint8'), 'RGB')

                mod_list = change_attributes(a,im3,experiment_no,mod_attr=0,neg_limit=-30,pos_limit=31,step=5)
                im1.save(final_save_folder_gender+"/base_img.jpg")
                im2.save(final_save_folder_gender+"/alae_inverted_original.jpg")
                im3.save(final_save_folder_gender+"/inverted_img.jpg")
                im_no = 0
                for image in mod_list:
                    image.save(final_save_folder_gender+"/img_{}.jpg".format(im_no))
                    im_no+=1

            except: pass
   

    def semantic_diffusion_run(experiment_no,ph1_iter=30,ph2_iter=10,rec=1.0,per=0.05,reg=1.0,lr=1e-2):
        
        context_img = np.asarray(Image.open(amber_heard).resize((1024,1024)))
        target_img = np.asarray(Image.open(irfan_khan).resize((1024,1024)))

        test_img = mask_and_replace(context=context_img, target=target_img)
        save_dir = "/mnt/hdd1/ravikiran/ALAE/alae_qualitative_semantic_diffusion/amber_irfan_cropped_optim.jpg"

        a,b = invert_mod(image=preprocess(test_img),np_img=test_img,context_img=preprocess(test_img),ph1_iter=ph1_iter,ph2_iter=ph2_iter,rec=rec,per=per,reg=reg,lr=lr)

        full_list = [Image.fromarray(i.astype('uint8'), 'RGB') for i in b]
    
        out = append_images(full_list,direction='horizontal')
        out.save(save_dir)
    
    run(experiment_no='26m',ph1_iter=200,ph2_iter=0,rec=50.0,per=0.00005,reg=0,lr=5e-3)

    # run(experiment_no='24e',ph1_iter=200,ph2_iter=0,rec=1.0,per=0.00005,reg=2.0,lr=5e-3)
    # run(experiment_no='24f',ph1_iter=200,ph2_iter=0,rec=1.0,per=0.0005,reg=2.0,lr=5e-3)
    # run(experiment_no='24g',ph1_iter=200,ph2_iter=0,rec=5.0,per=0.00005,reg=2.0,lr=5e-3)
    # run(experiment_no='24h',ph1_iter=200,ph2_iter=0,rec=5.0,per=0.0005,reg=2.0,lr=5e-3)
    # run(experiment_no='24i',ph1_iter=200,ph2_iter=0,rec=1.0,per=0.00005,reg=2.0,lr=7e-3)
    # run(experiment_no='24j',ph1_iter=200,ph2_iter=0,rec=1.0,per=0.0005,reg=2.0,lr=7e-3)
    # run(experiment_no='24k',ph1_iter=200,ph2_iter=0,rec=5.0,per=0.00005,reg=2.0,lr=7e-3)
    # run(experiment_no='24l',ph1_iter=200,ph2_iter=0,rec=5.0,per=0.0005,reg=2.0,lr=7e-3)

    # run(experiment_no='25e',ph1_iter=300,ph2_iter=0,rec=1.0,per=0.00005,reg=2.0,lr=5e-3)
    # run(experiment_no='25f',ph1_iter=300,ph2_iter=0,rec=1.0,per=0.0005,reg=2.0,lr=5e-3)
    # run(experiment_no='25g',ph1_iter=300,ph2_iter=0,rec=5.0,per=0.00005,reg=2.0,lr=5e-3)
    # run(experiment_no='25h',ph1_iter=300,ph2_iter=0,rec=5.0,per=0.0005,reg=2.0,lr=5e-3)
    # run(experiment_no='25i',ph1_iter=300,ph2_iter=0,rec=1.0,per=0.00005,reg=2.0,lr=7e-3)
    # run(experiment_no='25j',ph1_iter=300,ph2_iter=0,rec=1.0,per=0.0005,reg=2.0,lr=7e-3)
    # run(experiment_no='25k',ph1_iter=300,ph2_iter=0,rec=5.0,per=0.00005,reg=2.0,lr=7e-3)
    # run(experiment_no='25l',ph1_iter=300,ph2_iter=0,rec=5.0,per=0.0005,reg=2.0,lr=7e-3)

    # run(experiment_no='26e',ph1_iter=200,ph2_iter=0,rec=10.0,per=0.00005,reg=2.0,lr=5e-3)
    # run(experiment_no='26f',ph1_iter=200,ph2_iter=0,rec=10.0,per=0.0005,reg=2.0,lr=5e-3)
    # run(experiment_no='26g',ph1_iter=200,ph2_iter=0,rec=10.0,per=0.00005,reg=2.0,lr=5e-3)
    # run(experiment_no='26h',ph1_iter=200,ph2_iter=0,rec=10.0,per=0.0005,reg=2.0,lr=5e-3)
    # run(experiment_no='26i',ph1_iter=200,ph2_iter=0,rec=10.0,per=0.00005,reg=2.0,lr=7e-3)
    # run(experiment_no='26j',ph1_iter=200,ph2_iter=0,rec=10.0,per=0.0005,reg=2.0,lr=7e-3)
    # run(experiment_no='26k',ph1_iter=200,ph2_iter=0,rec=10.0,per=0.00005,reg=2.0,lr=7e-3)
    # run(experiment_no='26l',ph1_iter=200,ph2_iter=0,rec=10.0,per=0.0005,reg=2.0,lr=7e-3)


    # run(experiment_no='26l',ph1_iter=200,ph2_iter=0,rec=50.0,per=0.0005,reg=2.0,lr=5e-3)
    # run(experiment_no='26m',ph1_iter=200,ph2_iter=0,rec=50.0,per=0.00005,reg=2.0,lr=5e-3)
    # run(experiment_no='26n',ph1_iter=200,ph2_iter=0,rec=50.0,per=0.0005,reg=2.0,lr=7e-3)
    # run(experiment_no='26o',ph1_iter=200,ph2_iter=0,rec=50.0,per=0.00005,reg=2.0,lr=7e-3)


    # run(experiment_no='15i',ph1_iter=150,ph2_iter=0,rec=0.0,per=1.0,reg=0.0,lr=1e-2)
    # run(experiment_no='15j',ph1_iter=300,ph2_iter=0,rec=10.0,per=1.0,reg=0.0,lr=1e-3)
    # run(experiment_no='15k',ph1_iter=400,ph2_iter=0,rec=10.0,per=1.0,reg=0.0,lr=1e-3)
    # run(experiment_no='15l',ph1_iter=300,ph2_iter=0,rec=100.0,per=5.0,reg=1.0,lr=1e-3)
    # run(experiment_no='15m',ph1_iter=300,ph2_iter=0,rec=100.0,per=0.5,reg=0.0,lr=1e-3)


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-interactive', default_config='configs/ffhq.yaml',
        world_size=gpu_count, write_log=False)
    # run(sample, get_cfg_defaults(), description='ALAE-interactive', default_config='configs/celeba-hq256.yaml',
    #     world_size=gpu_count, write_log=False)

