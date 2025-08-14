import csv
import os
import torch
import numpy as np
from skimage.measure import block_reduce
from PIL import Image
import scipy.io as io

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def cal_maps(vmap):
    vmaps = []
    blocks = [28, 14, 7]
    H, W = vmap.shape
    for N in blocks:
        block_h = H // N
        block_w = W // N
        crop_h = block_h * N
        crop_w = block_w * N
        cmap = vmap[:crop_h, :crop_w]
        result = block_reduce(cmap, (block_h, block_w), np.mean)
        vmaps.append(result / np.mean(result))
    return vmaps

class IQADataset(torch.utils.data.Dataset):

    def __init__(self, config, db_path, txt_file_name, T_transform, S_transform, train_mode, scene_list,
                 keep_ratio=0.8):
        super(IQADataset, self).__init__()

        self.config = config
        self.db_path = db_path
        self.txt_file_name = txt_file_name
        self.T_transform = T_transform
        self.S_transform = S_transform
        self.train_mode = train_mode
        self.scene_list = scene_list
        self.train_size = keep_ratio

        self.data_dict = IQADatalist(
            db_path=self.db_path,
            txt_file_name=self.txt_file_name,
            train_mode=self.train_mode,
            scene_list=self.scene_list,
            train_size=self.train_size
        ).load_data_dict()

        self.n_images = len(self.data_dict['d_img_list'])

    def __len__(self):

        return self.n_images

    def get_spatial_fragments(self,
            img,
            fragments_h=7,
            fragments_w=7,
            fsize_h=32,
            fsize_w=32,
    ):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w
        ## video: [C,T,H,W]
        ## situation for images
        _, res_h, res_w = img.shape[-3:]

        size = size_h, size_w

        ## make sure that sampling will not run out of the picture
        hgrids = torch.LongTensor(
            [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
        )
        wgrids = torch.LongTensor(
            [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
        )
        hlength, wlength = res_h // fragments_h, res_w // fragments_w

        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids))
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids))).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids))
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids))).int()

        target_img = torch.zeros(img.shape[0:1] + size).to(img.device)
        # target_videos = []

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w

                h_so, h_eo = hs + rnd_h[i][j], hs + rnd_h[i][j] + fsize_h
                w_so, w_eo = ws + rnd_w[i][j], ws + rnd_w[i][j] + fsize_w
                target_img[:, h_s:h_e, w_s:w_e] = img[:, h_so:h_eo, w_so:w_eo]
        return target_img

    def __getitem__(self, idx):

        # d_img_org: H x W x C
        d_img_name = self.data_dict['d_img_list'][idx]
        j_img_name = self.data_dict['j_img_list'][idx]
        j_img_org = io.loadmat(j_img_name)
        vm_map = j_img_org['jnd_map']
        vm_maps = cal_maps(vm_map)
        d_img_org = Image.open(os.path.join(self.db_path, d_img_name)).convert('RGB')
        score = self.data_dict['score_list'][idx]
        if self.train_mode:
            t_img_org = self.T_transform(d_img_org)
            s_img_org = self.S_transform(d_img_org)
            if self.config['fragments']:
                s_img_org = self.get_spatial_fragments(s_img_org)
            return {'d_img_org': t_img_org, 'score': score}, {'d_img_org': s_img_org, 'score': score, 'img_vm1': vm_maps[0], 'img_vm2': vm_maps[1], 'img_vm3': vm_maps[2]}
        else:
            s_img_org = self.S_transform(d_img_org)
            if self.config['fragments']:
                s_imgs_org = []
                for i in range(self.config['num_avg_val']):
                    s_imgs_org.append(self.get_spatial_fragments(s_img_org))
                s_imgs_org = np.stack(s_imgs_org, 0)
                return {'d_img_org': s_imgs_org, 'score': score}

class IQADatalist():

    def __init__(self, db_path, txt_file_name, train_mode, scene_list, train_size=0.8):
        self.txt_file_name = txt_file_name
        self.train_mode = train_mode
        self.train_size = train_size
        self.scene_list = scene_list
        self.db_path = db_path

    def load_data_dict(self):

        d_img_list, score_list, j_img_list = [], [], []
        refpath, labels = [], []
        csv_file = os.path.join(self.txt_file_name)
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                refpath.append(row['image_name'])
                mos = np.array(float(row['MOS'])).astype(np.float32)
                labels.append(mos)

        index = self.scene_list

        for i, item in enumerate(index):
            d_img_list.append(os.path.join(self.db_path, 'images', refpath[item]))
            score_list.append(labels[item])
            j_img_list.append(os.path.join(self.db_path, 'JND_map', refpath[item][0:-4] + '.mat'))
        # reshape score_list (1xn -> nx1)
        score_list = np.array(score_list)
        score_list = self.normalization(score_list)
        score_list = score_list.astype('float').reshape(-1, 1)

        data_dict = {'d_img_list': d_img_list, 'score_list': score_list, 'j_img_list': j_img_list}

        return data_dict

    def normalization(self, data):

        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename