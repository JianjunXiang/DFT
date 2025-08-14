import os
import argparse
import yaml
import torch
import random
import numpy as np
from model.SwinTiny_Student import SWTIQA
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy import stats
from model.swin_transformer import swin_tiny_patch4_window7

def distillation_loss2(source, target, mask):
    b, c, _ = source.shape
    h = int(source.size()[2] ** 0.5)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    ini_loss = loss.view(b, c, h, h) * mask.unsqueeze(1)
    return ini_loss.mean()

def eval_epoch(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in test_loader:
            pred = 0
            for i in range(config['num_avg_val']):
                x_d = data['d_img_org'][:, i].to(config['device'])
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).to(config['device'])
                _, pred_s = net(x_d)
                pred += pred_s
            pred /= config['num_avg_val']
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s = stats.spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0]
        rho_p = stats.pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0]
        rho_k = stats.stats.kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0]
        rho_r = np.sqrt(((np.squeeze(pred_epoch) - np.squeeze(labels_epoch)) ** 2).mean())

        return np.mean(losses), rho_s, rho_p, rho_k, rho_r, np.squeeze(pred_epoch), np.squeeze(labels_epoch)

def train_epoch(config, epoch, T_net, S_net, criterion_q, optimizer, scheduler, train_loader):
    losses = []
    T_net.eval()
    S_net.train(True)
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []

    for T_data, S_data in train_loader:
        x_s = S_data['d_img_org'].to(config['device'])
        x_t = T_data['d_img_org'].to(config['device'])
        x_j0 = S_data['img_vm1'].to(config['device'])
        x_j1 = S_data['img_vm2'].to(config['device'])
        x_j2 = S_data['img_vm3'].to(config['device'])
        qlabels = S_data['score']
        qlabels = torch.squeeze(qlabels.type(torch.FloatTensor)).to(config['device'])
        with torch.no_grad():
            _, g_f = T_net.forward_features(x_t)

        f_f, pred_s = S_net(x_s)

        optimizer.zero_grad()
        loss = criterion_q(torch.squeeze(pred_s), qlabels) + (distillation_loss2(g_f[0], f_f[0], x_j0) + distillation_loss2(g_f[1], f_f[1], x_j1) + distillation_loss2(g_f[2], f_f[2], x_j2) + distillation_loss2(g_f[3], f_f[3], x_j2)) * config['lambda_k']
        losses.append(loss.item())
        #
        loss.backward()
        optimizer.step()
        scheduler.step()

        # save results in one epoch
        pred_batch_numpy = pred_s.data.cpu().numpy()
        labels_batch_numpy = qlabels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    rho_s = stats.spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0]
    rho_p = stats.pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0]

    ret_loss = np.mean(losses)
    print('Train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def RandShuffle(config):
    train_size = config['train_size']

    if config['type'] == 'IQA':
        if config['dataset_name'] == 'LIVE-2D':
            scenes = list(range(29))
        elif config['dataset_name'] == 'TID2013':
            scenes = list(range(25))
        elif config['dataset_name'] == 'CSIQ':
            scenes = list(range(30))
        elif config['dataset_name'] == 'KADID':
            scenes = list(range(81))
        elif config['dataset_name'] == 'CLIVE':
            scenes = list(range(1162))
        elif config['dataset_name'] == 'KONIQ':
            scenes = list(range(10073))
        elif config['dataset_name'] == 'FLIVE':
            scenes = list(range(39807))
        elif config['dataset_name'] == 'SPAQ':
            scenes = list(range(11125))
        else:
            pass
    else:
        scenes = config['dataset_name']

    n_scenes = len(scenes)
    n_train_scenes = int(np.floor(n_scenes * train_size))

    np.random.shuffle(scenes)
    train_scene_list = scenes[:n_train_scenes]
    test_scene_list = scenes[n_train_scenes:]

    return train_scene_list, test_scene_list

def main():
    parser = argparse.ArgumentParser(description='Train Swin-Transformer for BIQA in LIVE dataset.')
    parser.add_argument(
        # '-o', '--opt', default='/home/xiangjj/code/my_code/TIP2024/finetune/TIP2024/main_results/overall_map/option/train_on_koniq.yaml',
        '-o', '--opt', default='./option/train_on_koniq.yaml',
        help='Configuration file'
    )
    args = parser.parse_args()
    with open(args.opt, 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['device'] = device

    setup_seed(config['seed'])
    if not os.path.exists(config['ckpt_path']):
        os.makedirs(config['ckpt_path'])

    plcc_all = np.zeros((1, 10), dtype=np.float)
    srcc_all = np.zeros((1, 10), dtype=np.float)
    krcc_all = np.zeros((1, 10), dtype=np.float)
    rmse_all = np.zeros((1, 10), dtype=np.float)

    if config['dataset_name'] == 'LIVE-2D':
        from data.live_m import IQADataset
        dis_train_path = config['dataset_path']
        dis_val_path = config['dataset_path']
        label_train_path = config['dataset_label']
        label_val_path = config['dataset_label']
        Dataset = IQADataset
    elif config['dataset_name'] == 'TID2013':
        from data.tid2013_m import IQADataset
        dis_train_path = config['dataset_path']
        dis_val_path = config['dataset_path']
        label_train_path = config['dataset_label']
        label_val_path = config['dataset_label']
        Dataset = IQADataset
    elif config['dataset_name'] == 'CSIQ':
        from data.csiq_m import IQADataset
        dis_train_path = config['dataset_path']
        dis_val_path = config['dataset_path']
        label_train_path = config['dataset_label']
        label_val_path = config['dataset_label']
        Dataset = IQADataset
    elif config['dataset_name'] == 'KADID':
        from data.kadid_m import IQADataset
        dis_train_path = config['dataset_path']
        dis_val_path = config['dataset_path']
        label_train_path = config['dataset_label']
        label_val_path = config['dataset_label']
        Dataset = IQADataset
    elif config['dataset_name'] == 'CLIVE':
        from data.clive_m import IQADataset
        dis_train_path = config['dataset_path']
        dis_val_path = config['dataset_path']
        label_train_path = config['dataset_label']
        label_val_path = config['dataset_label']
        Dataset = IQADataset
    elif config['dataset_name'] == 'KONIQ':
        from data.koniq_m import IQADataset
        dis_train_path = config['dataset_path']
        dis_val_path = config['dataset_path']
        label_train_path = config['dataset_label']
        label_val_path = config['dataset_label']
        Dataset = IQADataset
    elif config['dataset_name'] == 'FLIVE':
        from data.flive_m import IQADataset
        dis_train_path = config['dataset_path']
        dis_val_path = config['dataset_path']
        label_train_path = config['dataset_label']
        label_val_path = config['dataset_label']
        Dataset = IQADataset
    elif config['dataset_name'] == 'SPAQ':
        from data.spaq_m import IQADataset
        dis_train_path = config['dataset_path']
        dis_val_path = config['dataset_path']
        label_train_path = config['dataset_label']
        label_val_path = config['dataset_label']
        Dataset = IQADataset
    else:
        pass

    for split_num in range(config['splits_num']):
        # train-test set splits
        train_list, val_list = RandShuffle(config)
        teach_extractor = swin_tiny_patch4_window7(pretrained=True).to(config['device'])
        for p in teach_extractor.parameters():
            p.requires_grad = False

        student_model = SWTIQA(config).to(config['device'])
        if config['resume'] == True:
            state_dict = torch.load(config['load_ckpt'])
            model_weight = state_dict['state_dict']
            student_model.load_state_dict(model_weight)

        # set criterion
        criterion_q = torch.nn.MSELoss()

        # set optimizer
        optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=config['lr'],
            weight_decay=config['wd'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])

        # set train and test datasets
        teacher_transforms = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        student_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=config['prob_aug']),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        train_dataset = Dataset(
            config=config,
            db_path=dis_train_path,
            txt_file_name=label_train_path,
            T_transform=teacher_transforms,
            S_transform=student_transforms,
            train_mode=True,
            scene_list=train_list,
            keep_ratio=config['train_size']
        )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  num_workers=config['num_worker'],
                                  drop_last=True,
                                  shuffle=True)
        val_dataset = Dataset(
            config=config,
            db_path=dis_val_path,
            txt_file_name=label_val_path,
            T_transform=None,
            S_transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),]),
            train_mode=False,
            scene_list=val_list,
            keep_ratio=None
        )
        val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'],
                                num_workers=config['num_worker'], shuffle=False)
        print('number of train scenes: {}'.format(len(train_dataset)))
        print('number of val scenes: {}'.format(len(val_dataset)))

        best_srcc = 0.0
        best_plcc = 0.0
        best_krcc = 0.0
        best_rmse = 0.0
        main_score = -2.0
        old_save_name = None
        for epoch in range(config['n_epoch']):
            print('Running training epoch {}'.format(epoch + 1))
            loss_val, trho_s, trho_p = train_epoch(config, epoch, teach_extractor, student_model, criterion_q, optimizer, scheduler, train_loader)

            loss, vrho_s, vrho_p, vrho_k, vrho_r, pr_results, gt_results = eval_epoch(config, epoch, student_model, criterion_q, val_loader)

            if vrho_s + vrho_p > main_score:
                main_score = vrho_s + vrho_p
                best_srcc = vrho_s
                best_plcc = vrho_p
                best_krcc = vrho_k
                best_rmse = vrho_r
                print('*Val epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== KRCC:{:.4} ===== RMSE:{:.4}'.format(epoch + 1, np.mean(loss), vrho_s,
                                                                                 vrho_p, vrho_k, vrho_r))
                # save_weights
                if epoch > 0:
                    if os.path.exists(old_save_name):
                        os.remove(old_save_name)
                model_name = 'dataset_{}_split_{}_epoch_{}.pth.tar'.format(config['dataset_name'], split_num, epoch + 1)
                model_save_path = os.path.join(config['ckpt_path'], model_name)
                now_state = {'best_epoch': epoch + 1,
                             'pr_results': pr_results,
                             'gt_results': gt_results,
                             'best_plcc': best_plcc,
                             'best_srcc': best_srcc,
                             'best_krcc': best_krcc,
                             'best_rmse': best_rmse}

                torch.save(now_state, model_save_path)
                old_save_name = model_save_path
            else:
                print(
                    'Val epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4} ===== KRCC:{:.4} ===== RMSE:{:.4}'.format(
                        epoch + 1, np.mean(loss), vrho_s,
                        vrho_p, vrho_k, vrho_r))
        plcc_all[0][split_num] = best_plcc
        srcc_all[0][split_num] = best_srcc
        krcc_all[0][split_num] = best_krcc
        rmse_all[0][split_num] = best_rmse
    return plcc_all, srcc_all, krcc_all, rmse_all

if __name__ == '__main__':
    plcc_all, srcc_all, krcc_all, rmse_all = main()
    print('Final Results! SRCC:{:.4} ===== PLCC:{:.4} ===== KRCC:{:.4} ===== RMSE:{:.4}'.format(np.median(srcc_all),
                                                                                                np.median(plcc_all),
                                                                                                np.median(krcc_all),
                                                                                                np.median(rmse_all)))