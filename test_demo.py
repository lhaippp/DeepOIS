import os
import cv2
import math
import torch
import shutil
import imageio
import logging
import resource
import argparse

import numpy as np

import stn
import utils

import model.unet as unet
import data_loader.optical_field_data_loader as optical_field_data_loader

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file',
                    default='best',
                    help="name of the file in --model_dir \
                     containing weights to load")
parser.add_argument('--test_id', type=int)


def computeKeypoint(img_1, img_2):
    MIN_MATCH_COUNT = 10

    img_gray_1, img_gray_2 = img_1, img_2

    # hessian = 1000
    surf = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = surf.detectAndCompute(img_gray_1, None)
    kp2, des2 = surf.detectAndCompute(img_gray_2, None)

    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)

    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    [good.append(m) for m, n in matches if m.distance < 0.6 * n.distance]

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.array([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.array([kp2[m.trainIdx].pt for m in good])
    else:
        raise ValueError("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

    return src_pts, dst_pts


def dlt_spatial_transform(flow, data_img):
    img_indices = stn.get_grid(batch_size=data_img.shape[0], H=data_img.shape[2], W=data_img.shape[3], start=0)
    vgrid = img_indices[:, :2, ...]
    grid_warp = vgrid - flow
    warp_imgs = stn.transformer(data_img, grid_warp)
    return warp_imgs


def denormalize_optical_field(optical_field, mean, std):
    mean_x, mean_y = mean[0], mean[1]
    std_x, std_y = std[0], std[1]

    optical_field[:, 0, ...] = optical_field[:, 0, ...] * std_x + mean_x
    optical_field[:, 1, ...] = optical_field[:, 1, ...] * std_y + mean_y
    return optical_field


def geometricDistance(flow, data_img, data):
    img_indices = stn.get_grid(batch_size=data_img.shape[0], H=data_img.shape[2], W=data_img.shape[3], start=0)
    # print(img_indices.shape)
    vgrid = img_indices[:, :2, ...]
    grid_warp = vgrid + flow

    errors = 0
    points = 6

    for i in range(points):
        points_LR = data[i]

        x1, y1, x2, y2 = points_LR[0][0], points_LR[0][1], points_LR[1][0], points_LR[1][1]

        if isinstance(x1, np.float64):
            x1_proj = (grid_warp[:, 0, math.ceil(y1), math.ceil(x1)].detach().cpu().numpy() + \
                       grid_warp[:, 0, math.floor(y1), math.floor(x1)].detach().cpu().numpy()) / 2
            y1_proj = (grid_warp[:, 1, math.ceil(y1), math.ceil(x1)].detach().cpu().numpy() + \
                       grid_warp[:, 1, math.floor(y1), math.floor(x1)].detach().cpu().numpy()) / 2
        else:
            x1_proj = grid_warp[:, 0, int(y1), int(x1)].detach().cpu().numpy()
            y1_proj = grid_warp[:, 1, int(y1), int(x1)].detach().cpu().numpy()

        error = np.sqrt(np.square(x1_proj - x2) + np.square(y1_proj - y2))
        errors += error

    err_avg = errors / points
    return err_avg


def test(model, dataloader, metrics, params, gif_path, verbose=True):
    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    cnt = 0

    # 整体测试
    pair_list = list(open('Test_List.txt'))
    npy_path = os.path.join('Test_Set')

    re_1_cnt = 0
    re_2_cnt = 0
    mf_1_cnt = 0
    mf_2_cnt = 0
    ll_1_cnt = 0
    ll_2_cnt = 0
    lt_1_cnt = 0
    lt_2_cnt = 0

    video_names = []
    # data mining to filter bad cases
    skip_frames = [
        '369', '448', '449', '475', '498', '162', '163', '172', '173', '187', '189', '214', '215', '216', '227', '230', '252', '256', '259',
        '278', '280', '315', '280', '282', '314', '316', '358', '359', '376', '379', '416', '419', '441', '442', '363', '398', '619', '638',
        '701', '703', '712', '785', '789', '790', '664', '666', '670', '673', '675', '676', '688', '689', '716', '725', '727', '737', '755',
        '777', '778', '791', '808', '809', '822', '834', '835', '836', '839', '840', '843', '846', '850', '2067', '2074', '2075', '2087',
        '2095', '2127', '2128', '2144', '2168', '2179', '2182', '2183', '2206', '2246', '2256', '2259', '2260', '510', '517', '524', '557',
        '581', '582', '608', '611', '612', '623', '667', '713', '765', '766', '792', '794'
    ]

    # compute metrics over the dataset
    for data_batch, frame_batch in dataloader:

        with torch.no_grad():
            data_img = frame_batch[:, :3]
            labels_img = frame_batch[:, 3:]

            if params.cuda:
                data_batch, data_img, labels_img = data_batch.cuda(), data_img.cuda(), labels_img.cuda()

            # compute model output
            output = model(data_batch)

            output = output[:, :, :270, :360]

            data_batch = denormalize_optical_field(data_batch, (0.17564012110233307, -0.01618252880871296),
                                                   (5.118218421936035, 4.462287902832031))

            output = denormalize_optical_field(output, (0.17564012110233307, -0.01618252880871296), (5.118218421936035, 4.462287902832031))

            # compute homography and compute psnr between nn_warp_img and match_img
            nn_warp_img = dlt_spatial_transform(output, data_img)

            # compute all metrics on this batch
            summary_batch = dict()

            summary_batch['psnr_nn'] = metrics['psnr'](nn_warp_img, labels_img)

            try:
                # compute geometry distance between predicted field and groundtruth field
                img_pair = pair_list[cnt]
                cnt += 1

                pari_id = img_pair.split(' ')
                npy_name = pari_id[0].split('/')[0] + '_' + pari_id[0].split('/')[1] + '_' + pari_id[1].split(
                    '/')[0] + '_' + pari_id[1].split('/')[1][:-1] + '.npy'
                video_name = pari_id[0].split('/')[0]

                video_names.append(video_name)

                idx = pari_id[0].split('/')[1].split('-')[1][:-4]

                npy_id = os.path.join(npy_path, npy_name)
                point_dic = np.load(npy_id, allow_pickle=True)
                data = point_dic.item()

                data = data['matche_pts']

                orig_dis = geometricDistance(0, data_img, data)
                gyro_dis = geometricDistance(data_batch, data_img, data)
                nn_dis = geometricDistance(output, data_img, data)

                if gyro_dis > 10:
                    print('detect bad case')
                    continue

                if idx in skip_frames:
                    continue

                if video_name == '阳台RE':
                    re_1_cnt += 1
                    summary_batch['RE_distance_nn'.format(video_name)] = nn_dis
                elif video_name == '路口MF':
                    mf_1_cnt += 1
                    summary_batch['MF_distance_nn'.format(video_name)] = nn_dis
                elif video_name == '夜景LL':
                    ll_1_cnt += 1
                    summary_batch['LL_distance_nn'.format(video_name)] = nn_dis
                elif video_name == '天空LT':
                    lt_1_cnt += 1
                    summary_batch['LT_distance_nn'.format(video_name)] = nn_dis
                elif video_name == '行人MF':
                    mf_2_cnt += 1
                    summary_batch['MF_distance_nn'.format(video_name)] = nn_dis
                elif video_name == '湖畔夜景LL':
                    ll_2_cnt += 1
                    summary_batch['LL_distance_nn'.format(video_name)] = nn_dis
                elif video_name == '广场RE':
                    re_2_cnt += 1
                    summary_batch['RE_distance_nn'.format(video_name)] = nn_dis

            except Exception as e:
                print(e)
                break
            summ.append(summary_batch)

    # compute mean of all metrics in summary
    beg, end = 0, re_1_cnt
    metrics_mean = {metric: np.mean([x[metric] for x in summ[beg:end]]) for metric in summ[beg]}
    print('beg: {} - end: {}'.format(beg, end))

    beg += re_1_cnt
    end += mf_1_cnt
    metrics_mean.update({metric: np.mean([x[metric] for x in summ[beg:end]]) for metric in summ[beg]})
    print('beg: {} - end: {}'.format(beg, end))

    beg += mf_1_cnt
    end += ll_1_cnt
    metrics_mean.update({metric: np.mean([x[metric] for x in summ[beg:end]]) for metric in summ[beg]})
    print('beg: {} - end: {}'.format(beg, end))

    beg += ll_1_cnt
    end += lt_1_cnt
    metrics_mean.update({metric: np.mean([x[metric] for x in summ[beg:end]]) for metric in summ[beg]})
    print('beg: {} - end: {}'.format(beg, end))

    beg += lt_1_cnt
    end += mf_2_cnt
    metrics_mean.update({metric: np.mean([x[metric] for x in summ[beg:end]]) for metric in summ[beg]})
    print('beg: {} - end: {}'.format(beg, end))

    beg += mf_2_cnt
    end += ll_2_cnt
    metrics_mean.update({metric: np.mean([x[metric] for x in summ[beg:end]]) for metric in summ[beg]})
    print('beg: {} - end: {}'.format(beg, end))

    beg += ll_2_cnt
    end += re_2_cnt
    metrics_mean.update({metric: np.mean([x[metric] for x in summ[beg:end]]) for metric in summ[beg]})
    print('beg: {} - end: {}'.format(beg, end))
    print('dataset length: {}'.format(len(summ)))

    _avg = np.sum([v for k, v in metrics_mean.items() if k != "psnr_nn"]) / 4
    metrics_mean["avg_distance"] = _avg

    if verbose:
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()  # use GPU is available

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = optical_field_data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model and optimizer
    model = unet.UNet(params)

    # fetch loss function and metrics
    if params.loss_fn == 'l1':
        loss_fn = unet.loss_fn_l1loss
    elif params.loss_fn == 'l2':
        loss_fn = unet.loss_fn_l2loss

    metrics = unet.metrics
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
        model = model.cuda()
    elif params.cuda:
        model = model.cuda()

    logging.info("Starting testing")

    # Reload weights from the saved file
    print('load state_dict from {}'.format(os.path.join(args.model_dir, args.restore_file + '.pth.tar')))
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test(model, test_dl, metrics, params, gif_path=None)
