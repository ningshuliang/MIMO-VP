import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Models import get_model
from Moving_mnist_dataset.moving_mnist import MovingMNIST
from skimage.metrics import structural_similarity as ssim
import os
import cv2
import preprocess
import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='Moving_mnist_dataset', help='folder for dataset')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
parser.add_argument('--checkpoint_path', type=str, default='mnist.pth', help='folder for dataset')
parser.add_argument('--lr', type=float, default=0.0005, help='learning_rate')
parser.add_argument('--n_epochs', type=int, default=1500, help='nb of epochs')
parser.add_argument('--print_every', type=int, default=1, help='')
parser.add_argument('--eval_every', type=int, default=10, help='')
parser.add_argument('--save_dir', type=str, default='checkpoints')
parser.add_argument('--gen_frm_dir', type=str, default='results_mnist')
parser.add_argument('--patch_size', type=int, default=2)

#
parser.add_argument('-d_model', type=int, default=128)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-heads', type=int, default=8)
parser.add_argument('-dropout', type=int, default=0)


parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

args = parser.parse_args()

mm = MovingMNIST(root=args.root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
train_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                           num_workers=4)

mm = MovingMNIST(root=args.root, is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])
test_loader = torch.utils.data.DataLoader(dataset=mm, batch_size=16, shuffle=False, drop_last=True,
                                          num_workers=4)


def train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion1, criterion2):
 
    # input_tensor : torch.Size([batch_size, input_length, 1, 64, 64])
    encoder_optimizer.zero_grad()
    output_image = encoder(input_tensor)
    target_tensor = torch.cat([target_tensor] * 10, dim=2)
    loss = 10 * criterion1(output_image, target_tensor) + criterion2(output_image, target_tensor)
    loss.backward()
    encoder_optimizer.step()
    return loss.item() / target_tensor.size(1)


def trainIters(encoder, n_epochs, print_every, eval_every):
    train_losses = []

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler_enc = ReduceLROnPlateau(encoder_optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    criterion1 = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    criterion2 = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    itr = 0
    for epoch in range(0, n_epochs):
        t0 = time.time()
        loss_epoch = 0
        for i, out in enumerate(train_loader, 0):
            itr += 1
            # print(itr)
            # input_batch =  torch.Size([8, 20, 1, 64, 64])
            input_tensor = out[1].to(device)
            input_tensor = preprocess.reshape_patch(input_tensor, args.patch_size)
            target_tensor = out[2].to(device)
            target_tensor = preprocess.reshape_patch(target_tensor, args.patch_size)
            loss = train_on_batch(input_tensor, target_tensor, encoder, encoder_optimizer, criterion1, criterion2)
            loss_epoch += loss

        train_losses.append(loss_epoch)
        if (epoch + 1) % print_every == 0:
            print('epoch ', epoch, ' loss ', loss_epoch, ' epoch time ', time.time() - t0)

        if (epoch + 1) % eval_every == 0:
            mse, mae, ssim = evaluate(encoder, test_loader)
            scheduler_enc.step(mse)
            stats = {}
            stats['net_param'] = encoder.state_dict()
            save_dir = os.path.join(args.save_dir, 'epoch-' + str(epoch))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint_path = os.path.join(save_dir, 'model.ckpt' + '-' + str(epoch))
            torch.save(stats, checkpoint_path)
    return train_losses


def evaluate(encoder, loader):
    total_mse, total_mae, total_ssim = 0,0,0
    encoder.eval()

    with torch.no_grad():
        for id, out in enumerate(loader, 0):
            # input_batch = torch.Size([8, 20, 1, 64, 64])
            input_tensor = out[1].to(device)
            input_tensor = preprocess.reshape_patch(input_tensor, args.patch_size)
            target_tensor = out[2].to(device)

            input_length = input_tensor.size(1)
            target_length = target_tensor.size(1)
     
            predictions = encoder(input_tensor)[:,:,-4:,]
            predictions = preprocess.reshape_patch_back(predictions, args.patch_size)
            predictions1 = predictions
            predictions= predictions.cpu().numpy()

            input_tensor = preprocess.reshape_patch_back(input_tensor, args.patch_size)
            input = input_tensor.cpu().numpy()
            target = target_tensor.cpu().numpy()
        

            # save prediction examples
            # if id < 200:
            #     path = os.path.join(args.gen_frm_dir, str(id))
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #     for i in range(10):
            #         name = 'gt' + str(i + 1) + '.png'
            #         # name = str(i + 1).zfill(2) + '.png'
            #         file_name = os.path.join(path, name)
            #         img_gt = np.uint8(input[0, i, :, :, :] * 255)
            #         img_gt = np.transpose(img_gt, [1, 2, 0])
            #         cv2.imwrite(file_name, img_gt)

            #     for i in range(10):
            #         name = 'gt' + str(i + 11) + '.png'
            #         # name = str(i + 11).zfill(2) + '.png'
            #         file_name = os.path.join(path, name)
            #         img_gt = np.uint8(target[0, i, :, :, :] * 255)
            #         img_gt = np.transpose(img_gt, [1, 2, 0])
            #         cv2.imwrite(file_name, img_gt)

            #     for i in range(10):
            #         name = 'pd' + str(i + 11) + '.png'
            #         # name = str(i + 21).zfill(2) + '.png'
            #         file_name = os.path.join(path, name)
            #         img_pd = predictions[0, i, :, :, :]
            #         img_pd = np.maximum(img_pd, 0)
            #         img_pd = np.minimum(img_pd, 1)
            #         img_pd = np.uint8(img_pd * 255)
            #         img_pd = np.transpose(img_pd, [1, 2, 0])
            #         cv2.imwrite(file_name, img_pd)



            mse_batch = np.mean((predictions - target) ** 2, axis=(0, 1, 2)).sum()
            mae_batch = np.mean(np.abs(predictions - target), axis=(0, 1, 2)).sum()
            total_mse += mse_batch
            total_mae += mae_batch

            for a in range(0, target.shape[0]):
                for b in range(0, target.shape[1]):
                    total_ssim += ssim(target[a, b, 0,], predictions[a, b, 0,]) / (target.shape[0] * target.shape[1])


    print('eval mse ', total_mse / len(loader), ' eval mae ', total_mae / len(loader), ' eval ssim ',
          total_ssim / len(loader))
    return total_mse / len(loader), total_mae / len(loader), total_ssim / len(loader)

print('BEGIN TRAIN')

model = get_model(args).to(device)

if args.checkpoint_path != '':
    print('load model:', args.checkpoint_path)
    stats = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(stats['net_param'])
    # plot_losses = trainIters(model, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every)
    mse, mae, ssim = evaluate(model, test_loader)
else:
    plot_losses = trainIters(model, args.n_epochs, print_every=args.print_every, eval_every=args.eval_every)
