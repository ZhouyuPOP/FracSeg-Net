import os
import time
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from Loss import MulticlassDiceLoss, SurfaceLoss
from utils import *
from pelvic_dataset import MyDataset
from tqdm import tqdm
from CSAUnet import CsaUnet
from tensorboardX import SummaryWriter

# initial ArgumentParser object
parser = argparse.ArgumentParser()
# add arguments
parser.add_argument('--batch_size', type=int, default=6, help='Batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=1e-2, help='Learning rate')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--num_classes', type=int,  default=2, help='Number of class')
parser.add_argument('--final_sigmoid_flag', action='store_true', help='Enable final sigmoid.')
parser.add_argument('--beta1', type=float,  default=0.5, help='Adam betas[0]')
parser.add_argument('--beta2', type=float,  default=0.999, help='Adam betas[1]')
parser.add_argument('--weight_decay', type=float,  default=0, help='L2 regularization weight.')
parser.add_argument('--save_root', type=str, default='./model/', help='RootPath to save the model')
parser.add_argument('--model_name', type=str, default='CsaUnet', help='Name of the model')
parser.add_argument('--loss_name', type=str, default='Dice+Surface', help='Name of the loss')
parser.add_argument('--extra_description', type=str, default='', help='Extra description')
# parses the command-line arguments and stores them in the 'ARGS' object.
ARGS = parser.parse_args()
PLATTE = np.arange(ARGS.num_classes)
SAVE_PATH = ARGS.save_root + ARGS.extra_description + ARGS.model_name + ARGS.loss_name + str(time.time())


def init_seeds(seed=2024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(num_classes=2):
    net = CsaUnet(1, num_classes, final_sigmoid_flag=ARGS.final_sigmoid_flag, init_channel_number=64)
    model = net.cuda()
    return model


def train():
    init_seeds()
    model = create_model(ARGS.num_classes)
    joint_transform = Compose([
        Std(dist_flag=True),
        RandomCrop([96, 128, 128], dist_flag=True),
        ToTensor()
    ])

    train_data_path = 'Path to Your Train Data Directory'
    train_dataset = MyDataset(train_data_path, joint_transform, cal_dist=True)
    train_dataloader = DataLoader(train_dataset, batch_size=ARGS.batch_size, shuffle=True, num_workers=0)
    writer = SummaryWriter(os.path.join(SAVE_PATH))

    best_dice = 0.
    iters = 0.
    max_epoch = 150
    dice_loss = MulticlassDiceLoss()
    edge_loss = SurfaceLoss(idc=ARGS.num_classes)
    optimizer = optim.Adam(model.parameters(),
                           lr=ARGS.base_lr, betas=(ARGS.beta1, ARGS.beta2),
                           weight_decay=ARGS.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=max_epoch)

    for epoch_index in tqdm(range(max_epoch), ncols=70):
        hip_dice = 0.
        model.train()
        alpha = np.clip(1.0 * (epoch_index + 1)/max_epoch, 0.01, 1)
        for batch_index, (data_batch, mask_batch) in enumerate(train_dataloader):
            dist_map_batch = data_batch[:, 1:, ...]
            data_batch = data_batch[:, 0:1, ...]
            data_batch, mask_batch, dist_map_batch = data_batch.cuda(), mask_batch.cuda(), dist_map_batch.cuda()
            pred_batch = model(data_batch)
            iters += ARGS.batch_size

            mask_batch_onehot = mask_to_onehot(mask_batch[:, np.newaxis, :, :, :], PLATTE, 1)
            dice_loss_value = dice_loss(pred_batch, mask_batch_onehot)
            edge_loss_value = edge_loss(pred_batch, dist_map_batch)
            loss = (1 - alpha) * dice_loss_value + alpha * edge_loss_value

            pred_batch = pred_batch.detach().cpu().numpy()
            mask_to_cal = mask_batch_onehot.detach().cpu().numpy()
            hip_dice += dice_ratio_channel_wise(pred_batch, mask_to_cal, 1)

            print(f'Epoch = {epoch_index} iters = {iters} Current_Loss = {loss.item()}')
            writer.add_scalar('train_main_loss', loss.item(), iters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        hip_dice_mean = hip_dice / len(train_dataloader)
        print(f'Epoch = {epoch_index} ,Mean Hip Dice = {hip_dice_mean}')
        writer.add_scalar('Hip_dice', hip_dice_mean, epoch_index)
        writer.add_scalar('Alpha', alpha, epoch_index)
        if hip_dice_mean > best_dice:
            best_dice = hip_dice_mean
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'Best_Dice.pth'))
        if epoch_index % 10 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'CsaUnet_model_epoch_{epoch_index}.pth'))
    writer.close()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu
    train()
