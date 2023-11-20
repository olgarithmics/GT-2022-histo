import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
import openslide
from datasets.dataset_h5 import  Whole_Slide_Bag_FP, Whole_Slide_Bag_FP_LH
import cl as cl
import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle


class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        img = img.resize((224, 224))
        sample = {'input': img}

        if self.transform:
            sample = self.transform(sample)
        return sample
def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    high_patches =[item[2] for item in batch]
    return [img, coords, high_patches]

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                     transform=Compose([
                                         ToTensor()
                                     ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if magnification == 'single' or magnification == 'low':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(
                os.path.join(bags_list[i], '*.jpeg'))
        elif magnification == 'high':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*' + os.sep + '*.jpg')) + glob.glob(
                os.path.join(bags_list[i], '*' + os.sep + '*.jpeg'))
            print()
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, iteration + 1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2],
                                   bags_list[i].split(os.path.sep)[-1] + '.csv'), index=False, float_format='%.4f')


def compute_tree_feats(args, low_patches, embedder_low, embedder_high, data_slide_dir, save_path=None):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(low_patches)
    Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(0, num_bags):
            slide_id = os.path.splitext(os.path.basename(low_patches[i]))[0]

            slide_file_path = os.path.join(data_slide_dir, slide_id + '.tif')

            wsi = openslide.open_slide(slide_file_path)
            os.makedirs(save_path, exist_ok=True)

            dataset = Whole_Slide_Bag_FP_LH(file_path=low_patches[i], wsi=wsi, target_patch_size=224,
                                         custom_transforms=Compose([transforms.ToTensor()]))
            low_dataloader = DataLoader(dataset=dataset, batch_size=512, collate_fn=collate_features, drop_last=False,
                                         shuffle=False)

            feats_list = []
            feats_tree_list = []
            wsi_coords=[]
            for count, (batch, coords, high_patches) in enumerate(low_dataloader):
                with torch.no_grad():

                    batch = batch.to(device, non_blocking=True)
                    wsi_coords.append(coords)

                    low_feats, classes = embedder_low(batch)

                    low_feats = low_feats.cpu().numpy()
                    feats_list.extend(low_feats)
                    for high_patch in high_patches:
                            feats, classes = embedder_high(high_patch)
                            if args.tree_fusion == 'fusion':
                                        feats = feats.cpu().numpy() + 0.25 * feats_list[count]
                            elif args.tree_fusion == 'cat':
                                        feats = np.concatenate((feats.cpu().numpy(), feats_list[None, :][count]), axis=-1)
                            else:
                                        raise NotImplementedError(
                                            f"{args.tree_fusion} is not an excepted option for --tree_fusion. This argument accepts 2 options: 'fusion' and 'cat'.")
                            feats_tree_list.extend(feats)
                            sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i + 1, num_bags, count + 1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + low_patches[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, low_patches[i].split(os.path.sep)[-2]), exist_ok=True)
                df.to_csv(os.path.join(save_path, low_patches[i].split(os.path.sep)[-2],
                                       low_patches[i].split(os.path.sep)[-1] + '.csv'), index=False, float_format='%.4f')
            print('\n')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=512, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='instance', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='single', type=str,
                        help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default=None, type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default=None, type=str,
                        help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default=None, type=str,
                        help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--tree_fusion', default='cat', type=str,
                        help='Fusion method for high and low mag features in a tree method [cat|fusion]')
    parser.add_argument('--dataset', default='path-to-patches', type=str,
                        help='Dataset folder name [TCGA-lung-single]')
    parser.add_argument('--output', default=None, type=str, help='path to the output graph folder')
    parser.add_argument('--slide_dir', default=None, type=str, help='path to the output graph folder')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in gpu_ids)

    if args.norm_layer == 'instance':
        norm = nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':
        norm = nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()

    if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
        i_classifier_h = cl.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
        i_classifier_l = cl.IClassifier(copy.deepcopy(resnet), num_feats, output_class=args.num_classes).cuda()

        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                raise ValueError('Please use batch normalization for ImageNet feature')
        else:
            weight_path = os.path.join(args.weights_high, 'model-v0.pth')
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_h.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_h.load_state_dict(new_state_dict, strict=False)
            # os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            # torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'model-v0.pth'))

            weight_path = os.path.join(args.weights_low, 'model.pth')
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_l.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_l.load_state_dict(new_state_dict, strict=False)
            # os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            # torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-low.pth'))
            print('Use pretrained features.')


    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':
        i_classifier = cl.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

        if args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                print('Please use batch normalization for ImageNet feature')
        else:
            if args.weights is not None:
                weight_path = os.path.join('simclr', 'runs', args.weights, 'checkpoints', 'model.pth')
            else:
                weight_path = glob.glob('simclr/runs/*/checkpoints/*.pth')[-1]
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder.pth'))
            print('Use pretrained features.')


    feats_path =  args.dataset

    os.makedirs(args.output, exist_ok=True)
    bags_list = glob.glob(args.dataset)

    if args.magnification == 'tree':
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, args.slide_dir, args.output)
    else:
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)

if __name__ == '__main__':
    main()