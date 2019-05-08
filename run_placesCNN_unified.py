# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from multiprocessing import Pool, Process
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import os.path as osp
import numpy as np
from scipy.misc import imresize as imresize
import cv2
from PIL import Image
import argparse
torch.set_num_threads( 1)

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = 'categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = 'IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = 'labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = 'W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365ces/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


# def returnCAM(feature_conv, weight_softmax, class_idx):
#     # generate the class activation maps upsample to 256x256
#     size_upsample = (256, 256)
#     nc, h, w = feature_conv.shape
#     output_cam = []
#     for idx in class_idx:
#         cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
#         cam = cam.reshape(h, w)
#         cam = cam - np.min(cam)
#         cam_img = cam / np.max(cam)
#         cam_img = np.uint8(255 * cam_img)
#         output_cam.append(imresize(cam_img, size_upsample))
#     return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
#    model_file = 'resnet50_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    #model = model.to('cuda')
    model.eval()



    # the following is deprecated, everything is migrated to python36

    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    #from functools import partial
    #import pickle
    #pickle.load = partial(pickle.load, encoding="latin1")
    #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

    # # hook the feature extractor
    # features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    # for name in features_names:
    #     model._modules.get(name).register_forward_hook(hook_feature)
    return model


# hook the feature extractor

def create_folder(fold_path):
    if not osp.exists(fold_path):
        os.makedirs(fold_path)


classes, _, labels_attribute, W_attribute = load_labels()

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder', help="Folder contains scene images")
    parser.add_argument('attribute_folder', help='Folder contains extracted attribute features')
    parser.add_argument('category_folder', help='Folder contains extracted category features')
    parser.add_argument('raw_folder', help='Folder contains raw features of placeCNN')
    parser.add_argument('--attribute-folder-log', dest='attribute_folder_log', help='Folder contains human-readable data of attribute features')
    parser.add_argument('--category-folder-log', dest='category_folder_log', help='Folder contains human-readable data of scene features')
    return parser

def forward(root, f, tf, W_attribute, model, features_blobs):
    print(f)

#        return model

    image_path = osp.join(root, f)
    attribute_filepath = image_path.replace(args.image_folder, args.attribute_folder).split('.')[0] + '.npy'
    category_filepath = image_path.replace(args.image_folder, args.category_folder).split('.')[0] + '.npy'
    raw_filepath = image_path.replace(args.image_folder, args.raw_folder).split('.')[0] + '.npy'

    # Do not process image which already have result
    if osp.exists(raw_filepath):
        return
    # ----------------------------------------------

    # load image and apply image transformation
    try:
        img = Image.open(image_path)
    except Exception as e:
        with open('logs.txt', 'a') as f:
            print >> f, e
        return
    # Handle some special image format in PIL Image library and convert them into RGB to process properly
    img = img.convert('RGB')
    input_img = V(tf(img).unsqueeze(0))
    #input_img = input_img.to('cuda')

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()

    # print('RESULT ON {}'.format(image_path))
    # # output the IO prediction
    # io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    # if io_image < 0.5:
    #     print('--TYPE OF ENVIRONMENT: indoor')
    # else:
    #     print('--TYPE OF ENVIRONMENT: outdoor')

    if args.category_folder_log != None:
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        category_file_logpath = image_path.replace(args.image_folder, args.category_folder_log).split('.')[0] + '.txt'
        with open(category_file_logpath, 'w') as f:
        # output the prediction of scene category
#            print >> f, '--SCENE CATEGORIES:'
            for i in range(0, 5):
                print >> f, '{:.3f} -> {}'.format(probs[i], classes[idx[i]])

    # output scene category score
    np.save(category_filepath, h_x.cpu().numpy())
    
    # output the scene attributes 
    responses_attribute = W_attribute.dot(features_blobs[1]) 
    np.save(attribute_filepath, responses_attribute) 

    # output raw features
    np.save(raw_filepath, features_blobs[1])
    
    if args.attribute_folder_log != None:
        attribute_file_logpath = image_path.replace(args.image_folder, args.attribute_folder_log).split('.')[0] + '.txt'
        idx_a = np.argsort(responses_attribute)
        with open(attribute_file_logpath, 'w') as f:
#            print >> f, '--SCENE ATTRIBUTES:'
            print >> f, ', '.join([labels_attribute[idx_a[i]] for i in range(-1,-11,-1)])


def group_async_task(params, model):
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(np.squeeze(output.data.cpu().numpy()))

#    def hook_feature_extractor(model):
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    for param in params:
        forward(*param, model = model, features_blobs = features_blobs)
        features_blobs = []

def run_placesCNN_extract_feature(args):
    # load the labels
    # classes, labels_IO, labels_attribute, W_attribute = load_labels()

    # load the model
    model = load_model()

    # load the transformer
    tf = returnTF() # image transformer

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.cpu().numpy()
    weight_softmax[weight_softmax<0] = 0

    create_folder(args.attribute_folder)
    create_folder(args.category_folder)
    create_folder(args.raw_folder)
    if args.category_folder_log != None:
        create_folder(args.category_folder_log)
    if args.attribute_folder_log != None:
        create_folder(args.attribute_folder_log)

    params = []
    for root, dirs, files in os.walk(args.image_folder):
        for d in sorted(dirs):
            _d = osp.join(root, d)
            attribute_dir = _d.replace(args.image_folder, args.attribute_folder)
            category_dir = _d.replace(args.image_folder, args.category_folder)
            raw_dir = _d.replace(args.image_folder, args.raw_folder)
            create_folder(attribute_dir)
            create_folder(category_dir)            
            create_folder(raw_dir)
            
            if args.attribute_folder_log != None:
                attribute_log_dir = _d.replace(args.image_folder, args.attribute_folder_log)
                create_folder(attribute_log_dir)
            
            if args.category_folder_log != None:
                category_log_dir = _d.replace(args.image_folder, args.category_folder_log)
                create_folder(category_log_dir)

        for f in sorted(files):
            image_path = osp.join(root, f)
            raw_filepath = image_path.replace(args.image_folder, args.raw_folder).split('.')[0] + '.npy'

            # Do not process image which already have result
            if osp.exists(raw_filepath):
                continue
            params.append((root, f, tf, W_attribute))

    num_processes = 20
    Nperprocess = params.__len__()//num_processes + 1
    p = []
    for i in range(num_processes):
        subparams = params[i*Nperprocess: min(i*Nperprocess + Nperprocess, params.__len__())]
        p.append(Process(target=group_async_task, args=(subparams, model)))
        p[-1].start()
    for task_group in p:
        task_group.join()
# generate class activation mapping
# print('Class activation map is saved as cam.jpg')
# CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# # render the CAM and output
# img = cv2.imread('test.jpg')
# height, width, _ = img.shape
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# result = heatmap * 0.4 + img * 0.5
# cv2.imwrite('cam.jpg', result)

if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()    
    run_placesCNN_extract_feature(args)
