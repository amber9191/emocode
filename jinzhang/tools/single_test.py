from argparse import ArgumentParser
import os
import sys
import os
sys.path.insert(0,os.getcwd())
import torch
from utils.inference import inference_model, init_model, show_result_pyplot
from utils.train_utils import get_info, file2dict
from models.build import BuildNet
result_txt = "D:\\result.txt"

def main():
    parser = ArgumentParser()
    # parser.add_argument('--img', help='Image file')
    parser.add_argument('config',help='Config file')
    parser.add_argument(
        '--classes-map', default='datas/annotations.txt', help='classes map of datasets')
    parser.add_argument(
        '--device', default='cuda', help='Device used for inference')
    parser.add_argument(
        '--save-path',
        help='The path to save prediction image, default not to save.')
    args = parser.parse_args()

    classes_names, label_names = get_info(args.classes_map)
    # build the model from a config file and a checkpoint file
    model_cfg,train_pipeline,val_pipeline,data_cfg,lr_config,optimizer_cfg = file2dict(args.config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')
    file = r'C:\Users\lyx\Documents\Awesome-Backbones-main\Awesome-Backbones-main\dataset\sscissors1'
    for root, dirs, files in os.walk(file):
        for i in range(len(files)):
            img = "datasets/test/scissors1/" + files[i]
            img="datasets/test/scissors1/IMG_20191014_204844.jpg"
            # print(img)
            result = inference_model(model, img, val_pipeline, classes_names, label_names)
            show_result_pyplot(model, img, result, out_file=args.save_path)
            # print(result)
            final_result = str(files[i]) +"    "+str(result['pred_class'])
            print(final_result)
            with open(result_txt, "w", encoding='utf-8') as f:
                f.write(final_result)
                f.close()
    img = "datasets/test/tweezers/IMG_20191014_213924.jpg"
    # test a single image
    result = inference_model(model, img, val_pipeline, classes_names,label_names)
    # show the results
    print(result)
    show_result_pyplot(model, img, result, out_file=args.save_path)


if __name__ == '__main__':
    main()
