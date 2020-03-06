import os
from pathlib import Path

from detectors.detector_factory import detector_factory
from opts import opts
from tqdm import tqdm

time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def predict_dataset(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    # images_path = Path(opt.demo)
    # data_root = Path('/media/adnan/ssd/new')
    # images_paths = []
    # for p in data_root.glob('*'):
    #     if p.is_dir() and ('_patches_' in p.parts[-1]):
    #         images_paths.append(p)
    images_paths = [Path('/media/adnan/ssd/test_set_small')]
    for images_path in images_paths:
        print(f'Predicting dataset: {images_path.parts[-1]}')
        for image_path in tqdm(Path(images_path).glob('*.jpg')):
            ret = detector.run(str(image_path))
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)


if __name__ == '__main__':
    opt = opts().init()
    predict_dataset(opt)
