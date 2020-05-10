# This file is to download and extract the datasets to be used for the Text to Image synthesis model training.
# To run this in colab use command ->   !python downloads_dataset.py

# Import necessary packages
import os
import sys
import errno
import tarfile
from urllib.request import urlretrieve
import nltk


# Main directory for all images, captions, models, samples and skip-thought vectors
DATA_DIR = 'Data'


# http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
# This function takes in a path and checks if the same exists.
def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

# Function to create subdirectories under the main  directory.
# samples -> all the real images used and the fake image generated for every batch during an epoch is saved in this dir.
# val_samples -> all images generated given a caption by the user is saved here. (Using generate_image.py)
# models -> the model created after every epoch is saved here. (Using train.py) Every model overwrites the previous one.
def create_data_paths():
    if not os.path.isdir(DATA_DIR):
        raise EnvironmentError('Needs to be run from project directory containing ' + DATA_DIR)
    # os.path.join is used to add a new directory in the given directory.(Like mkdir)
    needed_paths = [
        os.path.join(DATA_DIR, 'samples'),
        os.path.join(DATA_DIR, 'val_samples'),
        os.path.join(DATA_DIR, 'Models'),
    ]
    for p in needed_paths:
        # confirming if the above paths have been successfully created
        make_sure_path_exists(p)


# Function to observe the completion percentage of a download.
# adapted from http://stackoverflow.com/questions/51212/how-to-write-a-download-progress-indicator-in-python
def dl_progress_hook(count, blockSize, totalSize):
    percent = int(count * blockSize * 100 / totalSize)
    sys.stdout.write("\r" + "...%d%%" % percent)
    sys.stdout.flush()


# Main function which downloads certain datasets - (flower images & captions)
def download_dataset(data_name):
    if data_name == 'flowers':
        print('== Flowers dataset ==')
        flowers_dir = os.path.join(DATA_DIR, 'flowers')     #Create a new Directory "flowers" to store the images and captions.
        make_sure_path_exists(flowers_dir)

        # the original google drive link at https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view
        # the text_c10 directory from that archive as a bzipped file in the repo
        # DOWNLOAD CATIONS DIRECTLY FROM ABOVE LINK AND PASTE IT IN DATA_DIR
        captions_tbz = os.path.join(DATA_DIR, 'cvpr2016_flowers.tar.gz')
        print(('Extracting ' + captions_tbz))   
        captions_tar = tarfile.open(captions_tbz, 'r:gz')
        captions_tar.extractall(flowers_dir) #Extract captions in flowers directory


        # DOWNLOAD IMAGES
        flowers_jpg_tgz = os.path.join(flowers_dir, '102flowers.tgz')
        flowers_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
        print(('Downloading ' + flowers_jpg_tgz + ' from ' + flowers_url))

        urlretrieve(flowers_url, flowers_jpg_tgz, reporthook=dl_progress_hook)
        print(('Extracting ' + flowers_jpg_tgz))
        flowers_jpg_tar = tarfile.open(flowers_jpg_tgz, 'r:gz')
        flowers_jpg_tar.extractall(flowers_dir)  # archive contains jpg/ folder


    elif data_name == 'skipthoughts':
        print('== Skipthoughts models ==')
        SKIPTHOUGHTS_DIR = os.path.join(DATA_DIR, 'skipthoughts')   #Create a new skip-thought directory to store skip-thought related files.
        SKIPTHOUGHTS_BASE_URL = 'http://www.cs.toronto.edu/~rkiros/models/'
        make_sure_path_exists(SKIPTHOUGHTS_DIR)

        # following https://github.com/ryankiros/skip-thoughts#getting-started
        skipthoughts_files = [
            'dictionary.txt', 'utable.npy', 'btable.npy', 'uni_skip.npz', 'uni_skip.npz.pkl', 'bi_skip.npz',
            'bi_skip.npz.pkl',
        ]
        for filename in skipthoughts_files:
            src_url = SKIPTHOUGHTS_BASE_URL + filename
            print(('Downloading ' + src_url))
            urlretrieve(src_url, os.path.join(SKIPTHOUGHTS_DIR, filename), reporthook=dl_progress_hook)


    elif data_name == 'nltk_punkt':
        print('== NLTK pre-trained Punkt tokenizer for English ==')
        nltk.download('punkt')


    # Pretrained model not required.
    # elif data_name == 'pretrained_model':
    #     print('== Pretrained model ==')
    #     MODEL_DIR = os.path.join(DATA_DIR, 'Models')
    #     pretrained_model_filename = 'latest_model_flowers_temp.ckpt'
    #     src_url = 'https://bitbucket.org/paarth_neekhara/texttomimagemodel/raw/74a4bbaeee26fe31e148a54c4f495694680e2c31/' + pretrained_model_filename
    #     print(('Downloading ' + src_url))
    #     urlretrieve(src_url, os.path.join(MODEL_DIR, pretrained_model_filename), reporthook=dl_progress_hook,)

    else:
        raise ValueError('Unknown dataset name: ' + data_name)


def main():
    create_data_paths()     # Create certain paths
    download_dataset('flowers')
    download_dataset('skipthoughts')
    download_dataset('nltk_punkt')
    # download_dataset('pretrained_model')
    print('Done')


# To call the function from main driver
def startDownload():
    main()

def temp():
    print("In download_dataset.py")

if __name__ == '__main__':
    main()