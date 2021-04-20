# texture-descriptions
Re-implementation study of the main contributions in "Describing Textures with Natural Language"

## Setup Steps
DTD Download:
```
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvf dtd-r1.0.1.tar.gz
```
DTD^2 Download:
```
cd dtd/images
wget https://github.com/ChenyunWu/DescribingTextures/raw/master/data_api/data/image_descriptions.json
wget https://github.com/ChenyunWu/DescribingTextures/raw/master/data_api/data/image_splits.json
wget https://github.com/ChenyunWu/DescribingTextures/raw/master/data_api/data/phrase_freq.txt
wget https://github.com/ChenyunWu/DescribingTextures/raw/master/data_api/data/phrase_freq_train.txt
wget https://github.com/ChenyunWu/DescribingTextures/raw/master/data_api/data/word_freq.txt
wget https://github.com/ChenyunWu/DescribingTextures/raw/master/data_api/data/word_freq_train.txt
```
Download my code (CUDA 11 Env):
```
cd ../..
wget https://github.com/tonymisic/texture-descriptions.git
pip install -r requirements.txt
```

## Running experiments
To run my code you will be prompted to make a wandb account, for more details see: https://wandb.ai/site
```
python <filename>
```
Some files are left incomplete due to non reproducible results, and therefore will not run properly.
