# Video Generation from Single Semantic Label Map
## [Paper accepted at CVPR 2019 ](http://cvpr2019.thecvf.com/)

| ![Junting Pan][JuntingPan-photo]  | ![Chengyu Wang][ChengyuWang-photo]  |  ![Xu Jia][XuJia-photo] | ![Jing Shao][JingShao-photo] | ![Lu Sheng][LuSheng-photo] |![Junjie Yan][JunjieYan-photo]  | ![Xiaogang Wang][XiaogangWang-photo]  |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [Junting Pan][JuntingPan-web]  | [Chengyu Wang][ChengyuWang-web] | [Xu Jia][XuJia-web] | [Jing Shao][JingShao-web] |  [Lu Sheng][LuSheng-web] | [Junjie Yan][JunjieYan-web]  | [Xiaogang Wang][XiaogangWang-web]   |

[JuntingPan-web]: https://junting.github.io/
[ChengyuWang-web]: https://www.linkedin.com/in/chengyu-wang/
[XuJia-web]: https://stephenjia.github.io/
[JingShao-web]: https://amandajshao.github.io/
[LuSheng-web]: https://scholar.google.com.hk/citations?user=_8lB7xcAAAAJ&hl=en
[JunjieYan-web]: http://www.cbsr.ia.ac.cn/users/jjyan/main.htm
[XiaogangWang-web]: http://www.ee.cuhk.edu.hk/~xgwang/

[JuntingPan-photo]: https://github.com/junting/seg2vid/blob/junting/authors/JuntingPan.jpeg "Junting Pan"
[ChengyuWang-photo]: https://github.com/junting/seg2vid/blob/junting/authors/ChengyuWang.png "Chengyu Wang"
[XuJia-photo]: https://github.com/junting/seg2vid/blob/junting/authors/XuJia.png "Xu Jia"
[JingShao-photo]: https://github.com/junting/seg2vid/blob/junting/authors/JingShao.png "JingShao"
[LuSheng-photo]: https://github.com/junting/seg2vid/blob/junting/authors/LuSheng.jpeg "Lu Sheng"
[JunjieYan-photo]: https://github.com/junting/seg2vid/blob/junting/authors/JunjieYan.png "Junjie Yan"
[XiaogangWang-photo]: https://github.com/junting/seg2vid/blob/junting/authors/XiaogangWang.png "Xiaogang Wang"

A joint collaboration between:

| ![logo-sensetime] | ![logo-huawei] | ![logo-cuhk] | 
|:-:|:-:|:-:|
| [Sensetime Research][sensetime-web] | [Huawei][huawei-web] | [CUHK][cuhk-web] | 

[logo-sensetime]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Sensetime Research"
[logo-huawei]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/dcu.png "Huawei"
[logo-cuhk]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/logos/microsoft.jpg?token=AFOjyc8Q1kkjcWIP-yen0FTEo0lsWPk6ks5Yc3j4wA%3D%3D "cuhk"



[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/
[microsoft-web]: https://www.microsoft.com/en-us/research/





## Abstract

This paper proposes the novel task of video generation conditioned on a SINGLE semantic label map, which provides a good balance between flexibility and quality in the generation process. Different from typical end-to-end approaches, which model both scene content and dynamics in a single step, we propose to decompose this difficult task into two sub-problems. As current image generation methods do better than video generation in terms of detail, we synthesize high quality content by only generating the first frame. Then we animate the scene based on its semantic meaning to obtain the temporally coherent video, giving us excellent results overall. We employ a cVAE for predicting optical flow as a beneficial intermediate step to generate a video sequence conditioned on the initial single frame. A semantic label map is integrated into the flow prediction module to achieve major improvements in the image-to-video generation process. Extensive experiments on the Cityscapes dataset show that our method outperforms all competing methods.


## Publication

Find our work on [arXiv](https://arxiv.org/abs/1701.01081). 

![Image of the paper](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/figs/thumbnails.jpg)

Please cite with the following Bibtex code:

```
@InProceedings{Pan_2017_SalGAN,
author = {Pan, Junting and Canton, Cristian and McGuinness, Kevin and O'Connor, Noel E. and Torres, Jordi and Sayrol, Elisa and Giro-i-Nieto, Xavier and},
title = {SalGAN: Visual Saliency Prediction with Generative Adversarial Networks},
booktitle = {arXiv},
month = {January},
year = {2017}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Junting Pan, Cristian Canton, Kevin McGuinness, Noel E. O'Connor, Jordi Torres, Elisa Sayrol and Xavier Giro-i-Nieto. "SalGAN: Visual Saliency Prediction with Generative Adversarial Networks." arXiv. 2017.*



## Models

The SalGAN presented in our work can be downloaded from the links provided below the figure:

SalGAN Architecture
![architecture-fig]

* [[SalGAN Generator Model (127 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/gen_modelWeights0090.npz)
* [[SalGAN Discriminator (3.4 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/discrim_modelWeights0090.npz)

[architecture-fig]: https://github.com/junting/seg2vid/blob/junting/figs/full_architecture.png "SALGAN architecture"
[shallow-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/shallow_net.pickle
[deep-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_model.caffemodel
[deep-prototxt]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_deploy.prototxt

## Visual Results

![Qualitative saliency predictions](https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/figs/qualitative.jpg?token=AFOjyaO0uT7l7qGzV7IyrcSgi8ieeayTks5Yc4s2wA%3D%3D)


## Datasets

### Training
As explained in our paper, our networks were trained on the training and validation data provided by [SALICON](http://salicon.net/).

### Test
Two different dataset were used for test:
* Test partition of [SALICON](http://salicon.net/) dataset.
* [MIT300](http://saliency.mit.edu/datasets.html).


## Software frameworks

Our paper presents two convolutional neural networks, one correspends to the Generator (Saliency Prediction Network) and the another is the Discriminator for the adversarial training. To compute saliency maps only the Generator is needed.

### SalGAN on Lasagne

SalGAN is implemented in [Lasagne](https://github.com/Lasagne/Lasagne), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).
```
pip install -r https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/requirements.txt
```

### SalGAN on a docker

We have prepared [this Docker container](https://hub.docker.com/r/evamohe/salgan/) with all necessary dependencies for computing saliency maps with SalGAN. You will need to use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Using the container is like connecting via ssh to a machine. To start an interactive session run:

```
    >> sudo nvidia-docker run -it --entrypoint='bash' -w /home/ evamohe/salgan
```

This will open a terminal within the container located in the '/home' folder. 

Yo will find Salgan code in "/home/salgan". So if you want to test the installation, within the container, run:

```
   >> cd /home/salgan/scripts
   >> THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=0.5,optimizer_including=cudnn python 03-predict.py
```

That will process the sample images located in "/home/salgan/images" and store them in "/home/salgan/saliency". To exit the container, run:

```
   >> exit
```

You migh want to process your own data with your own custom scripts. For that, you can mount different local folders in the container. For example:

```
>> sudo nvidia-docker run -v $PATH_TO_MY_CODE:/home/code -v $PATH_TO_MY_DATA:/home/data -it --entrypoint='bash' -w /home/
```

will open a new session in the container, with '/home/code' and '/home/data' folders that will be share with your computer. If you edit your code locally, the changes will be updated automatically in the container. Similarly, all the files generated in '/home/data' will be available in your original data folder.

### Usage

To train our model from scrath you need to run the following command:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,optimizer_including=cudnn python 02-train.py
```
In order to run the test script to predict saliency maps, you can run the following command after specifying the path to you images and the path to the output saliency maps:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,optimizer_including=cudnn python 03-predict.py
```
With the provided model weights you should obtain the follwing result:

| ![Image Stimuli]  | ![Saliency Map]  |
|:-:|:-:|

[Image Stimuli]:https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/images/i112.jpg
[Saliency Map]:https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/master/saliency/i112.jpg

Download the pretrained VGG-16 weights from: [vgg16.pkl](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl)

## External implementation in PyTorch

Bat-Orgil Batsaikhan and Catherine Qi Zhao from the University of Minnesota released a [PyTorch implementation](https://github.com/batsa003/salgan/) in 2018 as part of their poster ["Generative Adversarial Network for Videos and Saliency Map"](http://hdl.handle.net/11299/194302).


## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  | ![JosepPujal-photo]  |
|:-:|:-:|
| [Albert Gil](AlbertGil-web)  |  [Josep Pujal](JosepPujal-web) |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan Z](http://www.nvidia.com/gtx-700-graphics-cards/gtx-titan-z/) and [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of the projects [BigGraph TEC2013-43935-R](https://imatge.upc.edu/web/projects/biggraph-heterogeneous-information-and-graph-signal-processing-big-data-era-application) and [Malegra TEC2016-75976-R](https://imatge.upc.edu/web/projects/malegra-multimodal-signal-processing-and-machine-learning-graphs), funded by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 
|  This publication has emanated from research conducted with the financial support of Science Foundation Ireland (SFI) under grant number SFI/12/RC/2289. |  ![logo-ireland] |

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"
[logo-ireland]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/sfi.png "Logo of Science Foundation Ireland"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/imatge-upc/saliency-salgan-2017/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:xavier.giro@upc.edu>.

<!---
Javascript code to enable Google Analytics
-->
