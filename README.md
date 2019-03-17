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

[JuntingPan-photo]: https://github.com/junting/seg2vid/blob/junting/authors/juntingpan.png "Junting Pan"
[ChengyuWang-photo]: https://github.com/junting/seg2vid/blob/junting/authors/ChengyuWang.png "Chengyu Wang"
[XuJia-photo]: https://github.com/junting/seg2vid/blob/junting/authors/XuJia.png "Xu Jia"
[JingShao-photo]: https://github.com/junting/seg2vid/blob/junting/authors/JingShao.png "JingShao"
[LuSheng-photo]: https://github.com/junting/seg2vid/blob/junting/authors/lusheng.png "Lu Sheng"
[JunjieYan-photo]: https://github.com/junting/seg2vid/blob/junting/authors/JunjieYan.png "Junjie Yan"
[XiaogangWang-photo]: https://github.com/junting/seg2vid/blob/junting/authors/XiaogangWang.png "Xiaogang Wang"

A joint collaboration between:

| ![logo-sensetime] | ![logo-huawei] | ![logo-cuhk] | 
|:-:|:-:|:-:|
| [Sensetime Research][sensetime-web] | [Huawei][huawei-web] | [CUHK][cuhk-web] | 

[logo-sensetime]: https://github.com/junting/seg2vid/blob/junting/logos/sensetime.png "Sensetime Research"
[logo-huawei]: https://github.com/junting/seg2vid/blob/junting/logos/huawei.png "Huawei"
[logo-cuhk]: https://github.com/junting/seg2vid/blob/junting/logos/cuhk.jpg "cuhk"



[sensetime-web]: https://www.sensetime.com/
[huawei-web]: http://www.noahlab.com.hk/#/home
[cuhk-web]: http://www.cuhk.edu.hk/english/index.html





## Abstract

This paper proposes the novel task of video generation conditioned on a SINGLE semantic label map, which provides a good balance between flexibility and quality in the generation process. Different from typical end-to-end approaches, which model both scene content and dynamics in a single step, we propose to decompose this difficult task into two sub-problems. As current image generation methods do better than video generation in terms of detail, we synthesize high quality content by only generating the first frame. Then we animate the scene based on its semantic meaning to obtain the temporally coherent video, giving us excellent results overall. We employ a cVAE for predicting optical flow as a beneficial intermediate step to generate a video sequence conditioned on the initial single frame. A semantic label map is integrated into the flow prediction module to achieve major improvements in the image-to-video generation process. Extensive experiments on the Cityscapes dataset show that our method outperforms all competing methods.


## Publication

Find our work on [arXiv](https://arxiv.org/abs/1701.01081). 

![Image of the paper](https://github.com/junting/seg2vid/blob/junting/figs/paper_thumbnail.jpg)

Please cite with the following Bibtex code:

```
@article{pan2019video,
  title={Video Generation from Single Semantic Label Map},
  author={Pan, Junting and Wang, Chengyu and Jia, Xu and Shao, Jing and Sheng, Lu and Yan, Junjie and Wang, Xiaogang},
  journal={arXiv preprint arXiv:1903.04480},
  year={2019}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Junting Pan, Chengyu Wang, Xu Jia, Jing Shao, Lu Sheng, Junjie Yan and Xiaogang Wang. "Video Generation from Single Semantic Label Map." CVPR 2019.*


## Models

The Seg2Vid presented in our work can be downloaded from the links provided below the figure:

Seg2Vid Architecture
![architecture-fig]

* [[SalGAN Generator Model (127 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/gen_modelWeights0090.npz)
* [[SalGAN Discriminator (3.4 MB)]](https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2017-salgan/discrim_modelWeights0090.npz)

[architecture-fig]: https://github.com/junting/seg2vid/blob/junting/figs/full_architecture.png "SALGAN architecture"
[shallow-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/shallow_net.pickle
[deep-model]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_model.caffemodel
[deep-prototxt]: https://imatge.upc.edu/web/sites/default/files/resources/1720/saliency/2016-cvpr/deep_net_deploy.prototxt

## Visual Results
### Cityscapes (Generation)
| ![Generated Video 1]  | ![Generated Video 2]  |
|:-:|:-:|

[Generated Video 1]:https://github.com/junting/seg2vid/blob/junting/gifs/generation/gcity_1.gif
[Generated Video 2]:https://github.com/junting/seg2vid/blob/junting/gifs/generation/gcity_2.gif

### Cityscapes (Prediction)
| ![Predicted Video 1]  | ![Predicted Video 2]  |
|:-:|:-:|
| ![Predicted Flow 1]  | ![Predicted Flow 2]  |

[Predicted Video 1]:https://github.com/junting/seg2vid/blob/junting/gifs/flow/pcity_1.gif
[Predicted Video 2]:https://github.com/junting/seg2vid/blob/junting/gifs/flow/pcity_2.gif
[Predicted Flow 1]:https://github.com/junting/seg2vid/blob/junting/gifs/flow/flow_1.gif
[Predicted Flow 2]:https://github.com/junting/seg2vid/blob/junting/gifs/flow/flow_2.gif

### Cityscapes 24 frames (Prediction)
| ![Long Video 1]  | ![Long Video 2]  | ![Long Video 3]  |
|:-:|:-:|:-:|

[Long Video 1]:https://github.com/junting/seg2vid/blob/junting/gifs/length/lcity_1.gif
[Long Video 2]:https://github.com/junting/seg2vid/blob/junting/gifs/length/lctiy_2.gif
[Long Video 3]:https://github.com/junting/seg2vid/blob/junting/gifs/length/lcity_3.gif

### UCF-101 (Prediction)
| ![UCF Video 1]  | ![UCF Video 2]  | ![UCF Video 3] | ![UCF Video 4]  | ![UCF Video 5]  | ![UCF Video 6]  |
|:-:|:-:|:-:|:-:|:-:|:-:|

[UCF Video 1]:https://github.com/junting/seg2vid/blob/junting/gifs/ucf101/ice_01.gif
[UCF Video 2]:https://github.com/junting/seg2vid/blob/junting/gifs/ucf101/ice_02.gif
[UCF Video 3]:https://github.com/junting/seg2vid/blob/junting/gifs/ucf101/ice_03.gif
[UCF Video 4]:https://github.com/junting/seg2vid/blob/junting/gifs/ucf101/violin_01.gif
[UCF Video 5]:https://github.com/junting/seg2vid/blob/junting/gifs/ucf101/violin_02.gif
[UCF Video 6]:https://github.com/junting/seg2vid/blob/junting/gifs/ucf101/violin_03.gif

### KTH (Prediction)
| ![KTH Video 1]  | ![KTH Video 2]  | ![KTH Video 3] | ![KTH Video 4]  | ![KTH Video 5]  | 
|:-:|:-:|:-:|:-:|:-:|

[KTH Video 1]:https://github.com/junting/seg2vid/blob/junting/gifs/kth/kth_1.gif
[KTH Video 2]:https://github.com/junting/seg2vid/blob/junting/gifs/kth/kth_2.gif
[KTH Video 3]:https://github.com/junting/seg2vid/blob/junting/gifs/kth/kth_3.gif
[KTH Video 4]:https://github.com/junting/seg2vid/blob/junting/gifs/kth/kth_4.gif
[KTh Video 5]:https://github.com/junting/seg2vid/blob/junting/gifs/kth/kth_5.gif

### Training
As explained in our paper, our networks were trained on the training and validation data provided by [Cityscapes](http://salicon.net/).

### Test
Two different dataset were used for test:
* Test partition of [SALICON](http://salicon.net/) dataset.
* [MIT300](http://saliency.mit.edu/datasets.html).


## Software frameworks

Our paper presents two convolutional neural networks, one correspends to the Generator (Saliency Prediction Network) and the another is the Discriminator for the adversarial training. To compute saliency maps only the Generator is needed.

### Seg2Vid on Pytorch

Seg2Vid is implemented in [Pytorch](https://pytorch.org/).

### Usage

To train our model from scrath you need to run the following command:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,optimizer_including=cudnn python 02-train.py
```
In order to run the test script to predict saliency maps, you can run the following command after specifying the path to you images and the path to the output saliency maps:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,optimizer_including=cudnn python 03-predict.py
```

Download the pretrained VGG-16 weights from: [vgg16.pkl](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl)


## Acknowledgements

We would like to especially thank Albert Gil Moreno and Josep Pujal from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  | ![JosepPujal-photo]  |
|:-:|:-:|
| [Albert Gil](AlbertGil-web)  |  [Josep Pujal](JosepPujal-web) |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/junting/seg2vid/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:junting.pa@gmail.com>.

<!---
Javascript code to enable Google Analytics
-->
