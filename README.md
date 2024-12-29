# HSI-for-Wheat-Crops-Survey-Article
A summary of articles, and visualizations related to hyperspectral imaging (HSI) Using Deep learning in wheat crops
## Table of Contents

- [Methodology](#Methodology)
- [Supervised Learning](#Supervised-Learning)
  - [CNN](#CNN)
  - [DBN](#DBN)
  - [GAN](#GAN)
  - [RNN](#RNN)
  - [SAE](#SAE)
  - [TL](#TL)
  - [Transformer](#Transformer)
  - [Mamba](#Mamba)
- [Semi-Supervised Learning](#Semi-Supervised-Learning)
- [Unsupervised Learning](#Unsupervised-Learning)

## Methodology
### CNN
- Kshitiz Dhakal, Upasana Sivaramakrishnan, Xuemei Zhang, Kassaye Belay, Joseph Oakes, Xing Wei, and Song Li. Machine learning analysis of
hyperspectral images of damaged wheat kernels. Sensors, 23(7):3523, 2023. [Paper](https://www.mdpi.com/1424-8220/23/7/3523)
- Zilong Zhong, Jonathan Li, Zhiming Luo, and Michael Chapman. Spectral–spatial residual network for hyperspectral image classification: A 3-d
deep learning framework. IEEE Transactions on Geoscience and Remote Sensing, 56(2):847–858, 2017. [Paper](https://ieeexplore.ieee.org/abstract/document/8061020)
- Yushi Chen, Hanlu Jiang, Chunyang Li, Xiuping Jia, and Pedram Ghamisi. Deep feature extraction and classification of hyperspectral images based
on convolutional neural networks. IEEE transactions on geoscience and remote sensing, 54(10):6232–6251, 2016. [Paper](https://ieeexplore.ieee.org/abstract/document/7514991)
- Xiaofei Yang, Yunming Ye, Xutao Li, Raymond YK Lau, Xiaofeng Zhang, and Xiaohui Huang. Hyperspectral image classification with deep learning
models. IEEE Transactions on Geoscience and Remote Sensing, 56(9):5408–5423, 2018. [Paper](https://ieeexplore.ieee.org/abstract/document/8340197)
- LIU Bing, YU Xuchu, ZHANG Pengqiang, and TAN Xiong. Deep 3d convolutional network combined with spatial-spectral features for hyperspectral
image classification. Acta Geodaetica et Cartographica Sinica, 48(1):53, 2019. [Paper](http://xb.chinasmp.com/EN/Y2019/V48/I1/53)
- Swalpa Kumar Roy, Gopal Krishna, Shiv Ram Dubey, and Bidyut B Chaudhuri. Hybridsn: Exploring 3-d–2-d cnn feature hierarchy for hyperspectral
image classification. IEEE Geoscience and Remote Sensing Letters, 17(2):277–281, 2019. [Paper](https://ieeexplore.ieee.org/abstract/document/8736016)
- Weiwei Song, Shutao Li, Leyuan Fang, and Ting Lu. Hyperspectral image classification with deep feature fusion network. IEEE Transactions on
Geoscience and Remote Sensing, 56(6):3173–3184, 2018. [Paper](https://ieeexplore.ieee.org/abstract/document/8283837)
- Vimal K. Shrivastava Somenath Bera and Suresh Chandra Satapathy. Advances in hyperspectral image classification based on convolutional
neural networks: A review. Tech Science Press (TSP), 2022. [Paper](https://www.techscience.com/CMES/v133n2/48965/html)
- Ying Li, Haokui Zhang, and Qiang Shen. Spectral–spatial classification of hyperspectral imagery with 3d convolutional neural network. Remote
Sensing, 9(1):67, 2017. [Paper](https://www.mdpi.com/2072-4292/9/1/67)
- Leyuan Fang, Zhiliang Liu, and Weiwei Song. Deep hashing neural networks for hyperspectral image feature extraction. IEEE Geoscience and
Remote Sensing Letters, 16(9):1412–1416, 2019. [Paper](https://ieeexplore.ieee.org/abstract/document/8663589)
- Hyungtae Lee and Heesung Kwon. Going deeper with contextual cnn for hyperspectral image classification. IEEE Transactions on Image Processing,
26(10):4843–4855, 2017. [Paper](https://ieeexplore.ieee.org/abstract/document/7973178)
### RNN
- R Venkatesan and Sevugan Prabu. Hyperspectral image features classification using deep learning recurrent neural networks. Journal of medical
systems, 43(7):216, 2019. [Paper](https://link.springer.com/article/10.1007/s10916-019-1347-9)
- Haowen Luo. Shorten spatial-spectral rnn with parallel-gru for hyperspectral image classification. arXiv preprint arXiv:1810.12563, 2018. [Paper](https://arxiv.org/abs/1810.12563)
- Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence
modeling. arXiv preprint arXiv:1412.3555, 2014. [Paper](https://arxiv.org/abs/1412.3555)
- Lichao Mou, Pedram Ghamisi, and Xiao Xiang Zhu. Deep recurrent neural networks for hyperspectral image classification. IEEE transactions on
geoscience and remote sensing, 55(7):3639–3655, 2017. [Paper](https://ieeexplore.ieee.org/abstract/document/7914752)
- Emile Ndikumana, Dinh Ho Tong Minh, Nicolas Baghdadi, Dominique Courault, and Laure Hossard. Deep recurrent neural network for agricultural
classification using multitemporal sar sentinel-1 for camargue, france. Remote Sensing, 10(8):1217, 2018. [Paper](https://www.mdpi.com/2072-4292/10/8/1217)
- Lauri Salmela, Nikolaos Tsipinakis, Alessandro Foi, Cyril Billet, John M Dudley, and Goëry Genty. Predicting ultrafast nonlinear dynamics in fibre optics with a recurrent neural network. Nature machine intelligence, 3(4):344–354, 2021. [Paper](https://www.nature.com/articles/s42256-021-00297-z)
### Transformer
- Lanxue Dang, Libo Weng, Yane Hou, Xianyu Zuo, and Yang Liu. Double-branch feature fusion transformer for hyperspectral image classification.
Scientific Reports, 13(1):272, 2023. [Paper](https://www.nature.com/articles/s41598-023-27472-z)
- Neetu Sigger, Quoc-Tuan Vien, Sinh Van Nguyen, Gianluca Tozzi, and Tuan Thanh Nguyen. Unveiling the potential of diffusion model-based
framework with transformer for hyperspectral image classification. Scientific Reports, 14(1):8438, 2024. [Paper](https://www.nature.com/articles/s41598-024-58125-4)
- Shukai Liu, Changqing Yin, and Huijuan Zhang. Cesa-mcformer: An efficient transformer network for hyperspectral image classification by
eliminating redundant information. Sensors, 24(4):1187, 2024. [Paper](https://www.mdpi.com/1424-8220/24/4/1187)
- Jiaxing Xie, Jiajun Hua, Shaonan Chen, PeiwenWu, Peng Gao, Daozong Sun, Zhendong Lyu, Shilei Lyu, Xiuyun Xue, and Jianqiang Lu. Hypersformer:
A transformer-based end-to-end hyperspectral image classification method for crop classification. Remote Sensing, 15(14):3491, 2023. [Paper](https://www.mdpi.com/2072-4292/15/14/3491)
### Mamba
- Jiamu Sheng, Jingyi Zhou, Jiong Wang, Peng Ye, and Jiayuan Fan. Dualmamba: A lightweight spectral-spatial mamba-convolution network for
hyperspectral image classification. arXiv preprint arXiv:2406.07050, 2024. [Paper](https://arxiv.org/abs/2406.07050)
- Judy X Yang, Jun Zhou, Jing Wang, Hui Tian, and Alan Wee Chung Liew. Hsimamba: Hyperpsectral imaging efficient feature learning with
bidirectional state space for classification. arXiv preprint arXiv:2404.00272, 2024. [Paper](https://arxiv.org/abs/2404.00272)
- Weilian Zhou, Sei-Ichiro Kamata, Haipeng Wang, Man-Sing Wong, et al. Mamba-in-mamba: Centralized mamba-cross-scan in tokenized mamba
model for hyperspectral image classification. arXiv preprint arXiv:2405.12003, 2024. [Paper](https://www.sciencedirect.com/science/article/pii/S0925231224015224)
- Lingbo Huang, Yushi Chen, and Xin He. Spectral-spatial mamba for hyperspectral image classification. arXiv preprint arXiv:2404.18401, 2024. [Paper](https://arxiv.org/abs/2404.18401)
- Yan He, Bing Tu, Bo Liu, Jun Li, and Antonio Plaza. 3dss-mamba: 3d-spectral-spatial mamba for hyperspectral image classification. arXiv preprint
arXiv:2405.12487, 2024. [Paper](https://arxiv.org/abs/2405.12487)
- Aitao Yang, Min Li, Yao Ding, Leyuan Fang, Yaoming Cai, and Yujie He. Graphmamba: An efficient graph structure learning vision mamba for
hyperspectral image classification. arXiv preprint arXiv:2407.08255, 2024. [Paper](https://ieeexplore.ieee.org/abstract/document/10746459)
