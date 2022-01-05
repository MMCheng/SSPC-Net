# SSPC-Net: Semi-supervised Semantic 3D Point Cloud Segmentation Network 
by Mingmei Cheng, Le Hui, Jin Xie and Jian Yang, details are in [paper](https://ojs.aaai.org/index.php/AAAI/article/view/16200/16007)
### Requirements:

- environment:

  ```
  Ubuntu 18.04
  ```

- install python package: 

  ```
  ./Anaconda3-5.1.0-Linux-x86_64.sh
  ```
  
- install PyTorch :

  ```
  conda install pytorch==1.4.0
  ```

### Usage:

- Dataset
  
  We will update the details of pseudo label generation and data processing later. The generation of superpoints can also refer to [SPGraph](https://github.com/loicland/superpoint_graph/tree/release).

- Train
  
  ```
  sh semantic_seg/train_s3dis.sh DATASET_S3DIS_PATH
  ```

### Citation:
```
  @article{cheng2021sspc,
  title={SSPC-Net: Semi-supervised Semantic 3D Point Cloud Segmentation Network},
  author={Cheng, Mingmei and Hui, Le and Xie, Jin and Yang, Jian},
  booktitle={AAAI},
  year={2021}
}
```

### Acknowledgement
Our code refers to [SPGraph](https://github.com/loicland/superpoint_graph/tree/release).
