# DeepOIS: Gyroscope-guided deep optical image stabilizer compensation
## How to generate GT labels

We introduce Essential-Mixtures Model to calculate GT labels, which is effective. However, it requires non-trivial computation resources and is not stable. So we here provide an alternative method called [Homography Mixtures](https://github.com/lhaippp/Homography-Mixtures).

## Model and Data
The pretrain-model, benchmark, trainset could be found at [google-drive](https://drive.google.com/drive/folders/1UeJdy4b2hl3uL2ar1wd2t2w7rKlkC942?usp=sharing)

- Download and Put `GF4-Val.txt` `GF4-Val` to the root path
- Download and Unzip`pretrain-model` to the root path

Here is the [colab](https://colab.research.google.com/drive/1ZJBiqm5E4-ooPmRLWohb0pSMemd5l0Tp?usp=sharing) demo for illustrating the GF4 benchmark

## Inference

`sudo python3 test_demo.py --data_dir TestDatasetItem.npy --model_dir pretrain-model --restore_file best`

## Training

coming soon

## Citation

```latex
@article{liu2021deepois,
  title={DeepOIS: Gyroscope-guided deep optical image stabilizer compensation},
  author={Liu, Shuaicheng and Li, Haipeng and Wang, Zhengning and Wang, Jue and Zhu, Shuyuan and Zeng, Bing},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  volume={32},
  number={5},
  pages={2856--2867},
  year={2021},
  publisher={IEEE}
}
```
