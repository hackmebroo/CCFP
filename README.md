# Cross Contrasting Feature Perturbation for Domain Generalization (ICCV'23)

Official PyTorch implementation of [Cross Contrasting Feature Perturbation for Domain Generalization](https://arxiv.org/abs/2307.12502).

Chenming Li, Daoan Zhang, Wenjian Huang, Jianguo Zhang

Note that this project is built upon [DomainBed@3fe9d7](https://github.com/facebookresearch/DomainBed).

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Download the datasets:
```sh
python -m domainbed.scripts.download \
       --data_dir=Your_data_dir
```

### Train a model:
```sh
python train.py --data_dir Your_data_dir --test_envs 0 --algorithm CCFP --dataset PACS
```

### Launch a sweep:
```sh
python sweep.py launch --data_dir=Your_data_dir\
       --output_dir=Your_output_dir\
       --command_launcher multi_gpu\
       --algorithms CCFP\
       --datasets PACS\
       --n_hparams 5\
       --n_trials 3
```

### Citation
```plaintext
@inproceedings{li2023cross,
  title={Cross contrasting feature perturbation for domain generalization},
  author={Li, Chenming and Zhang, Daoan and Huang, Wenjian and Zhang, Jianguo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1327--1337},
  year={2023}
}
```

