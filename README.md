# Introduction
A PyTorch implementation of our ECCV2022 paper titled "[Union-set Multi-source Model Adaptation for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890570.pdf)".
# Requirements
Pytorch version: 1.11.0, CUDA version: 11.1, GPU: one Tesla V100 32GB.
# Datasets
- Source domains:
  - [Synscapes Dataset](https://7dlabs.com/synscapes-overview)
  - [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/)
  - [SYNTHIA Dataset](https://synthia-dataset.net/)
- Target domain:
  - [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

Download the datasets, name the folders as the dataset names (Cityscapes, Synscapes, GTA5, SYNTHIA), and put them in the same folder.
# Training procedure
- Download the [initial weights](https://drive.google.com/file/d/1EkZ0FqmZ8cDp6vWx4J7qv4SoDfYljiUK/view?usp=sharing).
- Train source-domain models with the source-domain data.
```
python train_stage0.py --source two or three of Synscapes, GTA5 and SYNTHIA \
                       --label_setting f or p or n \
                       --data_dir path/to/datasets \
                       --checkpoint_path path/to/save/checkpoints/to
```
- Produce pseudo labels with the source-domain models.
```
python pseudo_label.py --source two or three of Synscapes, GTA5 and SYNTHIA \
                       --label_setting f or p or n \
                       --data_dir path/to/datasets \
                       --stage 1 \
                       --restore_path path/to/checkpoint/path/of/stage0 \
                       --pseudo_label_path path/to/save/pseudo/labels/to
```
- Train models of stage 1.
```
python train_stage1.py --source two or three of Synscapes, GTA5 and SYNTHIA \
                       --label_setting f or p or n \
                       --data_dir path/to/datasets \
                       --checkpoint_path path/to/save/checkpoints/to \
                       --restore_path path/to/checkpoint/path/of/stage0 \
                       --pseudo_label_path path/to/pseudo/labels
```
- Produce pseudo labels with the models of stage 1.
```
python pseudo_label.py --source two or three of Synscapes, GTA5 and SYNTHIA \
                       --label_setting f or p or n \
                       --data_dir path/to/datasets \
                       --stage 2 \
                       --restore_path path/to/checkpoint/path/of/stage1 \
                       --pseudo_label_path path/to/save/pseudo/labels/to
```
- Train the final models of stage 2.
```
python train_stage2.py --source two or three of Synscapes, GTA5 and SYNTHIA \
                       --label_setting f or p or n \
                       --data_dir path/to/datasets \
                       --checkpoint_path path/to/save/checkpoints/to \
                       --restore_path path/to/checkpoint/path/of/stage1 \
                       --pseudo_label_path path/to/pseudo/labels
```
