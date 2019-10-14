# CMAC

Code for paper [*Cross-Modal Commentator: Automatic Machine Commenting Based on Cross-Modal Information*](https://www.aclweb.org/anthology/P19-1257/) （ACL 2019 Oral Paper)

## About CMAC

**Cross Modal Automatic Commenting (CMAC)** is a new task proposed in our paper, which aims to automatically generate comments for graphic news. In this task, AI models are required to integrate the information from both news images and news articles, and generate a reasonable comment regarding to the visual and textual contents.

## Requirements

pytorch 1.1.0

python>=3.6

numpy>=1.16

## Dataset

The processed dataset can be found in [Google Drive](https://drive.google.com/drive/folders/1MmjiO5S8-nTU-vC-yxFEX01oTh4aaqVn?usp=sharing). ``dict_50000.json`` is the dictionary file collected from the training set. ``*.img`` files are processed images by pretrained ResNet. ``*.json`` files are the corresponding texts.

## Files

modules.py: neural network modules for the proposed model.
transformer.py: model definition, training and testing codes for the proposed model (Transformer version).

## Training & Testing

Please make sure that your data is located in ``data/`` under the project directory.

Train a new model:

```bash
python3 transformer.py -mode train -dir CKPT_DIR
```

where ``CKPT_DIR`` is the directory where you want to store your checkpoints.

We apply early stopping to the training process. The best checkpoint on the validation set as well as the checkpoints during the early stopping period will be stored. A complete list of command-line arguments can be found in the beginning of ``transformer.py``.

Test a trained model:

```bash
python3 transformer.py -mode test -dir CKPT_DIR -restore CKPT_DIR/MODEL_PATH
```

where ``MODEL_PATH`` is the file name of the trained model. The output will be stored in ``prediction.json`` by default. You can change this option using the ``-output`` argument.

## Cite

If you find the CMAC task or the dataset interesting, please kindly cite our paper:

```
  @inproceedings{yang2019cross,
  title={Cross-Modal Commentator: Automatic Machine Commenting Based on Cross-Modal Information},
  author={Yang, Pengcheng and Zhang, Zhihan and Luo, Fuli and Li, Lei and Huang, Chengyang and Sun, Xu},
  booktitle={Proceedings of the 57th Conference of the Association for Computational Linguistics},
  pages={2680--2686},
  year={2019}
}
```