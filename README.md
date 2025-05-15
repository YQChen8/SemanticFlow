# SemanticFlow: A Self-Supervised Framework for Joint Scene Flow Prediction and Instance Segmentation in Dynamic Environments

**SemanticFlow** is a self-supervised framework for joint scene flow prediction and instance segmentation in dynamic environments. This repository is based on [SeFlow](https://github.com/KTH-RPL/SeFlow) and extends it with semantic and instance segmentation capabilities, making it suitable for autonomous driving and other dynamic scenarios.

## Key Features

- **Self-supervised learning**: No manual labels required, end-to-end training.
- **Joint scene flow and instance segmentation**: Predicts both 3D motion and instance labels for point clouds.
- **Efficient inference**: Supports real-time inference on mainstream GPUs.
- **Compatible with SeFlow data and workflow**: Directly reuses SeFlow's data processing, training, and evaluation scripts.

## Environment Setup

It is recommended to use the same environment as SeFlow:

```bash
git clone --recursive https://github.com/KTH-RPL/SeFlow.git
cd SeFlow && mamba env create -f environment.yaml
```

Compile CUDA dependencies (requires nvcc):

```bash
mamba activate seflow
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

Install the provided PointNet2 module (required for training and inference):

```bash
cd assets/pointnet2 && python setup.py install && cd ../..
```

## Data Preparation

The data processing workflow is the same as SeFlow, supporting Argoverse2, Waymo, and other datasets. Please refer to `dataprocess/README.md` for details on downloading and preprocessing raw data.

For a quick start, you can download the official demo data:

```bash
wget https://zenodo.org/record/12751363/files/demo_data.zip
unzip demo_data.zip -d /home/kin/data/av2
```

## Training

```bash
python train.py model=deflow lr=2e-4 epochs=1000 batch_size=2 loss_fn=seflowLoss add_seloss="{chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0, rigid_loss: 1.0, smooth_loss:0.5, mask_loss:0.0, bf_mask_loss:0.1, fg_consistent:5.0}" model.target.num_iters=2 model.val_monitor=val/Dynamic/Mean
```

## Save Results

```bash
python save.py checkpoint=best.ckpt
```

## Visualization

```bash
python tools/visualization_save.py --res_name 'est_label0'
```

## Evaluation

```bash
python eval.py checkpoint=best.ckpt av2_mode=val
```

## Acknowledgements

This project is developed based on [SeFlow](https://github.com/KTH-RPL/SeFlow). Thanks to the original authors for their open-source contribution.

Feel free to submit issues or PRs if you have any questions.
