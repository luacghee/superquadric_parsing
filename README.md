## 

This repository consist of the submission for ME6401 Topics in Mechatronics I Robot Imagination. The contribution of this assignment extends the work of 
[Superquadrics Revisited: Learning 3D Shape Parsing beyond Cuboids](https://arxiv.org/pdf/1904.09970.pdf) with a modified RGB-D encoder from [RedNet: Residual Encoder-Decoder Network for indoor RGB-D Semantic Segmentation](https://arxiv.org/pdf/1806.01054.pdf). Most of the code for this assignment were forked from the former publication. As of now, the results only evaluates on the [ShapeNet](https://shapenet.org/) dataset.


Dependencies & Installation
----------------------------
Installation follows the original repository instructions.

They should be automatically installed by running
```
pip install --user -e .
```

In case this doesn't work automatically try this instead,
```
pip install -r requirements.txt
pip install --user -e .
```

Please note that you might need to install `python-qt4` in order to be able to
use mayavi. You can do that by simply typing
```
sudo apt install python-qt4
```
Now you are ready to start playing with the code!


Training
--------

To train a new network, go to the scripts folder and execute

```
$ ./train_network.py /path_to_training_dataset ../trained_models --architecture rednet --voxelizer_factory image_rgbd --n_primitives 20 --train_with_bernoulli --use_sq --lr 1e-4 --dataset_type shapenet_v2 --use_chamfer --run_on_gpu --use_deformations 
```

You need to indicate the training dataset directory and a output directory (here: ../trained_models) to store the generated files. The script will automatically create a subfolder in the designated output directory. Within this subfolder, it stores the trained models, three .txt files documenting the loss evolution, and a .json file containing the training parameters. The default encoder architecture is the tulsiani network. To train with the RGB-D encoder, use the following arguments --architecture rednet --voxelizer_factory image_rgbd.


Evaluation
----------

To evaluate a trained network, go to the scripts folder and execute

```
$ ./forward_pass.py ../demo/02691156_rgbd /tmp/ --model_tag "1a29042e20ab6f005e9e2656aff7dd5b" --weight_file ../trained_models/model_name/model_number --n_primitives 20 --train_with_bernoulli --use_sq --dataset_type shapenet_v2 --use_deformations --architecture rednet --voxelizer_factory image_rgbd
```

You need to indicate the evaluation dataset directory (here: ../demo/02691156_rgbd) and the path to save the output of the evaluation (here: /tmp/). Also, specify the model tag you wish to reconstruct --model_tag and the dataset type in use --dataset_type. Additionally, you need to provide the weights of the trained model through the --weight_file argument. It is important to note that you should furnish the same arguments used during the training phase.


Chamfer Loss and Volumetric IOU
----------

To quantitatively assess a trained network against an entire ShapeNet cateogry, go to the scripts folder and execute

```
$ ./eval_model.py /path_to_ShapeNet_category  /tmp/ --weight_file ../trained_models/model_name/model_number --n_primitives 20 --train_with_bernoulli --use_sq --dataset_type shapenet_v2 --use_deformations --architecture rednet --voxelizer_factory image_rgbd
```

You need to indicate the Shapenet category directory and the path to save the output of the evaluation (here: /tmp/). Also, specify the model tag you wish to reconstruct --model_tag and the dataset type in use --dataset_type. Additionally, you need to provide the weights of the trained model through the --weight_file argument. It is important to note that you should furnish the same arguments used during the training phase.
