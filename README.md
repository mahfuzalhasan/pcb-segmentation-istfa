# pcb-image-segmentation
## Dataset
>Will provide the modified split for the subset of [FPIC](https://arxiv.org/abs/2202.08414) soon.
## Framework
PyTorch 1.11.0
## Language
Python 3.7.12
## Task
Segmentation
## Dataloader
1. We are going to provide the image ids for the subset of [FPIC dataset](https://arxiv.org/abs/2202.08414) that we have used soon. This will be the filtered version of the FPIC Dataset. In addition to that we are also going to provide train, validation and test split. <br />
## Training
1. With the provided data split (under data folder), run **python3 train.py**.<br />
2. To run with coarse annotation, need to set bbox_number in the configuration.py and set allow_bbox=True in the parameters.py. To run only plain segmentation, you can just set allow_bbox=False. <br/>
## Testing
1. To test, set *test_model_path* in the configuration.py file. Then run **python3 test.py**
