# Efficient 3D Dose Generation with Latent Diffusion

### Description
Radiotherapy dose calculation requires many intermediate time-consuming steps until
delivery, while itself can also be quite time-consuming and challenging. Deep Learning
and Generative Modeling have shown great potential in creating real-world usable doses.
We provide a framework to automate the configuration, training and evaluation of 
efficient 3D dose generation models based on Latent Diffusion. We experiment with using
the mask labels for conditioning together with the images, and with simultaneous mask
and dose generation, conditioned only with images.

This project is inspired from [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and heavily based on 
[MONAI generative models](https://github.com/Project-MONAI/GenerativeModels). Given a 
dataset, nnU-Net automatically configures all the hyperparamaters that should be used 
for this dataset on a segmentation task. We follow the same idea to define robust enough
hyperparameters for the task of training diffusion-latent diffusion models for dose and
mask generation based on MONAI generative models.

### Requirements
- python 3.11.12, cuda 12.5, and at least one GPU with 24GB memory

### Installation

If you want to use this project as a python library:

- Install [pytorch](https://pytorch.org/get-started/locally/) 
- pip install dosegen

If you want to further develop this project:
1. clone the repository
2. create your virtual environment
3. install [pytorch](https://pytorch.org/get-started/locally/)
5. pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
3. in the environment run: pip install -e .

[//]: # (- If pip doesn't work:)

[//]: # (  - Clone the repository )

[//]: # (  - You can try installing the requirements.txt, but if this doesn't work:)

[//]: # ()
[//]: # (    - Install pytorch following the official [pytorch )

[//]: # (    installation guide]&#40;https://pytorch.org/get-started/locally/&#41;.)

[//]: # ()
[//]: # (    - Install the following libraries with pip:)

[//]: # (      - pip install pyyaml matplotlib tqdm nibabel scikit-image monai )

[//]: # (      monai-generative nnunet lpips xformers torchinfo)

[//]: # ()
[//]: # (  - &#40;Optional&#41; You can install these libraries also for jupyter notebooks and)

[//]: # (  interactive visualization:)

[//]: # (    - pip install jupyter matplotlib ipywidgets ipympl notebook tornado)

[//]: # (  - run pip install -e . when you are in the main directory)
 

## Usage Instructions

### Environment variables

First, set some environment variables for the main dataset results folders:

```bash
export dosegen_preprocessed="/path_to_your_folder/dosegen_preprocessed"
export dosegen_results="/path_to_your_folder/dosegen_results"
```
All the preprocessed folders of this project will be stored in dosegen_preprocessed,
and similarly all the results in dosegen_results.


### Dataset preparation

- We follow initial dataset formats similar the [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- To train models with this library you must put all your files in a folder with a name
of the form: TaskXXX_DatasetName, where XXX should be a 3-digit number, potentially
starting with 0s. 
- All the images, doses and label masks should be in .nii.gz format 
- The images should be stored in a folder called 'images', the doses in one called 
'doses' and the masks in a folder called 'labels'.

[//]: # (You must create a dataset based on nnU-Net conventions. You can start with )

[//]: # (a dataset which follows the [Medical Segmentation Decathlon]&#40;http://medicaldecathlon.com/&#41; format, where)

[//]: # (all training images are contained in a folder called **imagesTr**, and are compressed )

[//]: # (nifti files &#40;.nii.gz&#41;, and convert the dataset to nnUNetv2 format with:)

[//]: # ()
[//]: # (```bash)

[//]: # (nnUNetv2_convert_MSD_dataset -i /path_to_original_dataset/Task01_MyDataset)

[//]: # (```)

[//]: # ()
[//]: # (This should create a dataset in the nnUNet_raw folder called Dataset001_MyDataset, )

[//]: # (splitting multiple channel images to separate images. For other available dataset )

[//]: # (format options take a look at nnUNet documentation.)

### Dataset preprocessing

To preprocess a dataset and make it ready for latent diffusion training run:

```bash
dosegen_plan_and_preprocess /path_to_dataset
```

This will create a new folder in your dosegen_preprocessed folder containing:
- .zarr files with the images, doses and masks
- a dosegen.yaml configuration file with all the parameters/hyperparameters of the 
project

[//]: # (Given a new dataset, nnU-Net will extract a dataset fingerprint &#40;a set of )

[//]: # (dataset-specific properties such as image sizes, voxel spacings, intensity )

[//]: # (information etc&#41;. This information is used to design three U-Net configurations. )

[//]: # (Each of these pipelines operates on its own preprocessed version of the dataset.)

[//]: # ()
[//]: # (The easiest way to run fingerprint extraction, experiment planning and )

[//]: # (preprocessing is to use:)

[//]: # ()
[//]: # (```bash)

[//]: # (nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity)

[//]: # (```)

[//]: # ()
[//]: # (This will create a new subfolder in your nnUNet_preprocessed folder named after the )

[//]: # (dataset. All the images will be cropped to non-zero regions, resampled to the median voxel )

[//]: # (spacing of the dataset, and depending on the image modality other processes should )

[//]: # (be applied &#40;e.g., for MRI images a z-score normalization will be used&#41;. For more )

[//]: # (information check nnU-Net documentation.)


### Training

All the training commands include these arguments:
- DATASET_ID: corresponds to the numeric dataset identifier (e.g., 001 for Brain Tumour
dataset from Medical Segmentation Decathlon)
- SPLITTING: should be one of ['train-val-test', '5-fold'], defining the type of data 
splitting 
- MODEL_TYPE: should be either '2d' or '3d' 
- -p: indicates that we want to see progress bars.


#### Latent Diffusion Model
To train a Latent Diffusion Model, first we need to train an autoencoder:

```bash
dosegen_train_autoencoder DATASET_ID SPLITTING MODEL_TYPE -p
```
After finishing training, we can then train the Latent Diffusion Model:

```bash
dosegen_train_ldm DATASET_ID SPLITTING MODEL_TYPE -p
```

#### Output Files
Running an experiment will create a directory in dosegen_results path with the 
following folders and files: 
- checkpoints folder: contains the checkpoints of the last and the best epoch of 
training (the best one is derived based on the validation reconstruction loss)
- plots folder: in this folder a loss.png file will be saved for every kind of
training, showing the training and validation losses per epoch. Moreover, a gif will 
be saved in this folder (with frequency based on the argument 'val_plot_interval' 
in the config file) in case of 3D training, and a slice in case of 2D. For ddpm 
and ldm, the figures contain generated examples across the z direction, 
while for the autoencoder, an actual image and its reconstruction are visualized.
- loss_dict.pkl: a file containing lists with loss values per epoch


#### Tips for Training

- To continue training a model, just pass -c when running the training command.

[//]: # (- The autoencoder shouldn't have more than 2-3 downsampling layers, otherwise it )

[//]: # (won't be able to reconstruct details accurately.)

[//]: # (- Only a few convolutional filters for every layer of the autoencoder &#40;e.g., 32&#41;, )

[//]: # (can result in good enough reconstruction performance. )

[//]: # (- Loss weights in the training of the autoencoder are really important. Some works)

[//]: # (might use relatively small loss weights for the perceptual loss &#40;e.g., 0.01&#41; and the)

[//]: # (adversarial loss &#40;e.g., 0.1&#41;, but based on experiments a value of 1, and 0.25, )

[//]: # (respectively, gives much better and realistic results.)

### Sampling (<span style="color:red">*Will be added*</span>)

To sample doses with your model conditioned on images (and labels) run:

```bash
dosegen_sample DATASET_ID MODEL_TYPE NUM_IMAGES SAVE_PATH -p
```
This will sample NUM_IMAGES images and save them in SAVE_PATH.


[//]: # (## ToDos)

[//]: # ()
[//]: # (1. Pass nnUNet configured parameters to medimgen)

[//]: # (   1. create code that reads nnunet file and creates a medimgen config file)

[//]: # (   2. adapt autoencoder + diffusion for flexible architectures)

[//]: # (2. create code to select the additional hyperparameters not involved in nnUNet)

[//]: # (3. Adapt dataset class for 2D and 3D training, nnUNet augmentations, and ideally )

[//]: # (nnUNet patch selection when training &#40;oversampling&#41;.)

[//]: # ()
[//]: # (- Add intensity normalization?)

[//]: # (- Add option to include labels, so that we can train a model to generate labels )

[//]: # (together with images)

[//]: # (- Add efficient implementation of U-Net like in Medical Diffusion)

[//]: # (- Add GANs)

[//]: # (- Ultimate Goal: like nnU-Net, study and come up with heuristics that can be applied)

[//]: # (to multiple datasets and achieve high quality generation. Come up with ways to )

[//]: # (automatically configure every experiment's hyperparameters.)

[//]: # ()
[//]: # (- Experimental: nnUNet works with random cropped patches instead of full images.)

[//]: # (Wouldn't that be awesome to do also in image generation? This would reduce )

[//]: # (computational demands and also increase the training dataset size. We can train)

[//]: # (the autoencoder to output cropped patches and then do sliding window inference, but)

[//]: # (how to perform generation from multiple patches, so multiple latent vectors? IDEA: )

[//]: # (Train the diffusion model to generate latent vectors based on image patches, )

[//]: # (conditioned on latent vectors from image patches around the main patch. On inference,)

[//]: # (start generation with the top left patch conditioned on patches of zeros, and then)

[//]: # (generate patches sequentially based on previously generated patches.)