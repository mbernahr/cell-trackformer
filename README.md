# Project for cell tracking with TrackFormer
This is the project for the Computer Vision course in the summer semester 2024 at the University of Cologne. The aim of the project was to create a model that tracks the location of cells across a sequence of images.

## Customisation and experimentation
The [TrackFormer](https://github.com/timmeinhardt/trackformer/tree/main) model has been adapted to capture and analyse the dynamics and specific movement patterns of cells in sequential microscopic images. At the core of the system, ‘object queries’ are used within a transformer decoder that allow the identities and positions of cells to be continuously tracked across multiple frames.

In addition, we experimented with different backbone architectures to optimise the effectiveness of our cell tracking. Among the models tested were ResNet-50 and ResNet-101, which are used by default for deep image analysis tasks. We also used the ResNet50-dc5 ,as suggested in the TrackFormer paper, as a ResNet50 network that has been finetuned using the BYOL algorithm to improve its image representation abilities in our specific domain.

## Repository structure
The ```trackformer-main``` folder contains all python files we had to adapt in order to get the original trackformer code running with our data. In addition, the repository contains several special Jupyter notebooks that were used for different tasks in the project process:

```cv_data_preprocessing.ipynb```: 
contains the code for data preparation and preprocessing.

```cv_create_custom_backbone.ipynb```:
Another notebook in which we apply BYOL to a ResNet50 Network to finetune it using images sampled from our dataset.

```cv_cell_trackformer_backbone.ipynb```:
Contains the code for finetunning/pretraining the trackformer on our data using different backbones.

```cv_cell_trackformer_layers.ipynb```:
Contains the code for finetuning the trackformer on our data using different layer amounts.

```cv_data_visualisation.ipynb```:
Contains the code for visualising the tracking results.

Due to the large dataset and the custom modifications made to the code, the project was uploaded to Google Drive. This allows us to manage and execute the project directly from there, utilizing Google Colab’s integration with Google Drive for efficient access to files and seamless execution.

## Requirements
### Adaptation to Python 3.10
Due to the original development of TrackFormer for Python 3.7 and the use of Python 3.10 in Google Colab, it was necessary to adapt the requirements. The updated dependencies are listed in the ```requirements-3.10.txt``` file and can be installed as follows once drive is mounted and the updated requirements are uploaded to the according drive:
```py
!pip install -r /content/drive/MyDrive/requirements-3-10.txt
!pip install sacred --upgrade
```

### Package Compatibility Adjustments
Some packages required updates due to compatibility issues between different Python versions. Specific adjustments were needed for ```torch``` and ```torchvision```, as well as the installation of a customized version of ```cocoapi```:
```py
!pip install torch==2.4.0 torchvision==0.19.0
!pip3 install -U 'git+https://github.com/timmeinhardt/cocoapi.git#subdirectory=PythonAPI'
```

### Updating Deprecated Python Snippets
To ensure compatibility with Python 3.10, outdated Python snippets in various libraries also needed to be updated. This mainly involved adjusting data types in line with the latest versions of ```numpy```. A script for making these changes in the filesystem is as follows:
```py
import os

def replace_deprecated_python_types(directory, old_type, new_type):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()

                new_content = content.replace(old_type, new_type)

                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as file:
                        file.write(new_content)
                    print(f"Updated {filepath}")

directory = "/usr/local/lib/python3.10/dist-packages/pycocotools"
replace_deprecated_python_types(directory, 'np.float', 'float')

directory = "/usr/local/lib/python3.10/dist-packages/motmetrics"
replace_deprecated_python_types(directory, 'np.float', 'float')
replace_deprecated_python_types(directory, 'np.bool', 'bool')
```

These measures ensure that the project runs smoothly in the Colab environment with Python 3.10.

## Result

The result of the finetuned TrackFormer model optimised for cell tracking. The original detections and the predictions of the model are shown.

<div align="center">
    <img src="test_sequence_with_annotations_and_detections.gif" alt="Snakeboard demo" width="600"/>
</div>


