# CBEN
Code &amp; and Dataset for our publication CBEN -- A Multimodal Machine Learning Dataset for Cloud Robust Remote Sensing Image Understanding. Dataset is available at [IEEE dataport](https://ieee-dataport.org/documents/cben-multimodal-machine-learning-dataset-cloud-robust-remote-sensing-image-understanding) and the [Internet Archive](https://archive.org/details/cben_20260217_20260217)

Code has been adapted from [Wang, Yi, et al. "SSL4EO-S12: A large-scale multimodal, multitemporal dataset for self-supervised learning in Earth observation [Software and Data Sets]." IEEE Geoscience and Remote Sensing Magazine 11.3 (2023): 98-106.](https://github.com/zhu-xlab/SSL4EO-S12?tab=readme-ov-file)

## Dataset

Code to run the dataset can be found at: `data_processor/create_cloudy_ben.py`

## SSL

Code to run the self supervised learning part is here: `exp\ssl_cloud\proof_of_benefit\backbones`, where each file corresponds to the respective SSL method.

## Downstream

Code to run the finetuning downstream part is here: `exp\ssl_cloud\proof_of_benefit\backbones`, where each file corresponds to the respective SSL method.
