# 🌿 EcoTransformer

We propose a new Transformer architecture "EcoTransformer", replacing dot-product attention with a convolutional mechanism using Laplacian kernels, the L1 metric between queries and keys, eliminating costly matrix multiplication operations. 

The typical scaled dot-product attention mechanism used throughout contemporary AI is computationally expensive and consumes a significant  amount of energy. Our new attention score calculation is lightweight and efficient: it removes resource-hungry multiplications, while performing on par with, or even surpassing scaled dot-product attention in NLP, bioinformatics, and vision tasks. 

## Structure

``l1_distance.py`` contains the main functionality and required helper functions to use our EcoTransformer as a library. 

As we in part tested our architecture against the typical eager_attention_forward used in GPT2, classes inheriting from GPT2 have been provided, to demonstrate how EcoTransformer's L1 attention can replace dot-product attention at minimal cost.

``EcoTransformer_GPT2_Demo.ipynb`` contains a training/testing framework that can demonstrate EcoTransformer's performance on the StoryCloze dataset.

## Setup and Testing

To test the performance of EcoTransformer:
1. Upload ``EcoTransformer_GPT2_Demo.ipynb`` to Google Colab (recommended).
2. Follow the setup steps in the provided notebook, and click "Run All" with the A100 GPU for best performance, or T4 GPU if the former is unavailable. 

To use this library as the core of a repo, run the following commands in a bash terminal:
```
git clone https://github.com/xingmingxu/EcoTransformer_Temp.git
cd EcoTransformer_Temp
```
Create a fresh environment.
```
python -m venv .venv
```
Activate the virtual environment (the following demonstrates Linux, MacOS)
```
source .venv/bin/activate # in Linux, MacOS
```
Install the requirements.
```
pip install requirements.txt
```

This will provide you with a clean repo with l1_distance.py as a core library to build off of. 

## Datasets

We tested our model on the following NLP, Biological, and Vision datasets:
* **NLP**: SciQ, StoryCloze, HellaSwag, BoolQ
* **Biological/Vision**: TCGA, METABRIC, VDJdb, CIFAR-10

## Our Results

Our EcoTransformer performs on-par with, or even surpassing scaled dot-product attention on the datasets described above. 

![alt text](images/table1.png)
![alt text](images/table2.png)

## Credits

We used HuggingFace's [transformers](https://github.com/huggingface/transformers) library to develop and test EcoTransformer.