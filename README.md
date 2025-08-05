# 🌿 EcoTransformer

We propose a new Transformer architecture "**EcoTransformer**", in which the output context vector is constructed as the convolution of the values using a Laplacian kernel, where the distances are measured by the *L1 metric* between the queries and keys. Our architecture eliminates costly matrix multiplication operations for attention scores, paving the way for a **less energy-intensive** Transformer. 

The typical scaled dot-product attention mechanism used throughout contemporary AI is computationally expensive and consumes a significant  amount of energy. Our new attention score calculation is **lightweight** and **efficient**: it removes resource-hungry multiplications, while performing on par with, or even surpassing scaled dot-product attention in NLP, bioinformatics, and vision tasks. 

Check out our paper [here](https://arxiv.org/pdf/2507.20096)!!

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

## License

MIT License

Copyright (c) [2025] [Xin Gao]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.
