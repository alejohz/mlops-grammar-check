# mlops-grammar-check
Personal project to showcase the use of some MLOps techniques, tools and practices.

Bert based model to check grammar in a sentence. The model was trained using the [CoLA dataset](https://nyu-mll.github.io/CoLA/).

This repo uses python 3.11.4

### Wandb report
This repo used Weight & Bias for model tracking and experiment management. The report can be found [here](https://wandb.ai/alejohz/mlops-testing/reports/Debugging-Bert-Cola-model--Vmlldzo1MjgyNDA1).

### Hydra config
This repo uses [Hydra](https://hydra.cc/) for configuration management. The config files can be found in the `configs` folder.


### DVC
This repo uses [DVC](https://dvc.org/) for model versioning and remote storage of trained models. The data can be found in the `models` folder.


### Docker & FastAPI
This repo uses Docker and FastAPI to deploy a usable model into a very basic a simple API made with FastAPI.
