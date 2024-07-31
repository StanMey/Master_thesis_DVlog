# Master_thesis_DVlog
Welcome to the repository for the code supporting my Master Thesis titled,
> “Use your words”: Towards Gender Fairness for Multimodal Depression Detection.

where gender fairness in multimodal depression detection gets explored using the D-Vlog dataset. This study works further on the [D-vlog: Multimodal Vlog Dataset for Depression Detection](https://ojs.aaai.org/index.php/AAAI/article/view/21483) paper.

## Overview
This repository contains all the necessary code and configurations for the experiments and models developed during my Master Thesis.

### ⏳ Installation
Both a yaml file for a CPU and GPU environment have been supplied which will retrieve all the dependencies. 
```bash
conda env create -f cpu_environment.yml
conda activate dvlog_cpu_env
```

## 👩‍💻 Usage
### ⚙️ The configuration system
To improve reproducibility, the [`confection`](https://github.com/explosion/confection) library is used which offers a configuration system by parsing `.cfg` files and resolving it to a dictionary. The following [`file`](https://github.com/StanMey/Master_thesis_DVlog/blob/main/example.cfg) holds the template for setting up these configuration files needed for training which rougly look like

```ini
[general]
model_name = ""

[paths]
annotations_file = ""
data_dir = ""

[training]
epochs = 50
batch_size = 32
learning_rate = 0.0002
seed = 42
```

This also means that each experiment is saved as a separate config file in the [model configs](https://github.com/StanMey/Master_thesis_DVlog/tree/main/DVlog/model_configs) folder.

## 🏋️‍♂️ Training the models


### 📏 Straightforward models

### ⏱️ Synchronised model
The sentence-based features typically spanned multiple seconds, whereas both the original audio and video embeddings used fixed seconds-based representations. These modalities needed to be synchronised and to synchronise these modalities, the start and end timestamps for each sentence were extracted.

These timestamps allowed us to calculate the averages of the seconds-based features for each
sentence. Specifically, we aligned the timestamps of the sentences with the corresponding seconds-based features and computed the mean values of the features within those segments. A visual example of this synchronisation process is shown below.

![Synchronisation example](figures/synchronisation_example.PNG "Synchronisation example")


### 📊 Evaluating the models
