# Master_thesis_DVlog
Repository for the code surrounding my Master Thesis


## ⏳ Installation
An yaml file has been supplied to retrieve all the dependencies
```bash
conda env create -f environment.yml
conda activate dvlog_env
```

## 👩‍💻 Usage
### ⚙️ The configuration system
To improve reproducibility, the [`confection`](https://github.com/explosion/confection) library is used which offers a configuration system by parsing `.cfg` files and resolving it to a dictionary. The following [`file`](https://github.com/StanMey/Master_thesis_DVlog/blob/main/example.cfg) holds the template for setting up these configuration files needed for training which look like

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

## 🏋️‍♂️ Training the models

### Straightforward models

### ⏱️ Synchronised model


### 📊 Evaluating the models
