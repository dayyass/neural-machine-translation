### About
Pipeline for training [Stanford Seq2Seq Neural Machine Translation](https://paperswithcode.com/paper/stanford-neural-machine-translation-systems) using PyTorch.<br/>
Model trained on [IWSLT'15 English-Vietnamese](https://nlp.stanford.edu/projects/nmt/).<br/>
State-of-the-art on IWSLT'15 English-Vietnamese [reference](https://paperswithcode.com/sota/machine-translation-on-iwslt2015-english-1).

### Usage
First, install dependencies:
```
# clone repo
git clone https://github.com/dayyass/neural_machine_translation.git

# install dependencies
cd neural_machine_translation
pip install -r requirements.txt
```

### Data Format
Parallel corpora for Machine Translation.<br/>
More about it [here](data/README.md).

### Vocabulary
Before train any models, you need to create vocabularies for two languages.<br/>
More about it [here](vocab/README.md).

### Training
Train Neural Machine Translation:
```
python train.py
```

At the beginning of the script there is a list of parameters (written in uppercase) for training that can be changed.<br/>
Validation performed on every epoch, testing performed after the last epoch.

### Validation
**NotImplementedError**: opened [issue](https://github.com/dayyass/neural_machine_translation/issues/9).

### Inference
**NotImplementedError**: opened [issue](https://github.com/dayyass/neural_machine_translation/issues/3).

### Models
List of implemented models:
- [x] [Seq2SeqModel](https://github.com/dayyass/neural_machine_translation/blob/d2cdb4fdb7629ba9cec42e6f7e87915aff682f19/network.py#L91)
- [ ] Seq2SeqAttnModel
