### Vocabulary for Machine Translation
This is the folder to keep vocabularies for different languages.

### Vocabulary Format
Mapping from tokens to indicies (json, yaml, ...):
```
{token1: index1, token2: index2, ...}
```

### Create Vocabulary
To create vocabularies using *Moses tokenizer* run the following command:
```
python moses_vocab.py
```

At the beginning of the script there is a list of parameters (written in uppercase) that can be changed.
