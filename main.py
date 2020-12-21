from torch.utils.data import DataLoader

from dataset import WMTCollator, WMTDataset
from utils import set_global_seed

# hyperparams
SEED = 42
BATCH_SIZE = 16
PERCENTILE = 100

set_global_seed(SEED)

dataset = WMTDataset(
    from_lang_data_path="data/WMT14_English_German/train.en",
    to_lang_data_path="data/WMT14_English_German/train.de",
    from_lang_tokenizer_path="tokenizer/en.model",
    to_lang_tokenizer_path="tokenizer/de.model",
    verbose=True,
)

collator = WMTCollator(
    from_lang_padding_value=3,
    to_lang_padding_value=3,
    percentile=PERCENTILE,
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)
