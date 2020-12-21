from dataset import WMTDataset
from utils import set_global_seed

set_global_seed(42)

dataset = WMTDataset(
    from_lang_data_path="data/WMT14_English_German/train.en",
    to_lang_data_path="data/WMT14_English_German/train.de",
    from_lang_tokenizer_path="tokenizer/en.model",
    to_lang_tokenizer_path="tokenizer/de.model",
    verbose=True,
)
