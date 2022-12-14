## Bootstrapping meaning through listening   
This repo hosts the code necessary to replicate the paper [*Bootstrapping meaning through listening: Unsupervised learning of spoken sentence embeddings*](https://arxiv.org/abs/2210.12857) (Findings of EMNLP 2022). 

### Abstract  
Inducing semantic representations directly from speech signals is a highly challenging task but has many useful applications for speech mining and spoken language understanding. This study tackles the unsupervised learning of semantic representations for spoken utterances. Through converting speech signals into hidden units generated from acoustic unit discovery, we propose WavEmbed, a multimodal sequential autoencoder that predicts hidden units from a dense representation of speech. Secondly, we also propose S-HuBERT to induce meaning through knowledge distillation, in which a sentence embedding model is first trained on hidden units and passes its knowledge to a speech encoder through contrastive learning. The best performing model achieves a moderate correlation (0.5~0.6) with human judgments, without relying on any labels or transcriptions. Furthermore, these models can also be easily extended to leverage textual transcriptions of speech to learn much better speech embeddings that are strongly correlated with human annotations. Our proposed methods are applicable to the development of purely data-driven systems for speech mining, indexing and search.


### Usage
#### Speech models
```
# load dataset
import torch.nn.funtional as F
from datasets import load_dataset

dataset = load_dataset("charsiu/Common_voice_sentence_similarity")

dataset['dev']

# load audio preprocessor
speech_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=speech_tokenizer)

speech_a = processor(dataset['dev'][0]["audio_a"], sampling_rate=16000,return_tensors='pt').input_values
speech_b = processor(dataset['dev'][0]["audio_b"], sampling_rate=16000,return_tensors='pt').input_values

# Load a pretrained WavEmbed model

from WavEmbed import WavEmbedModel

model = WavEmbedModel.from_pretrained("charsiu/WavEmbed_100")
model.eval()

# calculate the semantic similarity between two speech utterance using WavEmbed
with torch.no_grad():
    emb_a = model(speech_a).encoder_last_hidden_state
    emb_b = model(speech_b).encoder_last_hidden_state

similarity = F.cosine_similarity(emb_a,emb_b,dim=-1)
                

# Load a pretrained S-HuBERT model

from SentHuBERT import SentHuBERT

model = SentHuBERT.from_pretrained('charsiu/S-HuBERT-from-simcse-sup-roberta')
model.eval()

# calculate the semantic similarity between two speech utterance using S-HuBERT
with torch.no_grad():
    emb_a = model(speech_a).last_hidden_state
    emb_b = model(speech_b).last_hidden_state
    
similarity = F.cosine_similarity(emb_a,emb_b,dim=-1)
```

#### Hidden unit models

```
#Load a pretrained BERT model trained on hidden units

from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("charsiu/Bert_base_hidden_unit_HuBERT100C")



# Load a pretrained TADAE model trained on hidden units

from TSDAE import EncoderDecoderModel

model = EncoderDecoderModel.from_pretrained('charsiu/TSDAE_hidden_units_HuBert100')
```

#### Training
See scripts in [src/](https://github.com/lingjzhu/spoken_sent_embedding/tree/main/src) for details.

### Pretrained checkpoints  
  Selected pretrained models can be found [here](https://huggingface.co/charsiu).  

### Pretrained tokenizers for hidden units
Pretrained tokenizers can be found in [tokenizers/](https://github.com/lingjzhu/spoken_sent_embedding/tree/main/tokenizers)

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('tokenizers/tokenizers-50') 
# Other pretrained hidden unit tokenizers: 'tokenizers/tokenizers-100', 'tokenizers/tokenizers-200' 

```

```
import sentencepiece as spm

spm_tokenizer = spm.SentencePieceProcessor(model_file='hubert-base-ls960-100_8000.model')
```

### Data  

#### Common Voice Spoken Sentence Similarities  
You can directly load the dataset from [our HuggingFace hub](https://huggingface.co/datasets/charsiu/Common_voice_sentence_similarity) via the following code. Only average semantic similarity ratings are included in the dataset object. However, we also provide individual ratings from all four annotators in the same repositry ([dev_ratings.csv](https://huggingface.co/datasets/charsiu/Common_voice_sentence_similarity/blob/main/dev_ratings.tsv) and [test_ratings.csv](https://huggingface.co/datasets/charsiu/Common_voice_sentence_similarity/blob/main/test_ratings.tsv)). 
```
from datasets import load_dataset

dataset = load_dataset("charsiu/Common_voice_sentence_similarity")

# dev data
dataset['dev']
# test data
dataset['test']
```  
    
#### Hidden-unit transcripts of the Common Voice English subset  
Each split of data is generated by a model from [textlesslib](https://github.com/facebookresearch/textlesslib).  
```
from datasets import load_dataset

dataset = load_dataset("charsiu/Common_voice_hidden_units")
```
#### Spoken STS     
Spoken STS can be accessed [here](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:237533). If you use the Spoken STS dataset, please cite the [Interspeech paper by Merkx et al.](https://www.isca-speech.org/archive/interspeech_2021/merkx21_interspeech.html).
```
@inproceedings{merkx21_interspeech,
author={Danny Merkx and Stefan L. Frank and Mirjam Ernestus},
title={{Semantic Sentence Similarity: Size does not Always Matter}},
year=2021,
booktitle={Proc. Interspeech 2021},
pages={4393--4397},
doi={10.21437/Interspeech.2021-1464}
}
```
### Citation  
  For now, you can cite this paper as:
   ```
   @article{zhu2022bootstrapping,
        title={Bootstrapping meaning through listening: Unsupervised learning of spoken sentence embeddings},
        author={Zhu, Jian and Tian, Zuoyu and Liu, Yadong and Zhang, Cong and Lo, Chia-wen},
        journal={Findings of the Association for Computational Linguistics: EMNLP 2022},
        year={2022}
      }
   ```
### References
  Some of our scripts are based on the following implementations. 
  - [Transformers](https://github.com/huggingface/transformers)
  - [textlesslib](https://github.com/facebookresearch/textlesslib)
  - [SimCSE](https://github.com/princeton-nlp/SimCSE)
  - [sentence-transformer](https://www.sbert.net/)
  - [SentencePiece](https://github.com/google/sentencepiece)
