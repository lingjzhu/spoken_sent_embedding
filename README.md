## Bootstrapping meaning through listening   
This repo hosts the code necessary to replicate the paper *Bootstrapping meaning through listening: Unsupervised learning of spoken sentence embeddings* (To appear in Findings of EMNLP 2022). 

### Abstract  
Inducing semantic representations directly from speech signals is a highly challenging task but has many useful applications for speech mining and spoken language understanding. This study tackles the unsupervised learning of semantic representations for spoken utterances. Through converting speech signals into hidden units generated from acoustic unit discovery, we propose WavEmbed, a multimodal sequential autoencoder that predicts hidden units from a dense representation of speech. Secondly, we also propose S-HuBERT to induce meaning through knowledge distillation, in which a sentence embedding model is first trained on hidden units and passes its knowledge to a speech encoder through contrastive learning. The best performing model achieves a moderate correlation (0.5~0.6) with human judgments, without relying on any labels or transcriptions. Furthermore, these models can also be easily extended to leverage textual transcriptions of speech to learn much better speech embeddings that are strongly correlated with human annotations. Our proposed methods are applicable to the development of purely data-driven systems for speech mining, indexing and search.

### Contents  
- Usage
- Pretrained checkpoints
- Data
- Citation
- License
- References
  Some of our scripts are based on the following implementations. 
  - [Tramsformers](https://github.com/huggingface/transformers)
  - [textlesslib](https://github.com/facebookresearch/textlesslib)
  - [SimCSE](https://github.com/princeton-nlp/SimCSE)
  - [sentence-transformer](https://www.sbert.net/)
  - [SentencePiece](https://github.com/google/sentencepiece)
