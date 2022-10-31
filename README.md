## Bootstrapping meaning through listening   
This repo hosts the code necessary to replicate the paper [*Bootstrapping meaning through listening: Unsupervised learning of spoken sentence embeddings*](https://arxiv.org/abs/2210.12857) (To appear in Findings of EMNLP 2022). 

### Abstract  
Inducing semantic representations directly from speech signals is a highly challenging task but has many useful applications for speech mining and spoken language understanding. This study tackles the unsupervised learning of semantic representations for spoken utterances. Through converting speech signals into hidden units generated from acoustic unit discovery, we propose WavEmbed, a multimodal sequential autoencoder that predicts hidden units from a dense representation of speech. Secondly, we also propose S-HuBERT to induce meaning through knowledge distillation, in which a sentence embedding model is first trained on hidden units and passes its knowledge to a speech encoder through contrastive learning. The best performing model achieves a moderate correlation (0.5~0.6) with human judgments, without relying on any labels or transcriptions. Furthermore, these models can also be easily extended to leverage textual transcriptions of speech to learn much better speech embeddings that are strongly correlated with human annotations. Our proposed methods are applicable to the development of purely data-driven systems for speech mining, indexing and search.

### Contents  
- Usage
- Pretrained checkpoints
- Data
  - **Common Voce Spoken Sentence Similarities**  
    You can directly load the dataset from [our HuggingFace hub](https://huggingface.co/datasets/charsiu/Common_voice_sentence_similarity) via the following code. Only average semantic similarity ratings are included in the dataset object. However, we also provide individual ratings from all four annotators in the same repositry ([dev_ratings.csv](https://huggingface.co/datasets/charsiu/Common_voice_sentence_similarity/blob/main/dev_ratings.tsv) and [test_ratings.csv](https://huggingface.co/datasets/charsiu/Common_voice_sentence_similarity/blob/main/test_ratings.tsv)). 
    ```
    from datasets import load_dataset

    dataset = load_dataset("charsiu/Common_voice_sentence_similarity")
    
    # dev data
    dataset['dev']
    # test data
    dataset['test']
    ```
  - **Spoken STS**  
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
- Citation
- License
- References
  Some of our scripts are based on the following implementations. 
  - [Transformers](https://github.com/huggingface/transformers)
  - [textlesslib](https://github.com/facebookresearch/textlesslib)
  - [SimCSE](https://github.com/princeton-nlp/SimCSE)
  - [sentence-transformer](https://www.sbert.net/)
  - [SentencePiece](https://github.com/google/sentencepiece)
