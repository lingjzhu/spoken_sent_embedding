import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader
import soundfile as sf
from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor,AutoTokenizer
from datasets import Dataset
import sentencepiece as spm

@dataclass
class SpeechUnitDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    tokenizer: AutoTokenizer
    sentencepiece_tokenizer: spm.SentencePieceProcessor = None
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 1500
    max_length_labels: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    acoustic_units: Dict = None 
    
    
    def audio_preprocess(self,features):
    
    #    features,sr = sf.read(path)
    #    assert sr == 16000
        return self.processor(features, sampling_rate=16000,return_tensors='pt').input_values.squeeze()


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": self.audio_preprocess(feature["audio"]['array'])} for feature in features]
        
        if not self.sentencepiece_tokenizer:
            input_units = [{"input_ids": self.tokenizer(self.acoustic_units[feature['path']])['input_ids'][:self.max_length_labels]} for feature in features]
        elif self.sentencepiece_tokenizer:
            input_units = [{"input_ids": self.sentencepiece_tokenizer.piece_to_id(['[CLS]']) + self.sentencepiece_tokenizer.encode(self.acoustic_units[feature['path']])[:self.max_length_labels-2] + self.sentencepiece_tokenizer.piece_to_id(['[SEP]'])} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
#        batch_size, raw_sequence_length = batch['input_values'].shape
#        sequence_length = speech_encoder._get_feat_extract_output_lengths(raw_sequence_length)
#        batch['mask_time_indices'] = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.1, mask_length=2)

        
        text_batch = self.tokenizer.pad(
                input_units,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )
        
        return batch, text_batch


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    tokenizer: AutoTokenizer
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    def audio_preprocess(self,features):
    
    #    features,sr = sf.read(path)
    #    assert sr == 16000
        return self.processor(features, sampling_rate=16000,return_tensors='pt').input_values.squeeze()


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": self.audio_preprocess(feature["audio"]['array'])} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
#        batch_size, raw_sequence_length = batch['input_values'].shape
#        sequence_length = speech_encoder._get_feat_extract_output_lengths(raw_sequence_length)
#        batch['mask_time_indices'] = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.1, mask_length=2)

        text_features = [feature["sentence"] for feature in features]
        text_batch = self.tokenizer(text_features,padding=True,return_attention_mask=self.return_attention_mask,return_tensors='pt')
        return batch, text_batch



@dataclass
class EvalDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    output_filename: bool = False
    
    def audio_preprocess(self,features):
    
    #    features,sr = sf.read(path)
    #    assert sr == 16000
        return self.processor(features, sampling_rate=16000,return_tensors='pt').input_values.squeeze()


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_a = [self.audio_preprocess(feature["audio_a"]) for feature in features]
        input_b = [self.audio_preprocess(feature["audio_b"]) for feature in features]
        
        if self.output_filename:
            names = [feature['path_a']+'----'+feature['path_b'] for feature in features]
    
        input_features = {'input_values':input_a+input_b}
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        
        scores = [feature['similarity'] for feature in features]
        if not self.output_filename:
            return batch, scores
        else:
            return batch, scores, names
        
        
@dataclass
class STSEvalDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    def audio_preprocess(self,path):
    
        features,sr = sf.read(path)
        assert sr == 16000
        return self.processor(features, sampling_rate=16000,return_tensors='pt').input_values.squeeze()


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_a = [self.audio_preprocess(utt) for feature in features for utt in feature['0']]
        input_b = [self.audio_preprocess(utt) for feature in features for utt in feature['1']]
        
        tasks = [feature['task'] for feature in features]
        entry = [feature['entry'] for feature in features] 
        
        batch_a = self.processor.pad(
            {'input_values':input_a},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        
        batch_b = self.processor.pad(
            {'input_values':input_b},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=self.return_attention_mask,
            return_tensors="pt",
        )
        
        scores = [feature['rating'] for feature in features]

        return batch_a, batch_b, scores, tasks, entry