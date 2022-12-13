import re
import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

from dataclasses import dataclass, field
import sentencepiece as spm
from transformers import BertForMaskedLM, BertConfig, BertModel, BertTokenizerFast
from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor,Wav2Vec2Config, AutoTokenizer,AutoModel
from transformers import Trainer,TrainingArguments
from datasets import load_dataset, Dataset, load_from_disk
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from loss import InfoNCE_loss
from WavEmbed import WavEmbedModel
from data_utils import EvalDataCollatorWithPadding

from tqdm import tqdm
import wandb



def evaluate(model, dataloader, args):
    print('Evaluating...')
    model.eval()
    preds = []
    scores = []
    filenames = []
    for batch, score in dataloader:
        #batch['decoder_input_ids'] = torch.zeros(batch['input_values'].size(0),1).long()
        with torch.no_grad():
            out = model(**batch.to(args.device)).encoder_last_hidden_state
            pred = F.cosine_similarity(out[0], out[1], dim=-1).detach().cpu().squeeze().numpy()
            preds.append(pred)
            scores.append(score)          
        
    
    preds = np.array(preds).squeeze()
    scores = np.array(scores).squeeze()
    pcor = pearsonr(preds,scores)
    scor = spearmanr(preds,scores)
    
    return {"pearsonr":pcor,"spearmanr":scor}
            



    


@dataclass
class EvalDataCollatorMPMWithPadding:
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

    tokenizer: AutoTokenizer
    sentencepiece_tokenizer: spm.SentencePieceProcessor = None
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    input_type: str = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        sentA = [self.sentencepiece_tokenizer.piece_to_id(['[CLS]']) + self.sentencepiece_tokenizer.encode(feature["sent_a_"+self.input_type])[:self.max_length_labels] + self.sentencepiece_tokenizer.piece_to_id(['[SEP]']) for feature in features]
        sentB = [self.sentencepiece_tokenizer.piece_to_id(['[CLS]']) + self.sentencepiece_tokenizer.encode(feature["sent_b_"+self.input_type])[:self.max_length_labels] + self.sentencepiece_tokenizer.piece_to_id(['[SEP]']) for feature in features]
    
        units = {'input_ids':sentA + sentB}
        
        batch = self.tokenizer.pad(
                units,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )
        
        scores = [feature['similarity'] for feature in features]
        
        return batch, scores

    
    

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
    max_length: Optional[int] = None
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
        if self.sentencepiece_tokenizer is None:
            labels = [{"input_ids": self.tokenizer(self.acoustic_units[feature['path']])['input_ids'][:self.max_length_labels]} for feature in features]
        elif self.sentencepiece_tokenizer:
            labels = [{"input_ids": self.sentencepiece_tokenizer.piece_to_id(['[CLS]']) + self.sentencepiece_tokenizer.encode(self.acoustic_units[feature['path']])[:self.max_length_labels-2] + self.sentencepiece_tokenizer.piece_to_id(['[SEP]'])} for feature in features]
          
        
        labels_batch = self.tokenizer.pad(
                labels,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )


        batch["labels"] = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        return batch
    
    
    
def train(model,train_loader, dev_loader, args):
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(args.out_dir+"/last_model"):
        os.mkdir(args.out_dir+"/last_model")
    if not os.path.exists(args.out_dir+"/best_model"):
        os.mkdir(args.out_dir+"/best_model")
    
        
    batch_size = args.batch_size
    grad_acc = args.grad_acc
    learning_rate = args.learning_rate
    device = args.device
    

    model.to(args.device)   
    
    optimizer = AdamW(model.parameters(), lr=learning_rate,weight_decay=args.weight_decay)
    wandb.watch(model)
    
    best_perf = 0

    for _ in range(args.epochs):
        for i, batch in tqdm(enumerate(train_loader)):
            
            model.train()
            out = model(**batch.to(device))
            
            loss = out.loss
            loss.backward()

            if i%grad_acc == 0:
                wandb.log({"loss": loss})
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
#            if i%100== 0:
#                print("Loss: %s at Iteration: %i"%(loss.item(),i))

            if i%(args.saving_step*args.grad_acc) == 0:
                # evaluate
                model.save_pretrained(args.out_dir)
            if i%(args.saving_step*args.grad_acc) == 0:
                # evaluate
                corr = evaluate(model,dev_loader,args)
                wandb.log({"spearmanr": corr['spearmanr'][0]})
                wandb.log({"pearsonr": corr['pearsonr'][0]})
                
                with open(os.path.join(args.out_dir,'results.txt'),'a') as out:
                    out.write("Iteration: %s - Loss: %s \n"%(i//args.grad_acc,loss.item()))
                    out.write("Pearsonr: %s; P: %s \n"% (corr["pearsonr"][0],corr["pearsonr"][1]))
                    out.write("Spearmanr: %s; P: %s \n\n"% (corr["spearmanr"][0],corr["spearmanr"][1]))
                
                model.save_pretrained(args.out_dir+"/last_model")
                if best_perf < corr["spearmanr"][0]:
                    model.save_pretrained(args.out_dir+"/best_model")
                    
                    best_perf = corr["spearmanr"][0]
                    wandb.log({"spearmanr_best":best_perf})




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/cv_train_filtered',type=str)
    parser.add_argument('--dev_data',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/cv_dev',type=str)
    parser.add_argument('--pseudo_texts',default=None,type=str)
    parser.add_argument('--out_dir',type=str,default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/models/cse-hubert-100_8000')
    parser.add_argument('--task',default='cse',type=str,help='lm or cse')
    parser.add_argument('--tokenizer',default='tokenizer-100',type=str)
    parser.add_argument('--sentencepiece',default=None,type=str)
    parser.add_argument('--pretrained_model',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/models/bert-hubert-100_8000/checkpoint-8000',type=str)
    parser.add_argument('--speech_model',default='facebook/hubert-base-ls960',type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--grad_norm', default=1.0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--grad_acc', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--saving_step',default=200,type=int)
    parser.add_argument('--weight_decay',default=1e-6,type=float)
    parser.add_argument('--project_name',default='cse-hubert-100_8000',type=str)
    parser.add_argument('--run_name',default=None,type=str)
    
    args = parser.parse_args()

    

    
        
    wandb.init(project=args.project_name, entity="lingjzhu")
    wandb.run.name = args.run_name
    
    if not args.sentencepiece:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        spm_tokenizer = None
        print('Pre-trained tokenizer loaded.')
        if re.search('gpt2',args.tokenizer):
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = BertTokenizerFast(args.sentencepiece+'.txt')
        spm_tokenizer = spm.SentencePieceProcessor(model_file=args.sentencepiece+'.model')
        print('Sentencepiece tokenizer loaded.')
    
    
    speech_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=speech_tokenizer)
    
    # load data
    train_dataset = load_from_disk(args.train_data)
    dev_dataset = load_from_disk(args.dev_data)
    
    # load pseudo-labels
    texts = pd.read_csv(args.pseudo_texts,sep='\t',header=None)
    texts = texts.rename({0:'path',1:'input_ids'},axis=1)
    paths = texts['path']
    input_ids = texts['input_ids']
    acoustic_units = {p:i for i,p in zip(input_ids,paths)}

    
    wandb.config.update({
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_norm":args.grad_norm,
        "weight_decay":args.weight_decay,
        "grad_acc": args.grad_acc
    })

    if re.search('-100',args.pretrained_model):
        input_type = 'hubert_100'
    elif re.search('-50',args.pretrained_model):
        input_type = 'hubert_50'
    elif re.search('-200',args.pretrained_model):
        input_type = 'hubert_200'

    train_collator = SpeechUnitDataCollatorWithPadding(processor=processor,tokenizer=tokenizer,sentencepiece_tokenizer=spm_tokenizer,acoustic_units=acoustic_units)
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=train_collator)
    
    eval_collator = EvalDataCollatorWithPadding(processor=processor)
    dev_loader = DataLoader(dev_dataset,batch_size=1,collate_fn=eval_collator)


    model = WavEmbedModel.from_encoder_decoder_pretrained(args.speech_model, args.pretrained_model)
    
    if re.search('gpt2',args.tokenizer):
        model.config.decoder_start_token_id = tokenizer.bos_token_id
    else:
        model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    train(model,train_loader, dev_loader, args)
