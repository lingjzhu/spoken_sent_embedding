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
from transformers import BertForMaskedLM, BertConfig, BertModel, BertTokenizerFast, AutoTokenizer
from transformers import Trainer,TrainingArguments
from datasets import load_dataset, Dataset, load_from_disk
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from loss import InfoNCE_loss
from TSDAE import EncoderDecoderModel


from tqdm import tqdm
import wandb


def evaluate(model, dataloader, args):
    
    model.eval()
    print('Evaluating...')
    preds = []
    scores = []
    for batch, score in dataloader:
        with torch.no_grad():
            out = model.encoder(**batch.to(args.device)).pooler_output
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

    tokenizer: BertTokenizerFast
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
        if self.sentencepiece_tokenizer is None:
            sentA = [self.tokenizer(feature["sent_a_"+self.input_type])['input_ids'][:self.max_length_labels] for feature in features]
            sentB = [self.tokenizer(feature["sent_b_"+self.input_type])['input_ids'][:self.max_length_labels] for feature in features]
        elif self.sentencepiece_tokenizer:        
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
class DataCollatorMPMWithPadding:
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

    tokenizer: BertTokenizerFast
    sentencepiece_tokenizer: spm.SentencePieceProcessor = None
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length: Optional[int] = 300
    max_length_labels: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    masking: Optional[bool] = True
    del_ratio: float = 0.6
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        if self.sentencepiece_tokenizer is None:
            input_features = [{"input_ids": self.tokenizer(feature['input_ids'])['input_ids'][:self.max_length_labels]} for feature in features]
        elif self.sentencepiece_tokenizer: 
            input_features = [{"input_ids": self.sentencepiece_tokenizer.piece_to_id(['[CLS]']) + self.delete(self.sentencepiece_tokenizer.encode(feature['input_ids'])[:(self.max_length-2)],del_ratio=self.del_ratio) + self.sentencepiece_tokenizer.piece_to_id(['[SEP]'])} for feature in features]
    
        
        batch = self.tokenizer.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )
        
        
        if self.sentencepiece_tokenizer is None:
            labels = [{"input_ids": self.tokenizer(feature['input_ids'])['input_ids'][:self.max_length_labels]} for feature in features]
        elif self.sentencepiece_tokenizer:        
            labels = [{"input_ids": self.sentencepiece_tokenizer.piece_to_id(['[CLS]']) + self.sentencepiece_tokenizer.encode(feature['input_ids'])[:(self.max_length-2)] + self.sentencepiece_tokenizer.piece_to_id(['[SEP]'])} for feature in features]
          
        
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
    
    # Deletion noise.
    @staticmethod
    def delete(words, del_ratio=0.6):
        n = len(words)
        if n == 0 or del_ratio == 0:
            return words

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True # guarantee that at least one word remains
        words_processed = np.array(words)[keep_or_not]
        return list(words_processed)

    
    
def train_tsdae(model,train_loader, dev_loader, args):
    
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
        for i, (text_batch) in tqdm(enumerate(train_loader)):
            
            model.train()
            out = model(**text_batch.to(device))
            
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
    parser.add_argument('--data', type=str, default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/cv_hubert-base-ls960-100')
    parser.add_argument('--dev_data',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/cv_dev',type=str)
    parser.add_argument('--out_dir',type=str,default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/models/cse-hubert-100_8000')
    parser.add_argument('--task',default='cse',type=str,help='lm or cse')
    parser.add_argument('--tokenizer',default='tokenizer-50',type=str)
    parser.add_argument('--sentencepiece',default=None,type=str)
    parser.add_argument('--pretrained_model',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/models/bert-hubert-100_8000/checkpoint-8000',type=str)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--grad_norm', default=1.0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--grad_acc', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--saving_step',default=50,type=int)
    parser.add_argument('--weight_decay',default=1e-6,type=float)
    parser.add_argument('--del_ratio',default=0.6,type=float)
    parser.add_argument('--project_name',default='cse-hubert-100_8000',type=str)
    
    args = parser.parse_args()

    

    
        
    wandb.init(project=args.project_name, entity="lingjzhu")
    
    if not args.sentencepiece:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        spm_tokenizer = None
        print('Pre-trained tokenizer loaded.')
    else:
        tokenizer = BertTokenizerFast(args.sentencepiece+'.txt')
        spm_tokenizer = spm.SentencePieceProcessor(model_file=args.sentencepiece+'.model')
        print('Sentencepiece tokenizer loaded.')

    # load text dataset
    texts = pd.read_csv(args.data,sep='\t',header=None)
    texts = texts.rename({0:'path',1:'input_ids'},axis=1)
    dataset = Dataset.from_pandas(texts)
    print(dataset[0]['input_ids'])
    
    wandb.config.update({
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_norm":args.grad_norm,
        "weight_decay":args.weight_decay,
        "grad_acc": args.grad_acc,
        'del_ratio': args.del_ratio
    })

    if re.search('-100',args.pretrained_model):
        input_type = 'hubert_100'
    elif re.search('-50',args.pretrained_model):
        input_type = 'hubert_50'
    elif re.search('-200',args.pretrained_model):
        input_type = 'hubert_200'

    train_collator = DataCollatorMPMWithPadding(tokenizer=tokenizer,sentencepiece_tokenizer=spm_tokenizer, padding=True, del_ratio=args.del_ratio)
    train_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=train_collator)

    dev_dataset = load_from_disk(args.dev_data)
    eval_collator = EvalDataCollatorMPMWithPadding(tokenizer=tokenizer,sentencepiece_tokenizer=spm_tokenizer, padding=True, input_type=input_type)
    dev_loader = DataLoader(dev_dataset,batch_size=1,collate_fn=eval_collator)
    
    
    
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.pretrained_model, args.pretrained_model)
    
    
    model.config.decoder_start_token_id = tokenizer.sep_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    train_tsdae(model,train_loader, dev_loader, args)
