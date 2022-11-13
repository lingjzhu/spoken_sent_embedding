import re
import os
import json
import torch
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

from dataclasses import dataclass, field
from transformers import BertTokenizerFast
import sentencepiece as spm
from transformers import BertForMaskedLM, BertConfig, BertModel
from transformers import Trainer,TrainingArguments
from datasets import load_dataset, Dataset, load_from_disk
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss import InfoNCE_loss
from tqdm import tqdm
import wandb



def torch_mask_tokens(inputs, special_tokens_mask,mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # We'll use the attention mask here
    special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # 
    inputs[indices_replaced] = torch.tensor(tokenizer.mask_token_id)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


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
        if not self.sentencepiece_tokenizer:
            sentA = [self.tokenizer(feature["sent_a_"+input_type])['input_ids'][:self.max_length_labels] for feature in features]
            sentB = [self.tokenizer(feature["sent_b_"+input_type])['input_ids'][:self.max_length_labels] for feature in features]
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

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        if not self.sentencepiece_tokenizer:
            label_features = [{"input_ids": self.tokenizer(feature['input_ids'])['input_ids'][:self.max_length]} for feature in features]
        elif self.sentencepiece_tokenizer:
            label_features = [{"input_ids": self.sentencepiece_tokenizer.piece_to_id(['[CLS]']) + self.sentencepiece_tokenizer.encode(feature['input_ids'])[:(self.max_length-2)] + self.sentencepiece_tokenizer.piece_to_id(['[SEP]'])} for feature in features]
    
        
        batch = self.tokenizer.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )
        
        if self.masking == True:
            inputs, labels = torch_mask_tokens(batch['input_ids'],1-batch['attention_mask'])
            # replace padding with -100 to ignore loss correctly
            labels = labels.masked_fill(batch.attention_mask.ne(1), -100)
            
            if not inputs.shape == labels.shape:
                print(inputs)
                print(labels)
                raise Exception()
            
            batch['input_ids'] = inputs
            batch["labels"] = labels
            
            

        return batch


def evaluate(model, dataloader, args):
    
    model.eval()
    print('Evaluating...')
    preds = []
    scores = []
    for batch, score in dataloader:
        with torch.no_grad():
            out = model(**batch.to(args.device)).pooler_output
            pred = F.cosine_similarity(out[0], out[1], dim=-1).detach().cpu().squeeze().numpy()
            preds.append(pred)
            scores.append(score)
    
    preds = np.array(preds).squeeze()
    scores = np.array(scores).squeeze()
    pcor = pearsonr(preds,scores)
    scor = spearmanr(preds,scores)
    
    return {"pearsonr":pcor,"spearmanr":scor}
            

    

def train_cse(encoder,train_loader, dev_loader, args):
    
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
    

    encoder.to(device)   
    
    
    #criterion = NTXentLoss(temperature=0.05, memory_bank_size=args.memory_bank)
    #criterion = BarlowTwinsLoss()
    
    optimizer = AdamW(encoder.parameters(), lr=learning_rate,weight_decay=args.weight_decay)
    wandb.watch(encoder)
    
    best_perf = 0

    for _ in range(args.epochs):
        for i, (text_batch) in tqdm(enumerate(train_loader)):
            encoder.train()
            out0 = encoder(**text_batch.to(device))
            out1 = encoder(**text_batch.to(device)) 
            
            p0 = out0.pooler_output
            p1 = out1.pooler_output
            
            #z0 = out0.last_hidden_state[:,0]
            #z1 = out1.last_hidden_state[:,0]
            loss = InfoNCE_loss(p0,p1,temperture=args.temperture)
#            loss = criterion(p0,p1) #+ criterion(p0,z1.detach()) + criterion(p1,z0.detach())


            loss.backward()

            if i%grad_acc == 0:
                wandb.log({"loss": loss})
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
#            if i%100== 0:
#                print("Loss: %s at Iteration: %i"%(loss.item(),i))

            if i%(args.saving_step*args.grad_acc) == 0:
                # evaluate
                encoder.save_pretrained(args.out_dir)
            if i%(args.saving_step*args.grad_acc) == 0:
                # evaluate
                corr = evaluate(encoder,dev_loader,args)
                wandb.log({"spearmanr": corr['spearmanr'][0]})
                wandb.log({"pearsonr": corr['pearsonr'][0]})
                
                with open(os.path.join(args.out_dir,'results.txt'),'a') as out:
                    out.write("Iteration: %s - Loss: %s \n"%(i//args.grad_acc,loss.item()))
                    out.write("Pearsonr: %s; P: %s \n"% (corr["pearsonr"][0],corr["pearsonr"][1]))
                    out.write("Spearmanr: %s; P: %s \n\n"% (corr["spearmanr"][0],corr["spearmanr"][1]))
                
                encoder.save_pretrained(args.out_dir+"/last_model")
                if best_perf < corr["spearmanr"][0]:
                    encoder.save_pretrained(args.out_dir+"/best_model")
                    
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
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--grad_norm', default=1.0, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--grad_acc', default=8, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--saving_step',default=10,type=int)
    parser.add_argument('--weight_decay',default=1e-6,type=float)
    parser.add_argument('--mask_prob',default=0.1,type=float)
    parser.add_argument('--memory_bank',default=0,type=int)
    parser.add_argument('--temperture',default=0.05,type=float)
    parser.add_argument('--project_name',default='cse-hubert-100_8000',type=str)
    args = parser.parse_args()

    
    wandb.init(project=args.project_name, entity="lingjzhu")
    
    if not args.sentencepiece:
        tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
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
    

    # load model
    if args.task == 'lm':
        print('Beginning training unit LM.')
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.vocab_size = len(tokenizer)
        model = BertForMaskedLM(config)

        data_collator = DataCollatorMPMWithPadding(tokenizer=tokenizer,sentencepiece_tokenizer=spm_tokenizer,padding=True)

        training_args = TrainingArguments(
                                          output_dir=args.out_dir,
                                          group_by_length=True,
                                          per_device_train_batch_size=64,
                                          gradient_accumulation_steps=8,
                                          num_train_epochs=5,
                                          fp16=True,
                                          save_steps=2000,
                                          logging_steps=50,
                                          learning_rate=1e-4,
                                          weight_decay=0.001,
                                          warmup_steps=1000,
                                          save_total_limit=2,
                                          ignore_data_skip=True
                                         )


        trainer = Trainer(
                            model=model,
                            data_collator=data_collator,
                            args=training_args,
                            train_dataset=dataset,
                         )


        trainer.train()
        
    elif args.task == 'cse':
        wandb.config.update({
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_norm":args.grad_norm,
            "weight_decay":args.weight_decay,
            "memory_bank":args.memory_bank,
            "temperture": args.temperture,
            "grad_acc": args.grad_acc
        })
        
        if re.search('-100',args.pretrained_model):
            input_type = 'hubert_100'
        elif re.search('-50',args.pretrained_model):
            input_type = 'hubert_50'
        elif re.search('-200',args.pretrained_model):
            input_type = 'hubert_200'
    
        train_collator = DataCollatorMPMWithPadding(tokenizer=tokenizer,sentencepiece_tokenizer=spm_tokenizer, padding=True, masking=False)
        train_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=train_collator)
    
        dev_dataset = load_from_disk(args.dev_data)
        eval_collator = EvalDataCollatorMPMWithPadding(tokenizer=tokenizer,sentencepiece_tokenizer=spm_tokenizer, padding=True, input_type=input_type)
        dev_loader = DataLoader(dev_dataset,batch_size=1,collate_fn=eval_collator)

        model = BertModel.from_pretrained(args.pretrained_model,
                                          attention_probs_dropout_prob=args.mask_prob,
                                          hidden_dropout_prob=args.mask_prob)

        train_cse(model,train_loader, dev_loader, args)
