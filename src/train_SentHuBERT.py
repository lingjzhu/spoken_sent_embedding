import os
import argparse
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import EvalDataCollatorWithPadding, DataCollatorWithPadding,SpeechUnitDataCollatorWithPadding
from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor,Wav2Vec2Config, AutoTokenizer,AutoModel,BertTokenizerFast
from torch.optim import AdamW
from simcse import SimCSE
import pandas as pd
from SentHuBERT import SentHuBERT, SentHuBERT_CLS
from loss import NTXentLoss,InfoNCE_loss
from scipy.stats import pearsonr,spearmanr
import numpy as np
import sentencepiece as spm
from tqdm import tqdm
from TSDAE import EncoderDecoderModel


import wandb



def evaluate(model, dataloader, args):
    print('Evaluating...')
    model.eval()
    preds = []
    scores = []
    filenames = []
    for batch, score in dataloader:
        with torch.no_grad():
            out = model(**batch.to(args.device)).last_hidden_state
            pred = F.cosine_similarity(out[0], out[1], dim=-1).detach().cpu().squeeze().numpy()
            preds.append(pred)
            scores.append(score)          
        
    
    preds = np.array(preds).squeeze()
    scores = np.array(scores).squeeze()
    pcor = pearsonr(preds,scores)
    scor = spearmanr(preds,scores)
    
    return {"pearsonr":pcor,"spearmanr":scor}
            


    
def train_knowledge_transfer(speech_encoder,text_encoder,train_loader,dev_loader,args):
    
    batch_size = args.batch_size
    grad_acc = args.grad_acc
    lr = args.lr
    device = args.device
    
    best_perf = 0

    text_encoder.to(device)

    speech_encoder.to(device)
    
    if args.freeze_text_encoder:
        optimizer = AdamW(speech_encoder.parameters(), lr=lr,weight_decay=args.weight_decay)
    elif not args.freeze_text_encoder:
        optimizer = AdamW(list(speech_encoder.parameters())+list(text_encoder.parameters()), lr=lr,weight_decay=args.weight_decay)
    
    if args.loss == "mse":
        loss_fn = nn.MSELoss(reduction='sum')
    elif args.loss == 'memory_bank':
        loss_fn = NTXentLoss(temperature=0.05, memory_bank_size=256)

    for _ in range(args.epoch):
        for i, (speech_batch, text_batch) in tqdm(enumerate(train_loader)):
            speech_encoder.train()
            

            if args.loss == 'info_nce' and not args.freeze_text_encoder:

                text_encoder.train()
                text_z1 = text_encoder(**text_batch.to(device)).pooler_output
                text_z2 = text_encoder(**text_batch.to(device)).pooler_output  

                speech_embed = speech_encoder(**speech_batch.to(device)).last_hidden_state

                loss = InfoNCE_loss(text_z1,speech_embed,temperture=0.05) + InfoNCE_loss(text_z2,speech_embed,temperture=0.05) + InfoNCE_loss(text_z1,text_z2,temperture=0.05)

            else:
                text_encoder.eval()
                with torch.no_grad():
                    if not args.text_encoder_decoder:
                        text_out = text_encoder(**text_batch.to(device))
                    else:
                        text_out = text_encoder.encoder(**text_batch.to(device))
                    if args.text_pooling == 'cls': 
                        text_embed = text_out.pooler_output
                    elif args.text_pooling == 'cls_before_pooling':
                        text_embed = text_out.last_hidden_state[:,0]

                speech_embed = speech_encoder(**speech_batch.to(device)).last_hidden_state

                

                loss = loss_fn(speech_embed,text_embed)
            
            loss.backward()
            
            if i%grad_acc == 0:
                wandb.log({"loss": loss})
                torch.nn.utils.clip_grad_norm_(speech_encoder.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
            if i%(args.saving_step*args.grad_acc) == 0:
                # evaluate
                corr = evaluate(speech_encoder,dev_loader,args)
                wandb.log({"spearmanr": corr['spearmanr'][0]})
                wandb.log({"pearsonr": corr['pearsonr'][0]})
                
                with open(os.path.join(args.outpath,'results.txt'),'a') as out:
                    out.write("Iteration: %s - Loss: %s \n"%(i//args.grad_acc,loss.item()))
                    out.write("Pearsonr: %s; P: %s \n"% (corr["pearsonr"][0],corr["pearsonr"][1]))
                    out.write("Spearmanr: %s; P: %s \n\n"% (corr["spearmanr"][0],corr["spearmanr"][1]))
                    
                speech_encoder.save_pretrained(args.outpath+"/speech/last_model")
                if not args.freeze_text_encoder:
                    speech_encoder.save_pretrained(args.outpath+"/text/last_model")
                if best_perf < corr["spearmanr"][0]:
                    speech_encoder.save_pretrained(args.outpath+"/speech/best_model")
                    if not args.freeze_text_encoder:
                        text_encoder.save_pretrained(args.outpath+"/text/best_model")
                    wandb.log({'spearmanr_best': corr["spearmanr"][0]})
                    best_perf = corr["spearmanr"][0]
                
                
                
                
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Model training arguments')
    
    parser.add_argument('--train_data',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/cv_train_filtered',type=str)
    parser.add_argument('--dev_data',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/cv_dev',type=str)
    parser.add_argument('--pooling', default='self-attention',type=str)
    parser.add_argument('--text_pooling',default='cls',type=str)
    parser.add_argument('--outpath',type=str)
    parser.add_argument('--test_data',type=str,default=None)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--grad_norm', default=10.0, type=float)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--grad_acc', default=8, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--saving_step',default=200,type=int)
    parser.add_argument('--weight_decay',default=1e-6,type=float)
    parser.add_argument('--loss',default='memory_bank',type=str)
    parser.add_argument('--text_model',default="roberta-base",type=str)
    parser.add_argument('--text_encoder_decoder',action='store_true')
    parser.add_argument('--speech_model',default='facebook/hubert-base-ls960',type=str)
    parser.add_argument('--freeze_text_encoder',action='store_true')
    parser.add_argument('--pseudo_units',action='store_true')
    parser.add_argument('--pseudo_texts',default=None,type=str)
    parser.add_argument('--tokenizer',default='bert-base-uncased',type=str)
    parser.add_argument('--sentencepiece',default=None,type=str)
    parser.add_argument('--project_name',type=str)

    args = parser.parse_args()
    
    wandb.init(project=args.project_name, entity="lingjzhu")

    wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "grad_norm":args.grad_norm,
            "weight_decay":args.weight_decay
        }
    
    
    if not os.path.exists(args.outpath):
        os.mkdir(args.outpath)
    if not os.path.exists(args.outpath+"/speech"):
        os.mkdir(args.outpath+"/speech")
    if not os.path.exists(args.outpath+"/text"):
        os.mkdir(args.outpath+"/text")
       
    
    # load tokenizers
    speech_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=speech_tokenizer)

    
    
    
    # load data
    train_dataset = load_from_disk(args.train_data)
    dev_dataset = load_from_disk(args.dev_data)
    
    
    if not args.pseudo_units:
        print("Training model distillation...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # load_model
        train_collator = DataCollatorWithPadding(processor=processor,tokenizer=tokenizer)
        eval_collator = EvalDataCollatorWithPadding(processor=processor)
        dev_loader = DataLoader(dev_dataset,batch_size=1,collate_fn=eval_collator)
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=train_collator)

        text_encoder = AutoModel.from_pretrained(args.text_model)


        if args.pooling == 'self-attention':
            speech_encoder = Wav2Vec2SAP.from_pretrained(args.speech_model)
        elif args.pooling == 'cls':
            speech_encoder = SentWav2Vec2.from_pretrained(args.speech_model)
        else:
            raise Exception("Please specify a pooling method!")

        train_knowledge_transfer(speech_encoder,text_encoder,train_loader,dev_loader,args)


    elif args.pseudo_units:
        
        if not args.sentencepiece:
            tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer)
            spm_tokenizer = None
            print('Pre-trained tokenizer loaded.')
        else:
            tokenizer = BertTokenizerFast(args.sentencepiece+'.txt')
            spm_tokenizer = spm.SentencePieceProcessor(model_file=args.sentencepiece+'.model')
            print('Sentencepiece tokenizer loaded.')
        
        
        # load pseudo-labels
        texts = pd.read_csv(args.pseudo_texts,sep='\t',header=None)
        texts = texts.rename({0:'path',1:'input_ids'},axis=1)
        paths = texts['path']
        input_ids = texts['input_ids']
        acoustic_units = {p:i for i,p in zip(input_ids,paths)}
        

        train_collator = SpeechUnitDataCollatorWithPadding(processor=processor,tokenizer=tokenizer,sentencepiece_tokenizer=spm_tokenizer,acoustic_units=acoustic_units)
        eval_collator = EvalDataCollatorWithPadding(processor=processor)
        dev_loader = DataLoader(dev_dataset,batch_size=1,collate_fn=eval_collator)
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=train_collator)

        if not args.text_encoder_decoder:
            text_encoder = AutoModel.from_pretrained(args.text_model)
            print('%s loaded.'%args.text_model)
        else:
            text_encoder = EncoderDecoderModel.from_pretrained(args.text_model)
            print('TSDAE %s loaded.'%args.text_model)

        if args.pooling == 'self-attention':
            speech_encoder = SentHuBERT.from_pretrained(args.speech_model)
        elif args.pooling == 'cls':
            speech_encoder = SentHuBERT_CLS.from_pretrained(args.speech_model)
        else:
            raise Exception("Please specify a pooling method!")

        train_knowledge_transfer(speech_encoder,text_encoder,train_loader,dev_loader,args)
