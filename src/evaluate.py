import os
import re
import json
import argparse
import soundfile as sf
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor,Wav2Vec2Config, AutoModel
from SpeechEncoder import Wav2Vec2SAP, SentWav2Vec2
from scipy.stats import pearsonr,spearmanr
import numpy as np
from tqdm import tqdm
from data_utils import EvalDataCollatorWithPadding,STSEvalDataCollatorWithPadding
import wandb

from WavEmbed import WavEmbedModel



def evaluate_common_voice(model, dataloader, args):
    print('Evaluating...')
    model.eval()
    preds = []
    scores = []
    filenames = []
    for batch, score, filename in tqdm(dataloader):
        with torch.no_grad():
            if args.encoder_decoder:
                out = model(**batch.to(args.device)).encoder_last_hidden_state
            else:
                out = model(**batch.to(args.device)).last_hidden_state
            if args.pooling == 'mean':
                out = torch.mean(out,dim=1)
            pred = F.cosine_similarity(out[0], out[1], dim=-1).detach().cpu().squeeze().numpy()
            preds.append(pred)
            scores.append(score)
            filenames.append(filename[0])
            
    data = pd.DataFrame()
    data['human'] = list(np.array(scores).squeeze())
    data['pred'] = list(np.array(preds).squeeze())
    data['item'] = filenames
    return data


def evaluate_STS(model,dataloader,args):
    print('Evaluating STS datasets...')
    
    model.eval()
    
    preds = []
    scores = []
    entries = []
    tasks = []
    with torch.no_grad():
        for batch_a, batch_b, score, task, entry in tqdm(dataloader):
            if args.encoder_decoder:
                out_a = model(**batch_a.to(args.device)).encoder_last_hidden_state
                out_b = model(**batch_b.to(args.device)).encoder_last_hidden_state
            else:
                out_a = model(**batch_a.to(args.device)).last_hidden_state
                out_b = model(**batch_b.to(args.device)).last_hidden_state
            if args.pooling == 'mean':
                out_a = torch.mean(out_a,dim=1)
                out_b = torch.mean(out_b,dim=1)
            out_a = F.normalize(out_a,dim=-1)
            out_b = F.normalize(out_b,dim=-1)
            pred = torch.matmul(out_a,out_b.t()).detach().cpu().squeeze().numpy()
            pred = np.mean(pred)
            
            
            preds.append(pred)
            scores.append(score)
            tasks.append(task[0])
            entries.append(entry[0])
            
    data = pd.DataFrame()
    data['human'] = list(np.array(scores).squeeze())
    data['pred'] = list(np.array(preds).squeeze())
    data['task'] = tasks
    data['entry'] = entries
    return data
    


def compute_corr(preds,scores):
    preds = np.array(preds).squeeze()
    scores = np.array(scores).squeeze()
    pcor = pearsonr(preds,scores)
    scor = spearmanr(preds,scores)
    
    return {"pearsonr":pcor,"spearmanr":scor}
            

def load_STS(path):
    
    with open(path,'r') as out:
        data = json.load(out)
    
    dataset = []
    for task, entry in data.items():
        for idx, content in entry.items():
            if '1' in content.keys() and '0' in content.keys():
                content['task'] = task
                content['entry'] = idx
                dataset.append(content)

    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation arguments')
    parser.add_argument('--dev_data',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/cv_dev',type=str)
    parser.add_argument('--test_data',default='/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/embeddings/cv_test',type=str)
    parser.add_argument('--STS_natural',default='STS_natural.json',type=str)
    parser.add_argument('--STS_synthetic',default='STS_synthetic.json',type=str)
    parser.add_argument('--pooling',default='self-attention',type=str)
    parser.add_argument('--pretrained_model',default=None,type=str)
    parser.add_argument('--run_name',default=None,type=str)
    parser.add_argument('--out_dir',default=None,type=str)
    parser.add_argument('--device',default='cuda',type=str)
    parser.add_argument('--encoder_decoder',action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_args()
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    wandb.init(project='Evaluation', entity="lingjzhu")
    if args.run_name:
        wandb.run.name = args.run_name
    
    if args.encoder_decoder:
        model = WavEmbedModel.from_pretrained(args.pretrained_model)
    else:
        if args.pooling == 'self-attention':
            model = SentHuBERT.from_pretrained(args.pretrained_model)
        elif args.pooling == 'cls':
            model = SentHuBERT_CLS.from_pretrained(args.pretrained_model)
        elif args.pooling == 'mean':
            model = AutoModel.from_pretrained(args.pretrained_model)
        else:
            raise Exception("Please specify a pooling method!")
        
    model.to(args.device)
    
    # load tokenizers
    speech_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('facebook/wav2vec2-base-960h')
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=speech_tokenizer)

        
    dev_dataset = load_from_disk(args.dev_data)
    test_dataset = load_from_disk(args.test_data)
        
        
    eval_collator = EvalDataCollatorWithPadding(processor=processor,output_filename=True)
    dev_loader = DataLoader(dev_dataset,batch_size=1,collate_fn=eval_collator)
    test_loader = DataLoader(test_dataset,batch_size=1,collate_fn=eval_collator)
    
    common_voice_dev = evaluate_common_voice(model,dev_loader,args)
    common_voice_test = evaluate_common_voice(model,test_loader,args)
    
    common_voice_dev_corr = compute_corr(common_voice_dev['human'],common_voice_dev['pred'])
    wandb.log({'cv_dev_pearsonr':common_voice_dev_corr['pearsonr'][0]})
    wandb.log({'cv_dev_spearmanr':common_voice_dev_corr['spearmanr'][0]})
    common_voice_dev.to_csv(os.path.join(args.out_dir,'cv_dev.tsv'),sep='\t',index=None)
    
    common_voice_test_corr = compute_corr(common_voice_test['human'],common_voice_test['pred'])
    wandb.log({'cv_test_pearsonr':common_voice_test_corr['pearsonr'][0]})
    wandb.log({'cv_test_spearmanr':common_voice_test_corr['spearmanr'][0]})
    common_voice_test.to_csv(os.path.join(args.out_dir,'cv_test.tsv'),sep='\t',index=None)
    
    # Evaluate STS dataset
    STS_natural_dataset = load_STS(args.STS_natural)
    STS_synthetic_dataset = load_STS(args.STS_synthetic)
    
    STScollator = STSEvalDataCollatorWithPadding(processor=processor)
    STS_natural_loader = DataLoader(STS_natural_dataset,batch_size=1,collate_fn=STScollator)
    STS_synthetic_loader = DataLoader(STS_synthetic_dataset,batch_size=1,collate_fn=STScollator)
    
    STS_natural_results = evaluate_STS(model,STS_natural_loader,args)
    
    
    STS_natural_results.to_csv(os.path.join(args.out_dir,'STS_natural.tsv'),sep='\t',index=None)
    
    STS_natural_corr = compute_corr(STS_natural_results['human'],STS_natural_results['pred'])
    wandb.log({'STS_natural_pearsonr':STS_natural_corr['pearsonr'][0]})
    wandb.log({'STS_natural_spearmanr':STS_natural_corr['spearmanr'][0]})    
    
    STS_synthetic_results = evaluate_STS(model,STS_synthetic_loader,args)
    STS_synthetic_results.to_csv(os.path.join(args.out_dir,'STS_synthetic.tsv'),sep='\t',index=None)
    
    STS_synthetic_results['year'] = [re.search(r'(STS\d\d)_.*?',i).group(1) for i in STS_synthetic_results['task'].tolist()]
    years = STS_synthetic_results['year'].unique().tolist()
    for k in years:
        subset = STS_synthetic_results[STS_synthetic_results['year']==k]
        STS_synthetic_corr = compute_corr(subset['human'],subset['pred'])
        wandb.log({'STS_synthetic_'+k+'_pearsonr':STS_synthetic_corr['pearsonr'][0]})
        wandb.log({'STS_synthetic_'+k+'_spearmanr':STS_synthetic_corr['spearmanr'][0]})           
