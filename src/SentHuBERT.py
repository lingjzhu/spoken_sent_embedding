import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2BaseModelOutput,Wav2Vec2EncoderStableLayerNorm,Wav2Vec2Encoder
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureExtractor,Wav2Vec2FeatureProjection,Wav2Vec2FeatureEncoder
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings


class SentHuBERT_CLS(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.projector = nn.Sequential(
                         nn.Linear(config.hidden_size, config.hidden_size),
                         nn.Tanh()
                        )

        self.cls = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
        self.cls_mask = nn.Parameter(torch.ones(1,dtype=torch.bool),requires_grad=False)
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # append the CLS token to hidden states
        batch_size,length,hidden_size = hidden_states.shape
        cls_tokens = self.cls.repeat(batch_size,1).unsqueeze(1)
        hidden_states = torch.concat((cls_tokens,hidden_states),dim=1)
        
        # expand the attention mask
        cls_mask = self.cls_mask.repeat(batch_size,1).to(hidden_states.device)
        attention_mask = torch.cat((cls_mask,attention_mask),dim=1)
        
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]
        projected = self.projector(hidden_states[:,0,:])
        
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=projected,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    

    
class AttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask=None):
        """
            N: batch size, T: sequence length, H: Hidden dimension
            input:
                batch_rep : size (N, T, H)
            attention_weight:
                att_w : size (N, T, 1)
            return:
                utter_rep: size (N, H)
        """
        att_logits = self.W(batch_rep).squeeze(-1)
        if att_mask is not None:
            att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep
    
    
    
class SentHuBERT(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.att = AttentionPooling(config.hidden_size)
        
        self.projector = nn.Sequential(
                         nn.Linear(config.hidden_size, config.hidden_size),
                         nn.Tanh()
                        )

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_values,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = self.att(encoder_outputs[0],attention_mask)
        
        projected = self.projector(hidden_states)
        
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=projected,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
    
    
    
 
