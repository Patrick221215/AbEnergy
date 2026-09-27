import functools
import logging
import random

import torch
from torch import nn

from abx.model.head import HeaderBuilder
from abx.model.common_modules import (
        pseudo_beta_fn_v2,
        dgram_from_positions)

logger = logging.getLogger(__name__)


def get_prev(batch, value, config):
    #import ipdb; ipdb.set_trace()
    prev_peusdo_beta = pseudo_beta_fn_v2(batch['seq'], value['heads']['folding']['final_atom_positions'])
    prev_disto_bins = dgram_from_positions(prev_peusdo_beta, **config.embeddings_and_seqformer.prev_pos)
    
    new_prev = {
        'prev_pos': prev_disto_bins.detach(),
        'prev_seq': value['heads']['folding']['representations']['representations_for_heads']['seq'].detach(),
        'prev_pair': value['heads']['folding']['representations']['representations_for_heads']['pair'].detach()
    }
    return new_prev

class ScoreNetworkIteration(nn.Module):
    def __init__(self, model_conf, diffuser, cfg_config={}) -> None:
        super(ScoreNetworkIteration, self).__init__()
        self._model_conf = model_conf

        # self.seqformer = EmbeddingAndSeqformer(self._model_conf.embeddings_and_seqformer)
        self.diffuser = diffuser
        self.heads = HeaderBuilder.build(
                self._model_conf.heads,
                config_seqformer=self._model_conf.embeddings_and_seqformer,
                parent=self,
                diffuser=diffuser,
                cfg_config=cfg_config # 将统一的CFG配置传下去
                )
    
    def forward(self, batch, global_step,compute_loss=False):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch
        Returns:
            model_out: dictionary of model outputs.
        """
        # import ipdb; ipdb.set_trace()
        # seq_act, pair_act = self.seqformer(batch)
        # representations = {'pair': pair_act, 'seq': seq_act}
        ret = {}
        
        # Evoformer Embedding
        # ret['representations'] = representations
        ret['heads'] = {}

        # for name, module, options in self.heads:
        #     if compute_loss or name == 'folding' or name == 'sequence_module':
        #         value = module(ret['heads'], representations, batch)
        #         if value is not None:
        #             ret['heads'][name] = value
        for name, module, options in self.heads:
            if compute_loss or name == 'folding' or name == 'sequence_module':

                current_headers = ret['heads'] # Headers from previously executed heads in this iteration

                if name == 'folding': # This is DiffusionHead -> IpaScore
                    value = module(current_headers, batch, global_step) # Pass batch, ignore reps
                elif name == 'sequence_module':
                    # SequenceHead operates on IpaScore's output activations
                    if 'folding' not in current_headers:
                        logger.error("SequenceModule called before FoldingModule (IpaScore). Skipping.")
                        value = None
                    else: # Pass IpaScore's output to SequenceHead
                        value = module(current_headers, current_headers['folding']['representations'], batch)
                elif name == 'distogram':
                        ipa_output_reps = current_headers['folding']['representations']['representations_for_heads']
                        value = module(current_headers, ipa_output_reps, batch)
                else: # MetricHead, TMscoreHead, PredictedLDDTHead
                    value = module(current_headers, current_headers['folding']['representations'], batch)

                if value is not None:
                    ret['heads'][name] = value          
        return ret


class ScoreNetwork(nn.Module):
    def __init__(self, model_conf, diffuser) -> None:
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf
        self.num_in_seq_channel = self._model_conf.embeddings_and_seqformer.seq_channel
        self.num_in_pair_channel = self._model_conf.embeddings_and_seqformer.pair_channel
        self.index_embed_size = self._model_conf.embeddings_and_seqformer.index_embed_size
        # import ipdb; ipdb.set_trace()
        # 在初始化时，就获取顶层CFG配置，并将其传递给下一层
        self.cfg_config = self._model_conf.get('cfg_config', {})
        self.impl = ScoreNetworkIteration(model_conf, diffuser, cfg_config=self.cfg_config) # ScoreNetworkIteration does not change much

    def forward(self, batch, global_step, compute_loss=True):
        
        if self.training:
            num_recycle = random.randint(0, self._model_conf.num_recycle)
        else:
            num_recycle = self._model_conf.num_recycle
            
        # Initialize 'prev_' features in the batch if not present for the first iteration
        if 'prev_pos' not in batch: # Example, from original code
            batch_size, num_res = batch['seq_t'].shape[:2]
            device = batch['seq_t'].device
            prev_seq_dim = self.num_in_seq_channel + self.index_embed_size 
            prev_pair_dim = self.num_in_pair_channel + 2*self.index_embed_size 
            
            batch['prev_pos'] = torch.zeros([batch_size, num_res, num_res], device=device, dtype=torch.int64) 
            batch['prev_seq'] = torch.zeros([batch_size, num_res, prev_seq_dim], device=device)
            batch['prev_pair'] = torch.zeros([batch_size, num_res, num_res, prev_pair_dim], device=device)

        current_batch_iter = batch
        with torch.no_grad() if num_recycle > 0 else torch.enable_grad(): # No grad for recycle iterations
            current_batch_iter.update(is_recycling=True) # Let modules know if it's a recycle iter
            for i_recycle in range(num_recycle):
                ret_recycle = self.impl(current_batch_iter, global_step,compute_loss=False) # compute_loss typically False for recycle
            
                    
                if i_recycle < num_recycle -1 : # Don't update prev for the very last recycle iteration before final pass
                    # `get_prev` needs the config for prev_pos dgram calculation
                    prev_features = get_prev(
                        current_batch_iter, ret_recycle, 
                        self._model_conf
                    )
                    current_batch_iter.update(prev_features)
                    if 'sequence_module' in ret_recycle['heads']: # Update seq_t if predicted
                        current_batch_iter['seq_t'] = ret_recycle['heads']['sequence_module']['seq_0'].detach()


        current_batch_iter.update(is_recycling=False) # Final pass
        ret_final = self.impl(current_batch_iter, global_step, compute_loss=compute_loss)
        
        return ret_final
        

