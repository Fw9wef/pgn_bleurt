import os
import numpy as np
import tensorflow as tf
import pyrouge
from train import VAL_DATA, batch_size



class Buffer():
    def __init__(self):
        sampled_logits = []
        greedy_logits = []
        gen_summaries = []
        ref_summaries = []
        articles = []
        ce_losses = []
        cover_losses = []
        sample_rewards = []
        greedy_rewards = []


def eval_model(model_and_losses, eval_data, train_buffer, full_path_to_examples, full_path_to_metrics, mode='rl'):
    assert mode in ['rl', 'ce'], "Invalid eval mode"
    iters_done = 0
    eval_buffer = Buffer()
    
    model, env, ce_loss, cover_loss = model_and_losse
    model.switch_decoding_mode('self_critic')
    
    for batch in eval_data:
        article = batch['article_text']
        input_tokens = batch['article_tokens']
        extended_input_tokens = batch['extended_article_tokens']
        article_oovs = batch['article_oovs']
            
        summary = batch['summary_text']
        gt_tokens = batch['summary_tokens']
        extended_gt_tokens = batch['extended_summary_tokens']
        summary_oovs = batch['summary_oovs']
        
        loss_mask = batch['summary_loss_points']
        
        greedy_logits, sample_logits, greedy_seqs, sample_seqs, coverage_losses = \
        model(input_tokens, extended_input_tokens, gt_tokens, training=False)
        
        
        scores, summaries, _ = env.get_rewards(summary, sample_seqs, article_oovs)
        eval_buffer.sample_rewards.append()
        
        scores, summaries, time_step_masks = env.get_rewards(summary, greedy_seqs, article_oovs)
        eval_buffer.greedy_rewards.append()
        
        eval_buffer.gen_summaries += [eval_data.data.detokenize(tokens, oovs)\
                                      for tokens, oovs in zip(greedy_seqs.numpy(), article_oovs)]
        eval_buffer.ref_summaries += summary
        eval_buffer.articles += article
        
        ce_losses = ce_loss(extended_gt_tokens, gt_logits, loss_mask).numpy()
        eval_buffer.ce_losses += list(ce_losses)
        
        cover_losses = cover_loss(coverage_losses, time_step_masks).numpy()
        eval_buffer.cover_losses += list(np.mean(cover_losses, axis=1))