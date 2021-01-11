import tensorflow as tf
from tensorflow.keras import layers
from bleurt import score as bleurt_score
from utils import remove_bad_words, lens_to_time_step_masks
from rouge_score import rouge_scorer


class Env():
    def __init__(self, data, bleurt_device):
        self.data = data
        self.bleurt_device = bleurt_device
        with tf.device(self.bleurt_device):
            self.bleurt_scorer = bleurt_score.BleurtScorer('/kaggle/sci/bleurt/bleurt/test_checkpoint')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    
    def get_rewards(self, batch_texts, batch_tokens, batch_oovs, scoring_model = 'final', no_bad_words=True, get_rogue=True):
        assert scoring_model in ['final'], "Not implemented scoring model"
        batch_size = len(batch_texts)
        
        if no_bad_words:
            for i, text in enumerate(batch_texts):
                batch_texts[i] = remove_bad_words(text)
        
        summaries, summary_lens = [], []
        for i, (tokens, oovs) in enumerate(zip(batch_tokens, batch_oovs)):
            summary = self.data.detokenize(tokens, oovs)
            summary_lens.append(len(summary.split()))
            if no_bad_words:
                summary = remove_bad_words(summary)
            summaries.append(summary)
        
        if scoring_model == 'final':
            with tf.device(self.bleurt_device):
                bleurt_scores = self.bleurt_scorer.score(batch_texts, summaries)
            
            rouge_1_scores, rouge_2_scores, rouge_l_scores, rouge_w_scores = [],[],[],[]
            if get_rogue:
                for hypothesis, reference in zip(batch_texts, summaries):
                    temp_score = self.rouge_scorer.get_scores(hypothesis, reference)
                    rouge_1_scores.append(temp_score['rouge-1']['f'])
                    rouge_2_scores.append(temp_score['rouge-2']['f'])
                    rouge_l_scores.append(temp_score['rouge-l']['f'])
                    rouge_w_scores.append(temp_score['rouge-w']['f'])
            
            scores = {
                'bleurt':bleurt_scores,
                '1':rouge_1_scores,
                '2':rouge_2_scores,
                'l':rouge_l_scores,
                'w':rouge_w_scores,
            }
        
        time_step_masks = lens_to_time_step_masks(summary_lens, len(batch_tokens[0]))
        
        return scores, summaries, time_step_masks


'''
class CELoss(layers.Layer):
    def __init__(self, alpha=1., vocab=None, layer_name='ce_loss'):
        super(CELoss, self).__init__(name=layer_name)
        self.alpha = alpha
        self.stop_token = vocab.STOPid
    
    def call(self, gt, logits):
        coords = tf.where(gt==self.stop_token)
        seq_len = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        for i in range(gt.shape[0]):
            sub_coords = tf.cast(tf.where(coords[:,0]==i, coords[:,1], gt.shape[-1]), tf.int32)
            seq_len = seq_len.write(i, tf.math.reduce_min(sub_coords))
        seq_lens = seq_len.stack()
        
        time_step = tf.
        time_step_mask = tf.stack([tf.ones()], axis=0)
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(gt, logits) * time_step_mask
        loss = tf.math.reduce_sum(loss, axis=1) * self.alpha
        return loss
'''


class CELoss(layers.Layer):
    def __init__(self, alpha=1., layer_name='ce_loss'):
        super(CELoss, self).__init__(name=layer_name)
        self.alpha = alpha
    
    def call(self, gt, probs, time_step_mask):
        #probs (batch, seqlen, classes)
        
        gt, probs, time_step_mask = gt[:,1:], probs[:,:-1], time_step_mask[:,1:]
        
        batch_inds = tf.expand_dims(tf.range(probs.shape[0]), 1)
        #batch_inds = tf.tile(batch_inds,[1,probs.shape[1]])
        batch_inds = tf.tile(batch_inds,[1,120])
        
        #seq_inds = tf.expand_dims(tf.range(probs.shape[1]), 0)
        seq_inds = tf.expand_dims(tf.range(120), 0)
        seq_inds = tf.tile(seq_inds, [probs.shape[0],1])
        
        inds = tf.stack([batch_inds, seq_inds, gt], axis=-1)
        inds = tf.reshape(inds, (-1,3))
        
        preds = tf.gather_nd(probs, inds)
        preds = tf.reshape(preds, time_step_mask.shape)
        
        loss = -tf.math.log(preds+1e-12) * time_step_mask
        loss = tf.math.reduce_sum(loss, axis=1) * self.alpha
        return loss


class CoverLoss(layers.Layer):
    def __init__(self, alpha=1., layer_name='cover_loss'):
        super(CoverLoss, self).__init__(name=layer_name)
        self.alpha=alpha
    
    def call(self, cover_loss, time_step_mask):
        return cover_loss * time_step_mask * self.alpha


class RLLoss(layers.Layer):
    def __init__(self, alpha=1., layer_name='rl_loss'):
        super(RLLoss, self).__init__(name=layer_name)
    
    def call(self, logits, delta_rewards, time_step_mask):
        loss = -tf.math.log(tf.nn.softmax(logits, axis=-1)) * delata_reward * time_step_mask
        loss = tf.math.reduce_sum(loss, axis = 1) * self.alpha
        return loss