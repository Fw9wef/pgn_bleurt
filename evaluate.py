import numpy as np
import tensorflow as tf
from data import Data
from model import PGN
from env import Env, CELoss, RLLoss, Detokenize, BleurtLayer
from tqdm import tqdm
from utils import save_model, save_scores, save_loss, save_examples, make_dirs, check_shapes
from settings import rl_train_epochs, pretrain_epochs, batch_size, gpu_ids, checkpoints_folder, experiment_name, load_model_path

assert load_model_path, 'Model path must be specified during testing'

tf.debugging.set_log_device_placement(False)
global_batch_size = batch_size * len(gpu_ids)
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus = [gpus[i] for i in gpu_ids]
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Define multigpu strategy
devices = ['/device:GPU:'+str(i) for i in gpu_ids]
train_strategy = tf.distribute.MirroredStrategy(devices=devices)


#################################################################################################
# LOADING DATA
#################################################################################################

val_data = Data(mode='val')
vocab = val_data.vocab
val_full_dataset = val_data.get_all_data()
val_article = val_full_dataset['article_text']
val_extended_input_tokens = val_full_dataset['extended_article_tokens']
val_summary = val_full_dataset['summary_text']
val_extended_gt_tokens = val_full_dataset['extended_summary_tokens']
val_index = val_full_dataset['index']
val_oovs = val_full_dataset['oovs']
val_loss_mask = val_full_dataset['summary_loss_points']
val_tensor_oovs = val_full_dataset['tensor_oovs']

with tf.device('CPU'):
    val_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (val_extended_input_tokens, val_extended_gt_tokens, val_loss_mask, val_tensor_oovs, val_index)).batch(int(global_batch_size))
    val_dist_dataset = train_strategy.experimental_distribute_dataset(val_tf_dataset)

max_oovs_in_text = max(0, np.max(val_extended_input_tokens) - vocab.size() + 1)
print('Max oovs in text :', max_oovs_in_text)


#################################################################################################
# DEFINE MULTIGPU TRAIN STEP FUNCTIONS
#################################################################################################

with train_strategy.scope():
    model = PGN(vocab=vocab, max_oovs_in_text=max_oovs_in_text)
    model.load_weights(load_model_path)


def eval_step(extended_input_tokens, extended_gt_tokens, loss_mask, oovs, idx):
    model.switch_decoding_mode('beam_search')
    greedy_seqs = model(extended_input_tokens, extended_gt_tokens, training=False)
    return greedy_seqs


# @tf.function
def distributed_step(dist_inputs):
    greedy_seqs = train_strategy.run(eval_step, args=(dist_inputs))
    return greedy_seqs


env = Env(data=val_data, bleurt_device='cpu')
model_checkpoints, examples_folder, metrics_folder = make_dirs(checkpoints_folder, experiment_name)
val_batches_per_epoch = len(val_tf_dataset)

val_sums = []
val_inds = []
val_iterator = iter(val_dist_dataset)
for val_batch_n in tqdm(range(val_batches_per_epoch)):
    batch = next(val_iterator)
    if check_shapes(batch):
        greedy_seqs = distributed_step(batch)

        with tf.device('CPU'):
            val_sums += list(tf.concat(greedy_seqs.values, axis=0).numpy())
            val_inds += list(tf.concat(batch[-1].values, axis=0).numpy().squeeze())

articles = [val_article[x] for x in val_inds]
gt_summaries = [val_summary[x] for x in val_inds]
examples_oovs = [val_oovs[x] for x in val_inds]
scores, summaries, time_step_masks = env.get_rewards(gt_summaries, val_sums, examples_oovs)
save_examples(examples_folder, articles, gt_summaries, summaries, 'NA', 'NA', 'test', stage='beam')
save_scores(metrics_folder, scores, 'beam_test')

print("Training complete:)")