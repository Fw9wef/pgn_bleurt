import numpy as np
import tensorflow as tf
from data import Data
from model import PGN
from env import Env, CELoss, RLLoss, Detokenize, BleurtLayer
from tqdm import tqdm
from utils import save_model, save_scores, save_loss, save_examples, make_dirs, check_shapes
from settings import pretrain_epochs, batch_size, gpu_ids, checkpoints_folder, experiment_name, load_model_path


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

# Loading TRAIN data
data = Data()
vocab = data.vocab
full_dataset = data.get_all_data()
article = full_dataset['article_text']
extended_input_tokens = full_dataset['extended_article_tokens']
summary = full_dataset['summary_text']
extended_gt_tokens = full_dataset['extended_summary_tokens']
index = full_dataset['index']
oovs = full_dataset['oovs']
loss_mask = full_dataset['summary_loss_points']
tensor_oovs = full_dataset['tensor_oovs']

with tf.device('CPU'):
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        (extended_input_tokens, extended_gt_tokens, loss_mask, tensor_oovs, index)).batch(global_batch_size)
    train_tf_dataset = train_tf_dataset.shuffle(32)
    train_dist_dataset = train_strategy.experimental_distribute_dataset(train_tf_dataset)

# Loading VAL data
val_data = Data(mode='val', vocab=vocab)
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

max_oovs_in_text = max(0, np.max(extended_input_tokens) - vocab.size() + 1,
                       np.max(val_extended_input_tokens) - vocab.size() + 1)
print('Max oovs in text :', max_oovs_in_text)


#################################################################################################
# DEFINE MULTIGPU TRAIN STEP FUNCTIONS
#################################################################################################

with train_strategy.scope():
    model = PGN(vocab=vocab, max_oovs_in_text=max_oovs_in_text)
    if load_model_path:
        model.load_weights(load_model_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    ce_loss = CELoss(alpha=0.5)
    rl_loss = RLLoss(alpha=0.5)
    detokenize = Detokenize(vocab)
    bleurt_scorer = BleurtLayer()



# def train_step(inputs):
def pretrain_step(extended_input_tokens, extended_gt_tokens, loss_mask, oovs, idx):
    model.switch_decoding_mode('cross_entropy')

    with tf.GradientTape() as tape:
        gt_probs, greedy_seqs, coverage_losses = model(extended_input_tokens, extended_gt_tokens, training=True)
        loss = tf.nn.compute_average_loss(ce_loss(extended_gt_tokens, gt_probs, loss_mask),
                                          global_batch_size=global_batch_size)

    grads = tape.gradient(loss, model.trainable_weights)
    grads = [tf.clip_by_norm(g, 2) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    greedy_summary, greedy_mask = detokenize(greedy_seqs, oovs)
    return loss, greedy_seqs, greedy_summary


def rl_train_step(extended_input_tokens, extended_gt_tokens, loss_mask, oovs, idx):
    model.switch_decoding_mode('self_critic')

    with tf.GradientTape() as tape:
        greedy_probs, sample_probs, greedy_seqs, sample_seqs, coverage_losses = model(extended_input_tokens,
                                                                                      extended_gt_tokens,
                                                                                      training=True)
        loss = 0

        # computing cross entropy loss
        loss += tf.nn.compute_average_loss(ce_loss(extended_gt_tokens, sample_probs, loss_mask),
                                           global_batch_size=global_batch_size)

        # computing self critic reward loss
        with tape.stop_recording():
            gt_summary = detokenize(extended_gt_tokens, oovs)
            greedy_summary, greedy_mask = detokenize(greedy_seqs, oovs)
            sample_summary, sample_mask = detokenize(sample_seqs, oovs)
            greedy_rewards = bleurt_scorer(gt_summary, greedy_summary)
            sample_rewards = bleurt_scorer(gt_summary, sample_summary)
            delta_rewards = sample_rewards - greedy_rewards

        loss += tf.nn.compute_average_loss(rl_loss(sample_seqs, sample_probs, sample_mask, delta_rewards),
                                           global_batch_size=global_batch_size)

    grads = tape.gradient(loss, model.trainable_weights)
    grads = [tf.clip_by_norm(g, 2) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss, greedy_seqs, greedy_summary


def eval_step(extended_input_tokens, extended_gt_tokens, loss_mask, oovs, idx):
    model.switch_decoding_mode('evaluate')

    greedy_probs, sample_probs, greedy_seqs, sample_seqs, coverage_losses = model(extended_input_tokens,
                                                                                  extended_gt_tokens, training=False)
    loss = tf.nn.compute_average_loss(ce_loss(extended_gt_tokens, greedy_probs, loss_mask),
                                      global_batch_size=global_batch_size)

    greedy_summary, greedy_mask = detokenize(greedy_seqs, oovs)
    return loss, greedy_seqs, greedy_summary


#@tf.function
def distributed_step(dist_inputs, mode):
    if mode == 'train':
        per_replica_losses, greedy_seqs = train_strategy.run(pretrain_step, args=(dist_inputs))

    elif mode == 'rl_train':
        per_replica_losses, greedy_seqs = train_strategy.run(pretrain_step, args=(dist_inputs))

    elif mode == 'val':
        per_replica_losses, greedy_seqs = train_strategy.run(eval_step, args=(dist_inputs))

    return train_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None), greedy_seqs


env = Env(data=data, bleurt_device='cpu')
model_checkpoints, examples_folder, metrics_folder = make_dirs(checkpoints_folder, experiment_name)
batches_per_epoch = len(train_tf_dataset)
val_batches_per_epoch = len(val_tf_dataset)

# tf.debugging.set_log_device_placement(True)

for epoch in range(1, pretrain_epochs + 1):
    new_learning_rate = 0.0005 - (0.0005 - 0.001) * (epoch - 1) / (pretrain_epochs - 1)
    optimizer.lr.assign(new_learning_rate)
    iterator = iter(train_dist_dataset)
    print('epoch', epoch)
    losses = []
    for batch_n in tqdm(range(1, batches_per_epoch+1)):

        batch = next(iterator)
        if check_shapes(batch):
            loss, greedy_seqs, greedy_summaries = distributed_step(batch, 'rl_train')
            losses.append(loss)

            if batch_n % 200 == 0:
                # if True:
                with tf.device('CPU'):
                    train_sums = list(tf.concat(greedy_seqs.values, axis=0).numpy())
                    train_inds = list(tf.concat(batch[-1].values, axis=0).numpy().squeeze())
                    in_graph_decodings = tf.concat(greedy_summaries.values, axis=0).numpy()
                    in_graph_decodings = [x.decode() for x in in_graph_decodings]

                articles = [article[x] for x in train_inds]
                gt_summaries = [summary[x] for x in train_inds]
                examples_oovs = [oovs[x] for x in train_inds]
                scores, summaries, time_step_masks = env.get_rewards(gt_summaries, train_sums, examples_oovs)
                save_examples(examples_folder, articles, gt_summaries, summaries, epoch, batch_n, 'rl_train',
                              in_graph_decodings=in_graph_decodings)
                save_scores(metrics_folder, scores, 'rl_train')

                mean_epoch_loss = np.mean(losses)
                losses = []
                save_loss(metrics_folder, mean_epoch_loss, 'rl_train')

                val_losses = []
                val_sums = []
                val_inds = []
                val_iterator = iter(val_dist_dataset)
                for val_batch_n in range(1, min(10, batches_per_epoch)):
                    batch = next(val_iterator)
                    loss, greedy_seqs, greedy_summaries = distributed_step(batch, 'val')
                    val_losses.append(loss)

                    with tf.device('CPU'):
                        val_sums += list(tf.concat(greedy_seqs.values, axis=0).numpy())
                        val_inds += list(tf.concat(batch[-1].values, axis=0).numpy().squeeze())
                        in_graph_decodings = tf.concat(greedy_summaries.values, axis=0).numpy()
                        in_graph_decodings = [x.decode() for x in in_graph_decodings]

                articles = [val_article[x] for x in val_inds]
                gt_summaries = [val_summary[x] for x in val_inds]
                examples_oovs = [val_oovs[x] for x in val_inds]
                scores, summaries, time_step_masks = env.get_rewards(gt_summaries, val_sums, examples_oovs)
                save_examples(examples_folder, articles, gt_summaries, summaries, epoch, batch_n, 'val',
                              in_graph_decodings=in_graph_decodings)
                save_scores(metrics_folder, scores, 'val')

                mean_epoch_loss = np.mean(val_losses)
                save_loss(metrics_folder, mean_epoch_loss, 'val')

        if batch_n % 600 == 0:
            save_model(model, model_checkpoints, epoch, batch_n)

    save_model(model, model_checkpoints, epoch, 'last')

print("Training complete:)")

