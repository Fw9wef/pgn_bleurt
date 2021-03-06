import tensorflow as tf
import os
import numpy as np
from settings import batch_size


# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
BAD_WORDS = [SENTENCE_START, SENTENCE_END, PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]


def remove_bad_words(text):
    """
    Функция удаляет из переданного текста специальные специальные (токен конца строки, пэддинги и т.д.)
    Params:
        text: str: строка текста
    Retirn:
        str: строка текста без специальных слов
    """
    words = text.split()
    good_words = [word for word in words if word not in BAD_WORDS]
    return ' '.join(good_words)


def lens_to_time_step_masks(lens, max_len):
    """
    Функция по данным ей длинам строк в тензоре составляет маску пэддингов (0 - pad, 1 - не pad)
    Params:
        lens: длины резюме в батче
        max_len: максимальная длина последовательности
    Return:
        маска пэддингов
    """
    masks = [[1 for i in range(l)]+[0 for i in range(max_len-l)] for l in lens]
    with tf.device('CPU'):
        masks = tf.constant(masks)
    return masks


def save_model(model, full_path_to_checkpoints, epoch, batch_n, stage='pretrain'):
    """
    Функция выполняет сохранение весов модели
    Params:
        model: tf Model: модель
        full_path_to_checkpoints: str: путь к папке с весами
        epoch: int: номер эпохи
        batch_n: int: номер батча / номер шага оптимизатора
        stage: str: этап обучения модели (предобучение, rl-обучение)
    """
    model.save_weights(os.path.join(full_path_to_checkpoints, 'pgn_' + str(stage) + '_epoch_' + str(epoch) + '_batch_' + str(batch_n)),
                       overwrite=True)


def save_loss(full_path_to_metrics, mean_epoch_loss, train_val='val'):
    """
    Функция выполняет запись текущей ошибки в текстовый файл
    Params:
        full_path_to_metrics: str: путь к папке с метриками
        mean_epoch_loss: float: значение ошибки для записи
        train_val: str: тип набора данных (train, val, test)
    """
    f = open(os.path.join(full_path_to_metrics, train_val + '_ce_loss.txt'), 'a')
    f.write(str(mean_epoch_loss) + '\n')
    f.close()


def save_scores(full_path_to_metrics, scores, train_val='val'):
    """
    Функция выполняет запись текущих метрик в текстовый файл
    Params:
        full_path_to_metrics: str: путь к папке с метриками
        scores: dict: словарь содержащий метрики для записи ('bleurt' - bleurt score, '1' - rouge1, '2' - rouge2, 'l' - rougeL)
        train_val: str: тип набора данных (train, val, test)
    """
    f = open(os.path.join(full_path_to_metrics, train_val + '_bleurt.txt'), 'a')
    mean_score = np.mean(scores['bleurt'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_r1.txt'), 'a')
    mean_score = np.mean(scores['1'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_r2.txt'), 'a')
    mean_score = np.mean(scores['2'])
    f.write(str(mean_score) + '\n')
    f.close()
    f = open(os.path.join(full_path_to_metrics, train_val + '_rl.txt'), 'a')
    mean_score = np.mean(scores['l'])
    f.write(str(mean_score) + '\n')
    f.close()


def save_examples(full_path_to_examples, articles, gt_summaries, summaries, epoch, batch_n,
                  train_val='val', stage='pretrain', in_graph_decodings=False):
    """
    Записывает в txt файлы примеры составления резюме (исходный текст, гт резюме, сгенерированное резюме)
    Params:
        full_path_to_examples: str: путь к папке с примерами
        articles: list of str: список исходных текстов
        gt_summaries: list of str: список гт резюме
        summaries: list of str: список сгенерированнных резюме
        epoch: int: номер эпохи обучения
        batch_n: int: номер батча / номер шага оптимизатора
        train_val: str: тип набора данных (train, val, test)
        stage: str: этап обучения модели (pretrain, rl_train)
        in_graph_decodings: False or list of str: параметр необходимый для проверки декодирования резюме внутри графа tf
                            Содержал резюме, декодированные внутри tf графа.
    """
    new_dir_name = str(stage) + '_' + train_val + '_epoch_' + str(epoch) + '_batch_' + str(batch_n)
    path_to_dir = os.path.join(full_path_to_examples, new_dir_name)
    make_dir(path_to_dir)
    for i, n in enumerate(np.random.choice(len(articles), min(10, len(articles)), replace=False)):
        path_to_file = os.path.join(path_to_dir, 'example_' + str(i + 1) + '.txt')
        f = open(path_to_file, 'w')
        f.write(remove_bad_words(articles[n]))
        f.write('\n' + '#' * 50 + '\n')
        f.write(remove_bad_words(gt_summaries[n]))
        f.write('\n' + '#' * 50 + '\n')
        f.write(remove_bad_words(summaries[n]))
        if in_graph_decodings:
            f.write('\n' + '#' * 50 + '\n')
            f.write(in_graph_decodings[n])



def make_dir(path_to_dir):
    """
    Проверяет существование папки и создает ее
    Params:
        path_to_dir: str: путь к папке
    """
    if not os.path.isdir(path_to_dir):
        os.mkdir(path_to_dir)


def make_dirs(path_to_checkpoints, experiment_name):
    """
    Создает необходимые для записи результатов эксперимента папки (для весов, метрик и примеров резюме)
    Params:
        path_to_checkpoints: str: путь папке с экспериментами
        experiment_name: str: название текущего эксперимента
    """
    make_dir(path_to_checkpoints)

    path_to_experiment = os.path.join(path_to_checkpoints, experiment_name)
    make_dir(path_to_experiment)

    model_checkpoints = os.path.join(path_to_experiment, 'model_checkpoints')
    examples_folder = os.path.join(path_to_experiment, 'examples')
    metrics_folder = os.path.join(path_to_experiment, 'metrics')

    make_dir(model_checkpoints)
    make_dir(examples_folder)
    make_dir(metrics_folder)

    return model_checkpoints, examples_folder, metrics_folder


def check_shapes(inputs):
    """
    Проверяет, совпадает ли размер батча в переданном тензоре с указанным размером в настройках.
    Params:
        inputs: tf.Tensor: тензор для проверки
    """
    for tensor in inputs[-1].values:
        if tensor.shape[0] != batch_size:
            return False
    return True
