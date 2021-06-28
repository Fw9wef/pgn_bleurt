data_folder = '../finished_files/chunked'  # путь к папке с данными
vocab_file = '../finished_files/vocab'  # путь к vocab файлу
bleurt_model = '../bleurt/bleurt/test_checkpoint'  # путь к весам bleurt модели
checkpoints_folder = './chk'  # путь к папке, в которую будет сохранен эксперимент (веса, метрики, примеры генерации)
experiment_name = 'rl_testing'  # название эксперимента. Так будет названа папка внутри checkpoints_folder
load_model_path = './chk/rl_learning/model_checkpoints/pgn_pretrain_epoch_10_batch_last'  # путь к существующим весам
                                                                                          # параметр необходим при тестировании
                                                                                          # и дообучении с помощью rl
vocab_size = 50000  # размер словаря модели
article_max_tokens = 400  # максимальная длина резюмируемого текста в токенах
summary_max_tokens = 120  # максимальная длина резюме в токенах
pretrain_epochs = 10  # количество эпох предобучения (необходимо для pretrain и train)
rl_train_epochs = 5  # количество эпох rl обучения (необходимо для rl_train и train)
batch_size = 4  # размер батча на один gpu. Итоговый размер батча = batch_size * len(gpu_ids)
gpu_ids = [0, 1]  # индексы используемых gpu
