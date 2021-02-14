

# Что это за проект?

Финальный проект [DLSchool](https://www.dlschool.org/).

### Про что проект
Применение модели Трансформер для решения задачи машинного перевода
Исследование методов уменьшения моделей, прунинг голов.

### Что делаем
Написать свой трансформер для решения задачи перевода, лучше разобраться в основных моментах реализации.
Разобраться с методами прунинга нейросетей; посмотреть, как работает для простых моделей и как работает удаление ""голов"" трансфрмера. Написать свой метод прунинга.

### План
1. Сделать модель трансформер и обучить её на задачу перевода.
2. Изучить тему прунинга нейронов и голов
3. Реализовать метод из статьи Voita et al."

---

## С какими трудностями столкнулся

* было не понятно, как при генерации src_seq_len может быть больше, чем последовательность, которую мы генерим.
В итоге тот факт, что при предсказании у меня не сходились размерности, я не сразу, но понял, что у меня была
ошибка в Attention'е... И прописывание размерностей для всех тензоров очень помогает.


## Что помогло:

* тесты -- так намного быстрее можно управиться с багами в размерностях или архитектуре, чем если тестить и разрабатывать в ноутбуках
* http://nlp.seas.harvard.edu/2018/04/03/attention.html (из доки hf-transformers)
* https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
* https://lena-voita.github.io/posts/acl19_heads.html
* https://lena-voita.github.io/posts/source_target_contributions_to_nmt.html было просто интересно!
* статья про оптимизацию L0 лоссов https://openreview.net/pdf?id=H1Y8hhg0b
* документация pytorch. Некоторые вещи там понятнее описаны, чем в статьях)

# Структура проекта

| Директория | Описание |
|--|---|
| models| модельки, логика, архитектура |
|datamodules| дата-модули, работа с датасетом |
|pl_transformer.py| Скрипт для обучения трансформера |
|pl_pruned_enconder_transformer.py| Скрипт для прунинга обученного трансформера |


# Эксперименты

Обучался на WMT16 en-ru. Переводил с английского на русский.

Провел три эксперимента.

1. В [первом](experiments/2_block_mvp.md) нужно было убедиться, что код хоть на сколько-нибудь рабочий, понять, что сейчас сделано не так и попробовать исправить это.

1. Во [втором](experiments/2_block_fixed.md) эксперименте повторил то, что было в первом, но с учетом исправлений ошибок.

1. В [третьем](experiments/4_block.md) эксперименте увеличил количество блоков до четырех. Странно, но если увеличить кол-во блоков до 6, то обучаться будет намного хуже. Возможно, надо поэкспереентировать с batch_size/lr. Но для себя я пока не понял, в чем может быть дело.

Попытки обучить трансформер из 6 блоков не увенчались успехом, но я все равно еще попробую разобраться и сделать это.

Сначала не хотелось верить, что невозможно обучить сразу маленькое кол-во голов так, чтобы они были по качеству примерно такие же, как "запруненая" модель. Но после того, как реализовал и начал обучать сам, стало понятно, что мы не "просто отрубаем ненужные головы", а во время занижения веса/важности ненужных голов, нужные, умные головы тоже обучаются так, что берут на себя те функции, которые раньше исполняли запруненные головы.


### Гиперпараметры

В экспериментах выше менялось количество блоков. Количество блоков у декодера и у энкодера во всех экспериментах одинаковое.

```
d_model = 512
{k,q,v}_dim = 64
src_vocab_size = 10000
trg_vocab_size = 10000
```

Еще batch_size во время прунинга я брал равным 20, а не 100, чтобы быстрее вычислялось и за одно и то же время сделалось бы больше шагов оптимизации.

### Известные проблемы (которые обнаружил уже после последнего эксперимента):
Не заморозил декодер перед прунингом энкодера. Поэтому получилось как и говорила Viota -- роль энкодера, скорее всего, взял на себя декодер.
Понял это, когда заметил, что для `lambda=0.5` когда все головы энкодера запрунились, bleu не сильно просадился.
Да и для остальных экспериментов с прунингом почти не было значительного изменения bleu после прунинга.
На стадии MVP на это не обратил внимания, потому что показалось, что блеу может так расти как раз из-за того, что данных немного. И хотя мы
запрунили большое количество голов, но модель все еще в состоянии переобучиться. И тк для MVP версии валидации проходили чаще, там были видны и падения bleu, и его рост.

Пофиксил в [этом](https://github.com/mrsndmn/dls-nmt-project/commit/b503380229560bf3e72a8576a1f51e854ef38686) коммите. Конечно да, евал не фризит параметры! И это было в домашке, возможно даже в перой части... И в проекте про CycleGAN'ы. Но все эксперименты проводил еще до фиксов. И для перезапуска этих экспериментов уже нет времени.


---

# Appendix

### Дальнейшие исследования

* можно ли для того, чтобы обучить большой трансформер, обучить сначала маленький, а потом
скопировать уже обученные слои? Поможет ли это ускорить решение? Из того, как слои прунятся, точно понятно,
что первый слой не надо копировать, он особенный и полезная голова там обычно только одна)).
* можно ли ускорить обучение трансформера, если сначала переобучить его на маленьком датасете, а
потом дообучить на большом. Потому что хотя мой первый двухблочный трансформер был точно переобучен на маленькой части датасета,
он довольно неплохо справлялся с переводами, которые я сам в него загонял интерактивно в ноутбуке

### Команды, которыми можно запустить обучение/прунинг

Обученние + скачка данных, если их еще нет в `.data`. Если нужно будет поменять trg_vocab_size/src_vocab_size, то надо сначала удалить старые даныне, чтобы пересчитать bpe.
```
!PYTHONPATH=. python3 pl_transformer.py --check_val_every_n_epoch 1 \
                                        --max_epochs 5 \
                                        --batch_size=100 \
                                        --val_batch_size=50 \
                                        --src_vocab_size=10000 \
                                        --trg_vocab_size=10000 \
                                        --scheduler=noam \
                                        --noam_step_factor=1 \
                                        --noam_opt_warmup_steps=4000 \
                                        --noam_scaler=1 \
                                        --scheduler_patience=10 \
                                        --lr=1 \
                                        --num_blocks 4 \
                                        --hidden_dim 512 \
                                        --key_query_value_dim=64 \
                                        --gpus=1 \
                                        --progress_bar_refresh_rate=20 \
                                        --checkpoint=''
```
PS lr=1 тут потому что настоящий lr считается в шедулере и значение которое получили в шедулере умножается на значение, которое мы передали в lr.

----

Прунинг
```
PYTHONPATH=. python3 pl_pruned_enconder_transformer.py \
                                        --checkpoint '/content/drive/MyDrive/Colab Notebooks/dls-nmt/4_blocks_512_bs_100_5epochs/checkpoints/epoch=4-step=12889.ckpt' \
                                        --lr=0.0004 \
                                        --gpus=1 \
                                        --scheduler=no \
                                        --noam_opt_warmup_steps=4000 \
                                        --noam_step_factor=2 \
                                        --batch_size=20 \
                                        --val_batch_size=20 \
                                        --check_val_every_n_epoch=1 \
                                        --max_epochs=15 \
                                        --src_vocab_size=10000 \
                                        --trg_vocab_size=10000 \
                                        --hcg_l0_penalty_lambda=0.1
```

PS некоторые параметры, кажется довольно бессмысленно передавать, но из-за того, что в скриптах довольно криво парсятся аргументы и интерфейс модудей, их приходится передавать (но на будущее я понял, что лучше не делать позиционных аргументах в лайтинг моделях)
