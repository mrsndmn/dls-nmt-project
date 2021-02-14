
Хотелось поэксперементировать с большим количеством голов и блоков.

Попытки обучить трансформер из 6 блоков не увенчались успехом, но я все равно еще попробую разобраться и сделать это.

## Обучение


Во время обучения `batch_size = 100`

![loss](resources/4blocks/loss.png)
![bleu](resources/4blocks/bleu.png)
![lr](resources/4blocks/lr.png)


## Примеры перевода

На последней эпохе:

```
translation:

более комплексный подход будет сосредоточен на укреплении украины в каждом отношении.

в обоих случаях уровень долга в конечном счете стал неустойчивым.

время не было бы хуже.

вы в толпе, когда вы слышите вашу имя.

esda сократила свой час на балканах.


=======

target:

более комплексный подход предполагает усиление украины по всем направлениям.

в обоих случаях уровни задолженности стали непосильными.

нельзя было выбрать худшего момента для проведения таких мер.

находясь в толпе людей, вы слышите свое имя.

епоб приобрела свой первый опыт на балканах.
```

## Прунинг голов энкодера

Во время прунинга `batch_size = 20`.

### lambda=0.5

Уже когда писал отчет понял, что не зафризил декодер.. Поэтому блеу растет

![l0.5_bleu](resources/4blocks/4blocks_pruned_l0.5_bleu.png)
![l0.5_loss](resources/4blocks/4blocks_pruned_l0.5_loss.png)
![l0.5_popen](resources/4blocks/4blocks_pruned_l0.5_popen.png)

![4_blocks_pruned_l0.5_heads](resources/4blocks/4_blocks_pruned_l0.5.gif)

### lambda=0.1

![l0.1_bleu](resources/4blocks/4blocks_pruned_l0.1_bleu.png)
![l0.1_loss](resources/4blocks/4blocks_pruned_l0.1_loss.png)
![l0.1_popen](resources/4blocks/4blocks_pruned_l0.1_popen.png)

![4_blocks_pruned_l0.1_heads](resources/4blocks/4_blocks_pruned_l0.1.gif)

### lambda=0.05

![l0.05_bleu](resources/4blocks/4blocks_pruned_l0.05_bleu.png)
![l0.05_loss](resources/4blocks/4blocks_pruned_l0.05_loss.png)
![l0.05_popen](resources/4blocks/4blocks_pruned_l0.05_popen.png)

![4_blocks_pruned_l0.05_heads](resources/4blocks/4_blocks_pruned_l0.05.gif)

PS вот тут не хватило колаба, чтобы сошлись некоторые гейты, но общий смысл ясен


## Веса

[тык](https://drive.google.com/file/d/1-Lg0ZgpnNXvBFbcExR2vNycfN_-9FBym/view?usp=sharing)
