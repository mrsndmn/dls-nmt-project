## Обучение

Фиксы после mvp версии:
* валидацию делаем на действительно отложенном датасете, а не на трейне
* рандомная сортировка предложений. В первой версии они были отсортированы по длинне. Поэтому у лосса были периоды.


Во время обучения `batch_size = 50`

![loss](resources/2blocks_fixed/loss.png)
![bleu](resources/2blocks_fixed/bleu.png)
![lr](resources/2blocks_fixed/lr.png)


## Примеры перевода

```
translation:

этот провал уже вызвал многие коренные проблемы.

это значительно повысило бы устойчивость долга этих стран.

популярность исламистов не трудно понять.

нью-йорк – я – американский, московский корпус.

среди нелояльности, однако, каждый из них становится преданным по-своему.


=======
target:

такие неудачи пугают многих корейцев.

это позволит значительно повысить устойчивость долга этих стран.

популярность исламистов понять несложно.

нью-йорк – я американка, родилась в москве.

а вот предателем каждый становится по-своему.
```

## Прунинг голов энкодера


### lambda=0.2

![l0.2_bleu](resources/2blocks_fixed/2blocks_fixed_pruned_l0.2_bleu.png)
![l0.2_loss](resources/2blocks_fixed/2blocks_fixed_pruned_l0.2_loss.png)
![l0.2_popen](resources/2blocks_fixed/2blocks_fixed_pruned_l0.2_popen.png)
> Тут в легенде перепутаны первый и второй слои


![pruned_l0.2_heads](resources/2blocks_fixed/2_blocks_fixed_pruned_l0.2.gif)
> По горизонтали головы (8 штук), по вертикали слои (2 штуки)

### lambda=0.1

![l0.1_bleu](resources/2blocks_fixed/2blocks_fixed_pruned_l0.1_bleu.png)
![l0.1_loss](resources/2blocks_fixed/2blocks_fixed_pruned_l0.1_loss.png)
![l0.1_popen](resources/2blocks_fixed/2blocks_fixed_pruned_l0.1_popen.png)
> Тут в легенде перепутаны первый и второй слои

![pruned_l0.1_heads](resources/2blocks_fixed/2_blocks_fixed_pruned_l0.1.gif)
> По горизонтали головы (8 штук), по вертикали слои (2 штуки)

### lambda=0.05

![l0.05_bleu](resources/2blocks_fixed/2blocks_fixed_pruned_l0.05_bleu.png)
![l0.05_loss](resources/2blocks_fixed/2blocks_fixed_pruned_l0.05_loss.png)
![l0.05_popen](resources/2blocks_fixed/2blocks_fixed_pruned_l0.05_popen.png)
> Тут в легенде перепутаны первый и второй слои


![pruned_l0.05_heads](resources/2blocks_fixed/2_blocks_fixed_pruned_l0.05.gif)
> По горизонтали головы (8 штук), по вертикали слои (2 штуки)

