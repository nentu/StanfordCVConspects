<!-- TOC -->
  * [Lecture 6 Training Neural Networks](#lecture-6-training-neural-networks)
    * [Activation funcstions](#activation-funcstions)
    * [Инициализация](#инициализация)
    * [Батч нормализация](#батч-нормализация)
  * [Lecture 7 Training Neural Networks (optimization)](#lecture-7-training-neural-networks-optimization)
    * [Опитизация](#опитизация)
    * [Image augmentation](#image-augmentation)
    * [Регулеризация](#регулеризация)
    * [Pre-rained](#pre-rained)
  * [Lecture 8 Deep Learning Software](#lecture-8-deep-learning-software)
    * [Programming on GPU](#programming-on-gpu)
      * [Frameworks reasons](#frameworks-reasons)
    * [NumPy](#numpy)
    * [TensorFlow](#tensorflow)
    * [PyTorch](#pytorch)
    * [TensorFlow](#tensorflow-1)
      * [UseCase](#usecase)
      * [Variables](#variables)
      * [Inner function](#inner-function)
      * [Optimization](#optimization)
      * [Loss](#loss)
      * [Layers](#layers)
    * [Keras](#keras)
      * [High-Level over TS](#high-level-over-ts)
      * [Tensorboard](#tensorboard)
    * [PyTorch](#pytorch-1)
      * [Abstract levels](#abstract-levels)
      * [PuTorch = Numpy + GPU](#putorch--numpy--gpu)
      * [PyTorch Autograd vars](#pytorch-autograd-vars)
      * [Create own PyTorch nodes](#create-own-pytorch-nodes)
      * [High-Level frameworks](#high-level-frameworks)
      * [Create own PyTorch model / class](#create-own-pytorch-model--class)
      * [DataLoader](#dataloader)
      * [PreTrain](#pretrain)
      * [PyTorch VS TensorFlow](#pytorch-vs-tensorflow)
        * [Static](#static)
        * [Dinamic](#dinamic)
    * [Caffe / Caffe 2](#caffe--caffe-2)
    * [Conclusion](#conclusion)
  * [Lecture 9 CNN Architectures](#lecture-9-cnn-architectures)
    * [Winners](#winners)
    * [LeNet-5](#lenet-5)
    * [AlexNet](#alexnet)
    * [VGGNet](#vggnet)
      * [Why 3x3](#why-3x3)
      * [Memory](#memory)
      * [Note](#note)
    * [GoogLeNet](#googlenet)
      * [Inception](#inception)
      * [Conclusion](#conclusion-1)
    * [ResNet](#resnet)
      * [Conclusion](#conclusion-2)
    * [All-in-all](#all-in-all)
    * [Other](#other)
      * [NiN Network in network](#nin-network-in-network)
      * [Improved ResNet](#improved-resnet)
      * [Wide ResNet](#wide-resnet)
      * [ResNeXT](#resnext)
      * [FractalNet](#fractalnet)
      * [DenseNet](#densenet)
      * [SqueezeNet](#squeezenet)
    * [Summary](#summary)
<!-- TOC -->

## Lecture 6 Training Neural Networks

<details>
  <summary>Развернуть</summary>

[video](https://www.youtube.com/watch?v=wEoyxE0GP2M&list=PLSVEhWrZWDHQTBmWZufjxpw3s8sveJtnJ&index=6&pp=iAQB)

### Activation funcstions

Минусы сигмоида:

- mean != 0
- затухание при больших по модулю значений
- exp() ложно считать

leaky Relu

### Инициализация

все 0 - одинаковые + затухание
маленькие с mean = 0 -> затухание
Лучше равномерное / |input|

Для RELu |input / 2|

### Батч нормализация

В батч нормализации есть парамеры

В перпроцессоринге сдвиг на mean
</details>

## Lecture 7 Training Neural Networks (optimization)

<details>
  <summary>Развернуть</summary>


[video](https://www.youtube.com/watch?v=_JB0AO7QxSA&list=PLSVEhWrZWDHQTBmWZufjxpw3s8sveJtnJ&index=7&pp=iAQB)

- Random search оч широкий

### Опитизация

![img_1.png](images/lec7/img_1.png)

![img_2.png](images/lec7/img_2.png)

![img_3.png](images/lec7/img_3.png)

![img_4.png](images/lec7/img_4.png)

![img_5.png](images/lec7/img_5.png)

![img_6.png](images/lec7/img_6.png)

![img_7.png](images/lec7/img_7.png)

![img_8.png](images/lec7/img_8.png)

![img_9.png](images/lec7/img_9.png)

Как правило, в этом алгоритме подбирают лишь один гиперпараметр
learning rate. Остальные же: B1 и B2 – оставляют стандартными и равными 0.9, 0.99 и 1e-8 соответственно.
Подбор a составляет главное искусство.

- Momentum - скорость "инерция"
- AdaGard
-

Идея следующая: если мы вышли на плато по какой-то координате и соответствующая компонента градиента начала затухать, то
нам нельзя уменьшать размер шага слишком сильно, поскольку мы рискуем на этом плато остаться, но в то же время уменьшать
надо, потому что это плато может содержать оптимум. Если же градиент долгое время довольно большой, то это может быть
знаком, что нам нужно уменьшить размер шага, чтобы не пропустить оптимум. Поэтому мы стараемся компенсировать слишком
большие или слишком маленькие координаты градиента.

Но довольно часто получается так, что размер шага уменьшается слишком быстро и для решения этой проблемы придумали
другой алгоритм.

RMSProp
Модифицируем слегка предыдущую идею: будем не просто складывать нормы градиентов, а усреднять их в скользящем режиме:

Не сразу!! Сначала просто a, а потом смотрим, нужно ли вообще decay
![img_10.png](images/lec7/img_10.png)

![img_11.png](images/lec7/img_11.png)

- Ансамбли!+2%
  ![img_12.png](images/lec7/img_12.png)

![img_13.png](images/lec7/img_13.png)

![img_15.png](images/lec7/img_15.png)

![img_14.png](images/lec7/img_14.png)

### Image augmentation

![img_16.png](images/lec7/img_16.png)

### Регулеризация

![img_17.png](images/lec7/img_17.png)

![img_18.png](images/lec7/img_18.png)

![img_19.png](images/lec7/img_19.png)

### Pre-rained

![img_20.png](images/lec7/img_20.png)

![img_21.png](images/lec7/img_21.png)

</details>

## Lecture 8 Deep Learning Software

<details>
  <summary>Развернуть</summary>

[video](https://youtu.be/_JB0AO7QxSA?list=PLSVEhWrZWDHQTBmWZufjxpw3s8sveJtnJ)

### Programming on GPU

![img.png](images/lec8/img.png)

- cuBLAST - matrix
- **Используй cuDNN** - ускоритель для Deep Learning.

![img_2.png](images/lec8/img_2.png)

#### Frameworks reasons

![img_3.png](images/lec8/img_3.png)

### NumPy

![img_4.png](images/lec8/img_4.png)

### TensorFlow

+ GPU and backprop

![img_7.png](images/lec8/img_7.png)

### PyTorch

+ GPU and backprop

![img_9.png](images/lec8/img_9.png)

### TensorFlow

![img_10.png](images/lec8/img_10.png)

#### UseCase

![img_11.png](images/lec8/img_11.png)

- Define vars
- Define computation graph
- Say that we want to count gradients
- Start session where we create/get data and then say what we want to get (loss functions and graads) and give data for
  train
- After that we get numpy array result

Training ![img_12.png](images/lec8/img_12.png)

#### Variables

Due to it is too slow to move data from ts to numpy we can define inner TS vars. But of cause we need to init it.
Then we can define how to use them in our graph. It will be done automatically

![img_14.png](images/lec8/img_14.png)

Now we update values
![img_16.png](images/lec8/img_16.png)

#### Inner function

#### Optimization

![img_17.png](images/lec8/img_17.png)

#### Loss

![img_18.png](images/lec8/img_18.png)

#### Layers

![img_19.png](images/lec8/img_19.png)

### Keras

![img_20.png](images/lec8/img_20.png)

#### High-Level over TS

![img_21.png](images/lec8/img_21.png)

#### Tensorboard

You can visualize computation graph

![img_22.png](images/lec8/img_22.png)

Граф можно разбивать!!

### PyTorch

#### Abstract levels

![img_23.png](images/lec8/img_23.png)

#### PuTorch = Numpy + GPU

![img_24.png](images/lec8/img_24.png)

#### PyTorch Autograd vars

![img_25.png](images/lec8/img_25.png)

#### Create own PyTorch nodes

![img_27.png](images/lec8/img_27.png)

#### High-Level frameworks

![img_28.png](images/lec8/img_28.png)

#### Create own PyTorch model / class

![img_29.png](images/lec8/img_29.png)

#### DataLoader

![img_30.png](images/lec8/img_30.png)

#### PreTrain

![img_31.png](images/lec8/img_31.png)

#### PyTorch VS TensorFlow

![img_26.png](images/lec8/img_26.png)

- PyTorch - create graph every time -> code is cleaner
- TF - define once and use many times.

##### Static

![img_33.png](images/lec8/img_33.png)
![img_34.png](images/lec8/img_34.png)
![img_35.png](images/lec8/img_35.png)

##### Dinamic

![img_37.png](images/lec8/img_37.png)

Пересчёт размеров!
![img_38.png](images/lec8/img_38.png)

![img_39.png](images/lec8/img_39.png)

### Caffe / Caffe 2

![img_40.png](images/lec8/img_40.png)

![img_41.png](images/lec8/img_41.png)

![img_42.png](images/lec8/img_42.png)

![img_43.png](images/lec8/img_43.png)

### Conclusion

![img_44.png](images/lec8/img_44.png)

![img_45.png](images/lec8/img_45.png)

</details>

## Lecture 9 CNN Architectures

<details>
  <summary>Развернуть</summary>

[video](https://youtu.be/DAOcjicFr1Y?list=PLSVEhWrZWDHQTBmWZufjxpw3s8sveJtnJ)

### Winners

![img_6.png](images/lac9/img_6.png)

### LeNet-5
![img.png](images/lac9/img.png)


### AlexNet
![img_3.png](images/lac9/img_3.png)

![img_4.png](images/lac9/img_4.png)

![img_5.png](images/lac9/img_5.png)

### VGGNet

![img_7.png](images/lac9/img_7.png)

#### Why 3x3

![img_8.png](images/lac9/img_8.png)

#### Memory
![img_10.png](images/lac9/img_10.png)

#### Note

**FC7** is rather good for relearning for other data 

### GoogLeNet

![img_11.png](images/lac9/img_11.png)

#### Inception
![img_12.png](images/lac9/img_12.png)

Count Operation problem

![img_14.png](images/lac9/img_14.png)

Solution 1x1
![img_16.png](images/lac9/img_16.png)

![img_17.png](images/lac9/img_17.png)

Can train each part
![img_18.png](images/lac9/img_18.png)

#### Conclusion

![img_19.png](images/lac9/img_19.png)

### ResNet

Mush more layers using residual connection 

То есть теперь мы учим остаток для *x*
![img_20.png](images/lac9/img_20.png)
![img_21.png](images/lac9/img_21.png)

#### Conclusion

![img_22.png](images/lac9/img_22.png)

- "Пронос" остатка
- /2: *2 #фильтров и /2 size by stride

![img_23.png](images/lac9/img_23.png)

![img_24.png](images/lac9/img_24.png)

### All-in-all
![img_25.png](images/lac9/img_25.png)

![img_26.png](images/lac9/img_26.png)

### Other

#### NiN Network in network

![img_27.png](images/lac9/img_27.png)
#### Improved ResNet

![img_28.png](images/lac9/img_28.png)

#### Wide ResNet

wide -> parallel

![img_29.png](images/lac9/img_29.png)

#### ResNeXT

![img_30.png](images/lac9/img_30.png)

#### FractalNet

![img_31.png](images/lac9/img_31.png)

#### DenseNet

![img_32.png](images/lac9/img_32.png)

#### SqueezeNet

![img_33.png](images/lac9/img_33.png)

### Summary

![img_34.png](images/lac9/img_34.png)

</details>