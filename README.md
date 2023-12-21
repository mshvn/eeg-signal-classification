
# EEG Signal Classification for Action Prediction

![HSE-MLDS22](https://img.shields.io/badge/HSE-MLDS22-blue)
![ML-OPS](https://img.shields.io/badge/ML%20--%20OPS-8A2BE2)


В задании используется датасет номер 4 из BCI competition II, задача бинарной классификации. Модель -- CatBoost.

## Описание датасета:

Даны 416 эпох длиной 500 мс, каждая из которых заканчивается за 130 мс до нажатия клавиши. Помечены 316 эпох (класс 0 для предстоящих движений левой руки и класс 1 для предстоящих движений правой руки), остальные 100 эпох не помечены (тест).
Цель состоит в том, чтобы предсказать предстоящее движение указательных пальцев левой(0), либо правой(1) руки за 130 мс до нажатия клавиши.

>The dataset IV of BCI competition II was also used for validation in this study.
>This dataset was recorded from a normal subject with 28 Ag/AgCl electrodes at 1000 Hz with a band-pass filter of 0.05–200 Hz.
>The dataset contained 416 epochs, including 316 epochs for training and 100 epochs for testing.
>Each epoch was 500 ms in length and ended at 130 ms before a keypress.
>The recording was made using a NeuroScan amplifier.

## Команды для запуска train и infer:

```
python3 commands.py train
```

```
python3 commands.py infer
```
