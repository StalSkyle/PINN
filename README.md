## Решение задачи Коши для системы уравнений Лотки-Вольтерры с использованием PINN

### Описание

Этот проект представляет собой решение задачи Коши для системы дифференциальных уравнений Лотки-Вольтерры (модель "хищник-жертва") с использованием метода нейронных сетей на основе физически информированных нейронных сетей (PINN). Метод PINN позволяет моделировать динамику популяций, обучая нейронную сеть на основе дифференциальных уравнений и начальных условий задачи.

### Система уравнений Лотки-Вольтерры

Система уравнений Лотки-Вольтерры описывает взаимодействие двух видов: хищников и жертв. Система имеет вид:

$$
\begin{cases}
\frac{dx}{dt} = \alpha x - \beta xy \\
\frac{dy}{dt} = -\gamma y + \delta xy
\end{cases}
$$

### Начальные условия

В модели задаются начальные условия для популяций на момент времени \$(t = 0)\$. Конкретно, начальные значения популяций жертв и хищников задаются как:

* \$(x(0) = x\_0)\$ — начальная популяция жертв,
* \$(y(0) = y\_0)\$ — начальная популяция хищников.

Значения \$x\_0\$ и \$y\_0\$ задаются в словаре `start`, который выглядит следующим образом:

```python
start = {
    "x0": 40.0,
    "y0": 9.0
}
```

### Задание

* Реализовать метод решения задачи Коши для системы уравнений Лотки-Вольтерры с использованием метода PINN.
* Исследовать влияние параметров на динамику популяций.
* Сравнить результаты с классическими численными методами, такими как метод Рунге-Кутты.

### Описание кода

1. **Построение графиков**
   В коде используется библиотека `matplotlib` для построения графиков, которые отображают динамику популяций жертв и хищников во времени, а также фазовые траектории.

2. **Решение системы уравнений методом Рунге-Кутты**
   Для сравнения с методом PINN решается система уравнений Лотки-Вольтерры с использованием стандартного численного метода Рунге-Кутты 4-5 порядка через функцию `solve_ivp`.

3. **Нейронная сеть (PINN)**
   Создана нейронная сеть, которая использует синусоидальные функции активации для аппроксимации решения задачи Коши. Входом для сети является время \$t\$, а выход — популяции \$x(t)\$ и \$y(t)\$.

4. **Функция потерь**
   Функция потерь состоит из двух частей:

   * Ошибка, связанная с тем, насколько хорошо сеть решает дифференциальные уравнения Лотки-Вольтерры.
   * Ошибка начальных условий (сравнение предсказанных значений \$x(0)\$ и \$y(0)\$ с реальными значениями).

5. **Обучение нейронной сети**
   Сеть обучается с использованием оптимизатора Adam с динамической настройкой скорости обучения. Потери рассчитываются в каждой эпохе, и происходит обратное распространение ошибки для обновления параметров сети.

6. **Сравнение с методом Рунге-Кутты**
   Для оценки качества работы модели PINN результаты предсказания популяций \$x(t)\$ и \$y(t)\$ сравниваются с результатами, полученными методом Рунге-Кутты.

### Пример запуска

1. Установите необходимые библиотеки:

   ```bash
   pip install torch numpy matplotlib scipy
   ```

2. Скачайте или скопируйте код в файл, например, `lotka_volterra_pinn.py`.

3. Запустите скрипт:

   ```bash
   python lotka_volterra_pinn.py
   ```

   Это вызовет обучение модели и вывод графиков, отображающих результаты решения задачи Коши для системы Лотки-Вольтерры.

### Параметры системы

В коде используются следующие значения параметров системы:

```python
constants = {
    "alpha": 0.8,
    "beta": 0.1,
    "gamma": 1.5,
    "delta": 0.1,
}

start = {
    "x0": 40.0,
    "y0": 9.0
}
```

### Влияние параметров

Влияние параметров на динамику популяций можно исследовать, меняя значения коэффициентов в словаре `constants` и наблюдая за изменениями в графиках, отображающих поведение популяций со временем.

### Результаты

Процесс обучения завершится после нескольких тысяч эпох. После этого будет выведено два графика:

* Графики временных рядов для популяций хищников и жертв.
* Фазовые графики, показывающие зависимость популяций друг от друга.

Дополнительно построен график потерь, отображающий процесс сходимости модели, а также график ошибки (Linf), сравнивающий результаты, полученные методом PINN, с результатами численного интегрирования.

### Результаты исследования

На основе проведенного численного эксперимента исследованы зависимости динамики популяций от параметров модели:

* **Коэффициент роста жертв $\alpha$:** при увеличении $\alpha$ наблюдается возрастание максимума численности жертв, что, в свою очередь, стимулирует рост числа хищников. При слишком больших значениях (например, $\alpha=25$) обе популяции вымирают: избыточное размножение жертв приводит к резкому росту хищников и полному истреблению жертв, после чего вымерзают и хищники.

* **Коэффициент взаимодействия $\beta$:** при низких значениях $\beta$ хищники слишком эффективно поедают жертв, что приводит к вымиранию обеих популяций. С увеличением $\beta$ амплитуда колебаний численности хищников снижается, и их динамика становится более сглаженной.

* **Коэффициент смерти хищников $\gamma$:** повышение $\gamma$ снижает устойчивость популяции хищников, что приводит к более выраженным и быстрому затуханию колебаний. При критически высоких $\gamma$ хищники не успевают реагировать на рост жертв и исчезают.

* **Коэффициент прироста хищников $\delta$:** увеличение $\delta$ ускоряет рост хищников за счет жертв, что приводит к усилению колебаний и потенциальной неустойчивости системы.

* **Состояния равновесия:** найдены два устойчивых состояния: $x=0, y=0$ (вымирание) и $x=\gamma/\delta, y=\alpha/\beta$ (неустойчивый фокус).

* **Общие закономерности:** пик численности жертв предшествует росту хищников, снижение числа хищников приводит к падению популяции жертв. Долгосрочное сохранение обеих популяций требует баланса параметров: высокие темпы роста хищников или жертв без противовеса приводят к коллапсу системы.

### Заключение

Использование метода PINN для решения системы уравнений Лотки-Вольтерры позволяет эффективно аппроксимировать решение задачи Коши и исследовать влияние параметров на динамику популяций. Численные эксперименты показали, что модель адекватно воспроизводит классические траектории и выявляет критические пороги параметров, при которых система переходит к вымиранию или устойчивому колебанию.

Дополнительно был выявлен ряд важных закономерностей: чрезвычайно высокие значения коэффициентов роста приводят к неустойчивости, а баланс между коэффициентами взаимодействия и смертями определяет устойчивость популяций. В будущем планируется:

* Исследовать модифицированные модели Лотки-Вольтерры с дополнительными факторами (внутривидовая конкуренция, миграция, сезонность).
* Применить подход PINN к другим нелинейным системам, например, гармоническому осциллятору или уравнению Пуассона.
