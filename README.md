## CookAI: Cooking Recipe Generation using Deep Learning Models

**Installation:**

We use Google Colab to run this code. This code uses python language to train model and plot the graphs.

Please install python 3.7 or later, install pytorch 0.4 or later.

**Training:**

Use the src/train.py to train the model, we train the model with different learning rate and number of iterations.

In src/modules, we create encoder, decoder and multi-head attention for the transformer model.

**Experiment:**

Use the src/CookingDemo.ipnb to get the model results. We input the food image in data/my_food folder, and the model will generate the Title of this input image, and instructions.

We use src/plot.py to create plots. These plots analyzes the performance of model under different number of blocks and multi-heads.

**We refer this paper:**

*Amaia Salvador, Michal Drozdzal, Xavier Giro-i-Nieto, Adriana Romero.
[Inverse Cooking: Recipe Generation from Food Images. ](https://arxiv.org/abs/1812.06164)
CVPR 2019*


