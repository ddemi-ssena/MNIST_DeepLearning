Project Goal & Scope

The main objectives of this project are:

Implement and train a sigmoid-activated neural network on the MNIST dataset.

Experimentally explore hyperparameter tuning by isolating the effects of:

Hidden layer size

Learning rate

Mini-batch size

Compare how these parameters influence classification accuracy.
Model Architecture and Algorithms
Feature	Detail
Model Type	Fully Connected Network (MLP)
Activation Function	Sigmoid function ϭ(z) for all layers
Learning Algorithm	Mini-batch Stochastic Gradient Descent (SGD)
Cost Function	Quadratic Cost (Mean Squared Error, MSE)
Input Layer	784 neurons (28×28 pixel vector)
Hidden Layer	30 neurons (Baseline)
Output Layer	10 neurons (One-Hot Encoded digits 0–9)
Hyperparameter Tuning Experiments

The table below summarizes the final accuracy (Epoch 29) for the baseline model and three isolated hyperparameter experiments.

Hyperparameter Comparison Table
Deney No	Parameter Changed	Baseline Setting	New Setting	Final Accuracy	Primary Observation
Baseline	–	(30N, η=3.0, m=10)	–	≈ 94.78%	Fast convergence due to large learning rate.
1	Hidden Layer Neurons	30	100	≈ 96.57%	Strong improvement; higher capacity = better feature learning.
2	Learning Rate (η)	3.0	0.5	≈ 94.02%	Slower convergence; cannot reach baseline performance in 30 epochs.
3	Mini-Batch Size (m)	10	32	≈ 94.58%	Slight drop; more stable gradients but fewer updates per epoch.
Learning Curve Visualizations

You can visualize the effect of each hyperparameter through the comparison graphs.

Make sure to upload the PNG files into your GitHub repo.

Graph 1: <img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/ab7ff3f0-2193-4677-866e-92556290cc79" />

Baseline vs. Hidden Layer Size Experiment

Graph 2: <img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/8331c9fd-dd24-49fd-b029-e4de2148d61b" />

Baseline vs. Learning Rate Experiment

Graph 3: <img width="789" height="490" alt="image" src="https://github.com/user-attachments/assets/a49e11e1-e4b1-4c44-ac6b-e2189dcd813b" />

Baseline vs. Mini-Batch Size Experiment

🚀 Setup and Execution
Requirements

Python 3.x

numpy

matplotlib

1. Clone the Repository
git clone https://github.com/MichalDanielDobrzanski/DeepLearningPython.git
cd DeepLearningPython

2. Download MNIST Dataset
wget https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz

3. Run the Notebook

Open and execute the MNIST.ipynb file:

jupyter notebook MNIST.ipynb


Run the cells sequentially to reproduce model training and experiment results.

📖 References & Attribution

Primary Code & Theory Source

Nielsen, Michael A. (2015). Neural Networks and Deep Learning.

Dobrzanski, Michal Daniel — DeepLearningPython GitHub Repository.


🤖 AI Assistance Statement

This project benefited from multiple AI tools used for explanation refinement, debugging support, experimental interpretation, and code improvements.
All AI tools were used under the DIR (Description–Interpretation–Reflection) framework to enhance clarity and technical depth.

AI Tools Used
1. Google AI Studio

Purpose: Python code generation and corrections

URL: https://aistudio.google.com/prompts/13YuLe3txd9q5x5shq8SfIpHkqm7eRJxC

2. Gemini AI Model

Purpose: Report analysis, hyperparameter interpretation, code generation & debugging

URL: https://gemini.google.com/app/7e950f1472b0c6e3?hl=tr

3. ChatGPT

Purpose: Debugging, report analysis, hyperparameter interpretation

URL: https://chatgpt.com/c/69173c8c-bc2c-832c-9b7d-918a1e72cbdc
