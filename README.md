

# SMS/Email Classifier

![Designer (1)](https://github.com/code-red-Marshall/EMAIL-SMS-Spam-Classifier/assets/82904501/dbf6f976-3ab3-4314-9b21-5046c7966e2e)


Streamlit app link: https://email-sms-spam-classifier-erf7teejfi7xyzdy8ejsml.streamlit.app/

This repository contains a Jupyter Notebook that demonstrates the process of building an SMS/Email classifier. The classifier can distinguish between spam and non-spam messages using various machine learning techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed Workflow](#detailed-workflow)
  - [1. Data Cleaning](#1-data-cleaning)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Text Preprocessing](#3-text-preprocessing)
  - [4. Model Building](#4-model-building)
  - [5. Evaluation](#5-evaluation)
  - [6. Improvement](#6-improvement)
  - [7. Deployment](#7-deployment)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project follows a structured 7-step process to build an effective SMS/Email classifier:
1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Text Preprocessing
4. Model Building
5. Evaluation
6. Improvement
7. Deployment

## Installation
To run this project, you will need to have Python installed along with the following packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- nltk

You can install the necessary packages using pip:
```bash
pip install pandas numpy scikit-learn matplotlib nltk
```

## Usage
1. **Clone this repository**:
    ```bash
    git clone https://github.com/your-username/sms-email-classifier.git
    cd sms-email-classifier
    ```

2. **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook "Tutorial notebook.ipynb"
    ```

3. **Run the cells in the notebook** to follow the step-by-step process of building the classifier.

## Detailed Workflow

### 1. Data Cleaning
- Loaded the dataset and performed initial exploration to understand its structure.
- Removed unnecessary columns, renamed columns for clarity, handled missing values, and removed duplicates.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the distribution of target labels (spam vs. ham).
- Explored text characteristics such as the number of characters, words, and sentences in each message.

### 3. Text Preprocessing
- Tokenized the text data and prepared it for model building.
- Used techniques like tokenization and normalization to convert text data into numerical features suitable for machine learning models.

### 4. Model Building
- Split the dataset into training and testing sets.
- Trained a machine learning model using scikit-learn's Multinomial Naive Bayes algorithm to classify messages.

### 5. Evaluation
- Evaluated the model's performance using metrics such as accuracy, confusion matrix, and classification report to assess its effectiveness.

### 6. Improvement
- Improved the model by tuning hyperparameters using GridSearchCV.
- Compared different algorithms to select the best-performing model.

### 7. Deployment
- Serialized and saved the trained model for deployment.
- Prepared the model for integration into a production environment.

## Technology Stack
- **Programming Language**: Python
- **Libraries**:
  - pandas for data manipulation
  - numpy for numerical operations
  - scikit-learn for machine learning algorithms
  - matplotlib for data visualization
  - nltk for natural language processing

## Contributing
Contributions are welcome! Please fork this repository, create a new branch, and submit a pull request. For major changes, open an issue first to discuss what you would like to change.


