# Predicting League of Legends Match Outcomes Based on Data from the First 10 Minutes

This repository contains the code and analysis developed as part of a Bachelor's thesis investigating the predictability of League of Legends match outcomes using data collected exclusively from the initial 10 minutes of gameplay.

## Project Description

League of Legends is a complex strategic game where early advantages can significantly influence the final outcome. This project explores the extent to which events and statistics accumulated within the first 10 minutes of a professional or high-elo match can serve as reliable indicators for predicting which team will ultimately win.

The analysis involves processing match data, engineering relevant features from the early game statistics, and training machine learning models to perform the prediction task.

## Motivation

The early game phase in League of Legends sets the stage for the rest of the match. Understanding which early game metrics are most indicative of future success can provide valuable insights for players, coaches, and analysts. This thesis aimed to quantify this relationship and build a predictive model based solely on information available relatively early in the game.

## Data

The project utilizes match data focusing specifically on statistics recorded within the first 10 minutes of each game. This includes metrics related to:

* Gold acquired by each team/player
* Experience gained
* Number of kills and assists
* Objective control (e.g., dragons, heralds)
* Lane control and minion scores
* Other relevant early-game events

The dataset used for this analysis is the publicly available **"League of Legends Diamond Ranked Games 10 min"** dataset from Kaggle.

**Source:** [League of Legends Diamond Ranked Games 10 min on Kaggle](https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min)

## Methodology

The project follows a standard data science pipeline:

1.  **Data Collection & Loading:** Gathering and loading the raw match data.
2.  **Data Preprocessing:** Cleaning the data, handling missing values, and structuring it for analysis.
3.  **Feature Engineering:** Creating relevant features from the raw early-game statistics that are hypothesized to be predictive of the outcome.
4.  **Exploratory Data Analysis (EDA):** Analyzing the distributions of key features and their correlation with the match outcome.
5.  **Model Training:** Training various machine learning models (e.g., Logistic Regression, RandomForest, Gradient Boosting) on the processed data.
6.  **Model Evaluation:** Assessing the performance of the models using appropriate metrics (e.g., Accuracy, Precision, Recall, F1-Score, AUC-ROC) to determine how well early-game data can predict the final outcome.

## Results & Findings

As the core component of this Bachelor's thesis, a machine learning model was trained to predict League of Legends match outcomes based on data exclusively from the first 10 minutes of gameplay. The evaluation on a test set provided insights into both the overall predictability and the significance of specific early-game factors.

**Overall Model Performance:**

* **Accuracy:** The model achieved an overall classification **accuracy of 73% (0.73)** on the test set, indicating that it correctly predicted the winning team in a significant majority of cases using only early game data.
* **ROC AUC:** The **Area Under the ROC Curve (AUC)** was calculated to be **0.82**. This value demonstrates a strong ability to discriminate between winning and losing teams based on the early game features, performing substantially better than random chance (AUC = 0.5).

**Insights from Feature Analysis:**

Beyond overall performance, the analysis revealed how specific early-game statistics influence the probability of winning:

* **Gold and Experience:** Both gold (`blueGoldDiff`) and total experience (`blueTotalExperience`) differences between the teams show a significant impact, but this influence becomes substantial primarily at large differences in these values.
    * For every additional unit of gold difference favoring the blue team, the chance of the blue team winning increases by approximately **0.0425%**. While small per unit, a lead of 1000 gold increases the win probability by roughly **47%**. This highlights how a significant gold advantage translates into better itemization and combat power.
    * Similarly, for every unit of experience difference favoring the blue team, the chance of winning increases by approximately **0.0188%**. The effect of experience becomes significant with large differences, as higher-level characters gain combat strength and abilities faster.
* **Dragons:** Securing dragons has a considerable impact. The analysis indicates that the **first dragon** killed by a team increases their chance of winning the game by approximately **39%**. This underscores the strategic importance of early objective control, particularly dragons, due to their long-term benefits.
* **Wards:** The sheer **number of wards placed** does not appear to have a large influence on the final outcome. While each additional ward minimally increases the win chance, this finding suggests that the *strategic placement* and *quality* of vision control are more important than the total quantity of wards.
* **Destroyed Red Towers:** Interestingly, the analysis indicated that the destruction of a tower by the **red team (`redTowersDestroyed`)** actually **decreases** their chance of winning. This finding is counter-intuitive, as destroying an enemy structure is typically considered an advantage. 

These findings provide granular insights into which early-game metrics are most predictive and how their magnitude impacts the likelihood of winning, complementing the overall model performance metrics.

## Technologies Used

* **Python:** The primary programming language.
* **pandas:** For data manipulation and analysis.
* **numpy:** For numerical operations.
* **matplotlib:** For data visualization.
* **seaborn:** For enhanced statistical data visualizations.
* **scikit-learn (sklearn):** For machine learning model implementation, training, and evaluation.

## Repository Contents

* `league10minsAnalysis.ipynb`: The main Jupyter Notebook containing the data loading, preprocessing, analysis, model training, and evaluation code.
* *(List any other files, e.g., data files like `matches_data.csv`, helper scripts, etc.)*

## How to Run

1.  Clone this repository:
    ```bash
    git clone https://github.com/Tutaj/League-of-legends-10-minutes-win-prediction
    ```
2.  Ensure you have Python and the necessary libraries installed. You can install them via pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  Navigate to the repository directory.
4.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook league10minsAnalysis.ipynb
    ```
    or
    ```bash
    jupyter lab league10minsAnalysis.ipynb
    ```
5.  Run through the cells in the notebook to execute the analysis and see the results.

## Bachelor's Thesis Context

This project was developed as the practical component of a Bachelor's thesis.
