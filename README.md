# T20I Cricket Second Innings Chase Prediction

## Project Overview
This project uses machine learning to predict the success of a second innings chase in T20 International (T20I) cricket matches. It leverages ball-by-ball data to analyze key features such as runs scored, wickets lost, balls remaining, and target score to forecast whether the chasing team will win. The project evaluates multiple classifiers, with logistic regression achieving the highest accuracy, and provides phase-specific models for Powerplay, Middle Overs, and Final Overs.

## Dataset
The dataset is sourced from a CSV file (`ball_by_ball_it20.csv`) containing ball-by-ball details of T20I matches. It includes:
- **Total Matches**: 1,842
- **Data Points**: 199,749
- **Key Features**:
  - Runs From Ball
  - Innings Runs
  - Innings Wickets
  - Balls Remaining
  - Target Score
  - Total Batter Runs
  - Total Non Striker Runs
  - Batter Balls Faced
  - Non Striker Balls Faced
- **Target Variable**: `Chased Successfully` (binary: win/loss)

## Prerequisites
- Python 3.x
- Required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn sklearn xgboost
  ```

## Project Structure
- **Data Preprocessing**:
  - Loads and filters second innings data.
  - Drops rows where the chase outcome is known (`Balls Remaining == 0`).
  - Splits data into training (pre-2018) and test (2018 and later) sets based on a cutoff date of January 1, 2018.
  - Scales features using `StandardScaler`.

- **Modeling**:
  - Evaluates multiple classifiers: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
  - Logistic Regression is selected for its simplicity and highest accuracy (82.6%).
  - Phase-specific models are trained for:
    - Powerplay (Overs 1-6): 76.15% accuracy
    - Middle Overs (Overs 7-15): 83.63% accuracy
    - Final Overs (Overs 16-20): 91.24% accuracy

- **Visualization**:
  - Plots logistic regression coefficients to show feature importance.

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - Place the `ball_by_ball_it20.csv` file in the project directory or update the file path in the script.

4. **Run the Notebook**:
   - Open the Jupyter notebook (`T20I_Chase_Prediction.ipynb`) and execute all cells.
   - Alternatively, run the Python script if provided:
     ```bash
     python main.py
     ```

## Results
- **Overall Accuracy**:
  - Logistic Regression: 82.6%
  - Random Forest: 79.7%
  - Gradient Boosting: 81.4%
  - XGBoost: 79.1%
- **Phase-Specific Accuracy**:
  - Powerplay: 76.15%
  - Middle Overs: 83.63%
  - Final Overs: 91.24%
- The model performs best in the Final Overs, where the outcome is more predictable, and worst in the Powerplay, reflecting higher uncertainty early in the innings.

## Adaptability
- The code can be modified to use other datasets, such as IPL or ODI data, by updating the input file and adjusting preprocessing steps.
- To predict live games, set the cutoff date to the previous day.

## Future Improvements
- Incorporate additional features (e.g., player-specific stats, venue conditions).
- Experiment with deep learning models for improved accuracy.
- Optimize hyperparameters using grid search or cross-validation.
