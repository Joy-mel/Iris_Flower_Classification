# Iris Flower Species Classification

A machine learning project that classifies Iris flowers (*setosa*, *versicolor*, *virginica*) 
using sepal and petal measurements, built with Scikit-learn.

## What it does

- Loads and explores the classic Iris dataset (150 samples, 4 features, 3 balanced classes)
- Generates EDA visualizations: scatter plots, pairplot, correlation heatmap
- Trains and compares 5 classifiers: Logistic Regression, KNN, SVM, Decision Tree, Random Forest
- Evaluates every model with test accuracy, 5-fold cross-validation, and a full classification report
- Visualizes the best model's confusion matrix
- Plots 2D decision boundaries (petal length/width vs. sepal length/width) to show *why* petal 
  measurements separate the species far better than sepal measurements
- Saves every chart and a results summary to `outputs/`

## Project structure
Iris_Classification/
├── .venv/
├── archive/
│   └── Iris.csv
├── iris_classification.ipynb
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md

## Setup (VS Code)

1. Clone the repo:
```bash
   git clone <your-repo-url>
   cd Iris_Classification
```

2. (Optional but recommended) Create a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Open `iris_classification.ipynb` in VS Code and run all cells (top to bottom).

5. Charts and `results_summary.txt` will appear in the `outputs/` folder, and also render inline 
   in the notebook.

## Results

On the held-out test set (20%, stratified), the linear SVM achieved **100% accuracy**, confirmed by 
5-fold cross-validation. Logistic Regression and KNN followed at ~93%.

The decision boundary plots show why: petal length and width almost perfectly separate all three 
species on their own, while sepal length and width leave versicolor and virginica overlapping — 
which is where a simpler model would be more likely to make mistakes.

## Dataset

The classic [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris): 150 samples across 3 
species, 4 measurements each (sepal length/width, petal length/width) in centimeters.