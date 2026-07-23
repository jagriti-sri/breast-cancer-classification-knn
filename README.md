# Breast Cancer Classification using K-Nearest Neighbors (KNN)

## Author

**Name:** Jagriti Srivastava

**Registration Number:** 23MIP10068

**Application Number:** IN26011508

**Batch Number:** 1A

**Email ID:** <jagriti.23mip10068@vitbhopal.ac.in>

---

# Objective

The objective of this project is to build a K-Nearest Neighbors (KNN) classification model to predict whether a breast tumor is Malignant (M) or Benign (B) based on diagnostic measurements from the Breast Cancer Wisconsin Diagnostic Dataset.

---

# Dataset Link

- [Kaggle: Breast Cancer Wisconsin Diagnostic Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

# Libraries Used

- pandas
- scikit-learn

---

# Methodology

### Data Understanding
- Loaded the Breast Cancer Wisconsin Diagnostic Dataset using Pandas.
- Displayed the first five records.
- Identified numerical features and the target variable.
- Displayed dataset information and summary statistics.

### Data Preprocessing
- Checked for missing values.
- Removed unnecessary columns such as **id** and **Unnamed: 32**.
- Encoded the target variable using Label Encoding.
- Standardized the feature values using **StandardScaler**.
- Split the dataset into **80% training** and **20% testing** datasets.

### Model Development
- Built a **K-Nearest Neighbors (KNN)** classifier with **K = 5**.
- Trained the model on the training dataset.
- Predicted the class labels for the testing dataset.

### Model Evaluation
- Evaluated the model using Accuracy, Precision, Recall, and F1-Score.
- Generated a Confusion Matrix to analyze prediction performance.

---

# Results

- **Accuracy Score:** **94.74%**
- **Precision:** **0.9302**
- **Recall:** **0.9302**
- **F1-Score:** **0.9302**

### Confusion Matrix

```
[[68  3]
 [ 3 40]]
```

---

# Conclusion

This project successfully developed a K-Nearest Neighbors (KNN) model to classify breast tumors as malignant or benign. After preprocessing the dataset by removing unnecessary columns, encoding the target variable, and standardizing the features, the model achieved an accuracy of **94.74%**, demonstrating excellent classification performance. Feature scaling played a crucial role in improving KNN because the algorithm relies on distance calculations between data points. One limitation of KNN is that it becomes computationally expensive for large datasets and its performance is highly dependent on selecting an appropriate value of **K**.
