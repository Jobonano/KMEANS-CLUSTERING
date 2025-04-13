# 💳 Customer Segmentation using KMeans Clustering

This project applies **unsupervised learning** techniques to analyze and segment customers based on their credit card usage patterns. The data is preprocessed, cleaned, and clustered using the **KMeans algorithm**, and the results are visualized using **Principal Component Analysis (PCA)**. The primary goal is to identify distinct groups of customers for better marketing strategies, product targeting, and customer relationship management.

---

## 📁 Dataset Overview

The dataset used is a **credit card customer dataset**, which includes the following features:

- **CUST_ID**: Unique identifier for each customer.
- **BALANCE**: The average balance in the customer's account.
- **PURCHASES**: Total purchase amount.
- **ONEOFF_PURCHASES / INSTALLMENTS_PURCHASES**: Type-specific purchases.
- **CASH_ADVANCE**: Amount of cash advance taken.
- **FREQUENCIES**: Behavioral frequencies (e.g., how often purchases or cash advances are made).
- **CREDIT_LIMIT**: Maximum credit limit.
- **PAYMENTS**: Total payments made.
- **MINIMUM_PAYMENTS**: Minimum payments due.
- **PRC_FULL_PAYMENT**: Percentage of full payments made.
- **TENURE**: Number of months of activity.

---

## 🧼 Data Preprocessing

- **Missing Values**:
  - `CREDIT_LIMIT`: 1 missing value.
  - `MINIMUM_PAYMENTS`: 313 missing values.
  - These were imputed using the **median strategy** with `SimpleImputer` from scikit-learn.
- **Dropped**: `CUST_ID` column, as it is non-numeric and not useful for clustering.

---

## 📊 Clustering: KMeans

- **Algorithm**: KMeans (Elkan variant for efficiency)
- **Parameters**:
  - `n_clusters=2`
  - `init='k-means++'`
  - `max_iter=300`
  - `random_state=101`
- **Cluster Assignment**: Done using `kmeans.predict()`
- **Silhouette Score**:  
  **`0.5117`** — This indicates a reasonably good clustering structure. A score above 0.5 suggests that the clusters are well-separated and meaningful.

---

## 📉 Dimensionality Reduction: PCA

To visualize the clustering results, **Principal Component Analysis** was applied to reduce the feature space to 2 components:

```python
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df_imputed)
```

---

## 📌 Visualization

The final visualization uses a scatter plot to show the two customer clusters:

![Clustering Result](https://via.placeholder.com/600x300?text=Cluster+Visualization)

Each point represents a customer, colored by their assigned cluster.

---

## 🧠 Insights & Interpretation

- The clustering process revealed **2 distinct customer segments**.
- Based on PCA-reduced space, the two clusters are **well-separated**, as supported by the silhouette score.
- These customer groups likely differ in:
  - Spending patterns (e.g., frequent vs. infrequent purchase behavior).
  - Use of credit (e.g., high vs. low cash advances).
  - Payment behavior (e.g., full vs. partial payments).

---

## 📦 Technologies Used

- **Python 3**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib** – Visualization
- **Scikit-learn** – Machine learning (KMeans, PCA, imputation, metrics)

---

## 🔧 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook or Python script:

```bash
jupyter notebook
```

---

## 📈 Future Improvements

- Try different values for `k` and use the **Elbow Method** or **Silhouette Analysis** to determine the optimal number of clusters.
- Incorporate **feature scaling** for improved clustering results.
- Apply **other clustering algorithms** like DBSCAN or Hierarchical Clustering.
- Perform more **detailed profiling** of clusters for actionable business insights.

---

## 📬 Contact

For questions, suggestions, or contributions, feel free to reach out:

**Author**: Andrew Joshua Adole
**Email**: joshuaandrew159@gmail.com 
**GitHub**: [@jobonano](https://github.com/jobonano)
