# Loan Default Prediction

## 1. Problem Definition

We selected the **Loan Prediction Dataset** because it addresses a real-world challenge: loan approval decision-making. Incorrect approvals can lead to financial loss, while false rejections hurt customer relationships and revenue. Our goal is to develop a reliable machine learning model to support objective, efficient, and scalable loan decisions.


---

## 2. Exploratory Data Analysis/Visualization – Initial

### Key Observations:
- The dataset had 100,000 rows and 19 columns.
- **Target variable**: `Loan Status` (Fully Paid or Charged Off).
- Discovered class imbalance: We investigated the distribution of the target variable (Loan Status) and found it to be imbalanced, with more "Fully Paid" loans than "Charged Off" loans.  ~83% Fully Paid vs 17% Charged Off.
- **Feature Relationship:** Heatmaps and box plots were used to explore the relationship between features and the target variable. We identified key features that had strong correlations with loan status.
- **Data Distribution:** We analyzed the distribution of numerical features like Credit Score, Annual Income, Current Loan Amount, and Monthly Debt using histograms and box plots. This helped us understand data patterns, identify outliers, and guide our data cleaning process.
- **Categorical Features:**  We examined categorical features like "Home Ownership" and "Purpose" and analyzed their influence on loan defaults.
- **Missing values**: Credit Score, Annual Income (~19%); Months since last delinquent (~53%).
- Notable anomalies:
  - `Current Loan Amount` had placeholder value 99,999,999 in ~11.5% of records.
  - Some credit scores exceeded 850, an unrealistic value.

### Visual tools used:
- Histograms, KDEs, boxplots.
- Categorical distributions and heatmaps.
- Target variable distributions by features like `Home Ownership`, `Purpose`, `Term`.


---

## 3. Data Preparation and Cleaning

- **Unnecessary Columns Removal:** Columns like "Loan ID" and "Customer ID" were removed to prevent bias and improve model efficiency.
- **Missing Value Handling:** Rows with missing values in critical features (Credit Score, Annual Income) were dropped. Remaining missing values in numerical columns were imputed using the median.
- **Outlier Treatment:** Outliers in Credit Score and Current Loan Amount were identified and either removed or adjusted using domain knowledge researched and data distribution analysis.

---

## 4. Feature Engineering
- **Feature Engineering:**  To enhance model performance, we engineered two new features:
    - **Loan-to-Income Ratio (LTI)**: Calculated by dividing the loan amount by the annual income.
    - Captures how much loan is being taken relative to income.
    - High LTI = higher repayment burden = greater risk.
- **Credit Score to Loan Ratio**: Calculated by dividing the credit score by the loan amount.
    - Explored but discarded due to counterintuitive behavior (higher ratios showed higher defaults).


---

## 5. Post-Cleaning EDA

- After cleaning, new insights emerged:
  - Credit Score vs Loan Status: Higher scores linked to repayment.
  - Current Loan Amount: Higher values more often defaulted.
  - LTI: Higher ratios linked to Charged Off status.
  - Confirmed removal of extreme outliers and invalid placeholders.
  - Categorical variables like `Term` and `Home Ownership` showed clearer relationships post-cleaning.


---

## 6. Data Preprocessing

- **Target Encoding**: `Loan Status` mapped to binary (1 = Fully Paid, 0 = Charged Off).
- **Feature Selection**: Focused on Credit Score, Income, Loan Amount, LTI, Term, and categorical variables like Home Ownership.
- **Train-Test Split**: 80/20 split with stratification to preserve class ratios.
- **Handling Class Imbalance**:
  - Manual upsampling.
  - **SMOTENC** for categorical-aware oversampling.
- **Feature Scaling**:
  - `StandardScaler` for Logistic Regression.
  - `MinMaxScaler` where needed for models sensitive to scale.


---

## 7. Machine Learning Techniques

We employed the following machine learning models:

1. **Dummy Classifier:** Used as a baseline model to establish a performance benchmark.
2. **Logistic Regression:** This simple and interpretable model was used as an initial predictive model.
3. **Random Forest:** A powerful ensemble method capable of handling complex relationships and imbalanced data.
4. **XG boost:** Known for its accuracy and handling of complex relationships

### Techniques:
- GridSearchCV for hyperparameter tuning.
- SMOTENC balancing improved minority class recall.
- Confusion matrices used to evaluate misclassification impact.
- Random Forest had best balance of precision, recall, and interpretability.

## Something New

To tackle the class imbalance problem, we explored advanced data balancing techniques:

- **SMOTENC:** A robust data augmentation method that creates synthetic data to balance the dataset.

Further, we implemented XGBoost, an advanced machine learning model known for its robustness and high performance.


---

## 8. Individual Contributions

- **Member 1:** Data preparation and cleaning, Exploratory data analysis, machine learning
- **Member 2:** Data preparation and cleaning, machine learning, SMOTENC and XGBOOST research.
