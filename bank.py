import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- Load dataset ---
df = pd.read_excel("bank.xlsx")

# --- Clean column names ---
df.columns = df.columns.str.strip().str.lower()

# --- Rename target column ---
df = df.rename(columns={"y": "target"})

# --- Encode target ---
df["target"] = df["target"].map({"yes": 1, "no": 0})

# --- Select categorical columns safely ---
categorical_cols = [col for col in df.select_dtypes(include=["object"]).columns if col != "target"]

# --- One-hot encode ---
df_enc = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# --- Split features and target ---
X = df_enc.drop("target", axis=1)
y = df_enc["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Train Decision Tree ---
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# --- Evaluate ---
y_pred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Feature Importance ---
importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Features:\n", importances.head(10))

# --- Visualize Tree ---
plt.figure(figsize=(25,12))
plot_tree(dt, feature_names=X.columns, class_names=["No", "Yes"], filled=True, fontsize=8)
plt.show()
