import code_train
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


X_train, y_train_sk, X_test_sk, y_test_sk = code_train.load_data()

print("training an SVC model")
svm_model = SVC(kernel="rbf", C=10)
svm_model.fit(X_train, y_train_sk)

# Evaluation



y_pred_svm = svm_model.predict(X_test_sk)
