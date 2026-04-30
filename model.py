import code_train as data
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score, confusion_matrix
import code_extract_mfccs as cemfccs


#X_train, y_train_sk, X_test_sk, y_test_sk = code_train.load_data()

print("training an SVC model")
svm_model = SVC(kernel="rbf", C=10)
svm_model.fit(data.X_train_sk, data.y_train_sk)

# Evaluation
y_pred_svm = svm_model.predict(data.X_test_sk)

print("SVM Classification Report:")
print(classification_report(data.y_test_sk, y_pred_svm, target_names=data.label_encoder_sk.classes_))

print("SVM Accuracy:", accuracy_score(data.y_test_sk, y_pred_svm))




#loading my own audio file and predicting the label with the trained SVM model
my_audio_path = "recordings/yes/1.wav"
my_mffcs = cemfccs.extract_mfccs(my_audio_path, sr=16000, n_mfcc=13)
custom_prediction = svm_model.predict([my_mffcs.mean(axis=1)])
print(f"Predicted label for {my_audio_path}: {data.label_encoder_sk.inverse_transform(custom_prediction)[0]}")