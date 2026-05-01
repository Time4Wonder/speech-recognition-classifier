# Summary
## implementation of MFCC extractor
During the implementation, I found it surprising that Mel-frequency cepstral coefficients (MFCCs) show very little visual difference when plotted as graphics. While it is difficult for the human eye to distinguish patterns in these plots, they provide a highly efficient feature set for computational analysis and machine learning models.

## Data Preperation and Model Evaluation
The data loading process was structured based on the lecture exercise notes. To improve the user experience during long-running tasks, I integrated the tqdm library to provide a real-time progress bar.


**Scope Adjustment:**
- Initial Approach: I started with a dataset containing 10 different words. However, this significantly increased loading times, and the initial models lacked the complexity to generalize well across so many classes.

- Refined Approach: To ensure a reliable and robust result, I pivoted to a binary classification task, focusing specifically on the words "yes" and "no".


## Model Evaluation & Results
I compared two different architectures: Random Forest and Support Vector Machine (SVM). Both models performed consistently well, achieving metrics (Precision, Recall, and F1-score) of approximately 0.90.

###  Model Classification Report (Random Forest Model)

| x | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| no | 0.90 | 0.93 | 0.91 | 592 |
| yes | 0.93 | 0.90 | 0.91 | 606 |

###  Model Classification Report (SVM)

| x | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| no | 0.89 | 0.92 | 0.90 | 592 |
| yes | 0.92 | 0.89 | 0.90 | 606 |


## Recordings
To test the model's practical application, I used original recordings. I gathered these samples in a social environment (a bar), which provided a challenging acoustic background.

**Results:** Despite the ambient noise, the model performed exceptionally well, all custom recordings were classified correctly. This successful "field test" confirmed the reliability of the binary classification approach.