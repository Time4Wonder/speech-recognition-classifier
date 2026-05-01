# Binary Speech Classifier (Yes/No)

This repository contains a lightweight Python script designed to classify spoken words—specifically **"yes"** and **"no"**. The project serves as a practical introduction to Natural Language Processing (NLP) using the popular `scikit-learn` library.

---

## Getting Started

### Prerequisites
Before running the scripts, ensure you have Python installed. You can install all necessary dependencies using the provided `requirements.txt`:

pip install -r requirements.txt

---

## Data Preparation

The dataset is not included in this repository due to its size. Please follow these steps to set up the environment:

1. **Download:** Get the dataset from [TensorFlow Speech Commands](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
2. **Extract:** Uncompress the downloaded archive.
3. **Structure:** Place the files in a `data/` directory. Your folder structure should look like this:

| Path | Description |
| :--- | :--- |
| `data/speech_commands_v0.02/` | Root folder for the speech data |
| `data/speech_commands_v0.02/yes/` | Audio samples for "yes" |
| `data/speech_commands_v0.02/no/` | Audio samples for "no" |

---

## Further Information

For a deeper dive into the methodology, model performance, and technical details, please refer to the:

 **[summary.md](summary.md)**