## Gender and Age Prediction using CNN

This project utilizes Convolutional Neural Networks (CNN) to predict the gender and age of individuals from facial images. The dataset used for training and testing is the UTKFace dataset available on Kaggle, which contains a large collection of face images categorized by age, gender, and ethnicity.

### Dataset
The UTKFace dataset is a large-scale face dataset with long age span (ranging from 0 to 116 years old). The images cover large variation in pose, facial expression, illumination, and imaging conditions. This dataset is ideal for training machine learning models to predict age and gender from facial images.

- **Dataset Source:** [UTKFace on Kaggle](https://www.kaggle.com/jangedoo/utkface-new)

### Project Structure
The project repository is organized as follows:

- **`data/`**: Contains the dataset (not included in this repository due to size).
- **`models/`**: Stores trained models (not included in this repository).
- **`notebooks/`**: Jupyter notebooks for data exploration, model training, and testing.
- **`src/`**: Python scripts for data preprocessing, model architecture, training, and evaluation.
  
### Requirements
To run the project, you need the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Usage
1. **Data Preparation**: 
   - Download the UTKFace dataset from [Kaggle](https://www.kaggle.com/jangedoo/utkface-new) and place it in the `data/` directory.
   - Preprocess the dataset using the `preprocess_data.py` script to prepare it for model training.

2. **Model Training**:
   - Run the `train_model.py` script to train the CNN model on the preprocessed dataset.
   - The trained model will be saved in the `models/` directory.

3. **Model Evaluation**:
   - Use the `evaluate_model.py` script to evaluate the trained model on test data.
   - This will provide accuracy metrics for gender and age prediction.

4. **Inference**:
   - Use the trained model to make predictions on new images using `predict.py`.

### References
- Dataset: [UTKFace on Kaggle](https://www.kaggle.com/jangedoo/utkface-new)
- TensorFlow: [tensorflow.org](https://www.tensorflow.org/)
- Keras: [keras.io](https://keras.io/)

### Acknowledgements
- The UTKFace dataset is provided by Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2017) in their paper [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

### License
This project is licensed under the [MIT License](LICENSE).

