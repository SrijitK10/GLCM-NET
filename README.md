# GAN-Generated Face Image Detection

## Project Overview
This project aims to detect GAN-generated face images using machine learning techniques. The primary method involves feature extraction using the Gray Level Co-occurrence Matrix (GLCM) combined with deep learning models, including InceptionV3, ResNet50, Support Vector Machine (SVM), and a custom Convolutional Neural Network (CNN). The objective is to distinguish between real and synthetic face images, contributing to the fight against misinformation and fraudulent activities involving AI-generated faces.

## Dataset
The dataset consists of 32,000 images:
- **Real Images**: 16,000 images from the FFHQ dataset.
- **GAN-Generated Images**: 16,000 images generated using StyleGAN.

### Dataset Split:
- **Training Set**: 20,000 images (62.5%)
- **Validation Set**: 4,000 images (12.5%)
- **Testing Set**: 8,000 images (25%)

## Methodology
1. **Feature Extraction**: 
   - Gray Level Co-occurrence Matrix (GLCM) is used to capture texture-based differences between real and synthetic images.
2. **Image Preprocessing**:
   - Images are converted to grayscale and resized to 256x256 pixels.
   - Normalization is applied for CNN-based models, and feature vectors are flattened for SVM.
3. **Model Training**:
   - Models are trained for 30 epochs using Binary Cross-Entropy as the loss function.
   - Adam optimizer is used with a learning rate of 0.001.
   - SVM is fine-tuned using Grid Search.

## Models Used
- **InceptionV3**: Achieved the best performance with an accuracy of 93.50%.
- **ResNet50**: Close second with 92.33% accuracy.
- **SVM**: Achieved 87.00% accuracy using GLCM features.
- **Custom CNN**: Performed at 87.75% accuracy.

## Performance Metrics
Evaluation metrics used:
- Accuracy
- Precision, Recall, F1-score
- Matthews Correlation Coefficient (MCC)
- Balanced Accuracy
- Cohen’s Kappa
- Log Loss
- Jaccard Index

## Results
- **Best Performing Model**: InceptionV3 with 93.50% accuracy, highest MCC (0.87), and best balanced accuracy (94%).
- **ResNet50**: Performed well in reducing false positives.
- **SVM and CNN**: Showed lower precision and recall values but still provided useful insights.

## Future Enhancements
- **Integration of newer GAN architectures (BigGAN, StyleGAN3) for better generalization.**
- **Improved feature extraction techniques using wavelet transformations and deep texture analysis.**
- **Real-time detection capability using model optimization techniques such as pruning and quantization.**
- **Cross-domain validation to test model effectiveness in various real-world applications.**

## Conclusion
This project successfully demonstrates that deep learning models combined with GLCM-based feature extraction can effectively detect GAN-generated face images. The comparative analysis provides insights into model selection for different application needs, emphasizing the importance of advanced detection mechanisms in combating synthetic media fraud.

## Installation & Usage
1. Install dependencies:
   ```sh
   pip install tensorflow keras numpy opencv-python scikit-learn
   
2.   Run the model training script:
     ```sh
     python train.py

4. Evaluate the model:
    ```sh
    python evaluate.py

## Contact
For any queries, please contact:

[Srijit Kundu](https://github.com/SrijitK10) 

[Rudrajit Dutta](https://github.com/Prorudrajit23)
