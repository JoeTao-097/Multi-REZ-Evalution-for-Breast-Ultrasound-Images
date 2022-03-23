# Multi-REZ-Evalution-for-Breast-Ultrasound-Images
We provide code, model weights (EfficientNetB0, MobileNet, Xception, ResNet50 and DenseNet121, for 224\*224, 320\*320, 448\*448 input resolution) and a test set with 300 breast ultrasound images for physician-ai comparasion in this respository.

All models output predictions have been saved in 'model and physicians performance on AI-Physician Comparasion set' dictionary.

3oo test images for physician-ai comparasion are saved in 'Images of AI-Physician Comparasion'.

Pathological results and birads classes of these 300 images could be found in AI-Physician Comparasion Dataset.xlsx.

Staticial figures are saved in 'Figures'.

15 model weights are available from https://drive.google.com/drive/folders/1Vk7vMobeO0ZqbZwR1ErmDxP4PnAPEzyq?usp=sharing. 


To test model performance on 300 image test set, run Model_test.py.

To draw models comparasion figure, run Model Perfomance Comparasion.py.

To draw model-physician performance comparasion of Wilcoxon test figures, run Model-Physician Performance Comparasion.py.

To draw AUROC curve of AI and physicians, run AI-Physician-Comparasion.py
