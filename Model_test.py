# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 17:32:52 2021

@author: jiangyt
"""

from Tools import *
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, add, AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential

"""
Weight Dict
"""
Weight = {'Resnet50_448':"./model_checkpoints/ResNet50_448_checkpoints/20218131038.h5",
          'MobileNet_224':"./model_checkpoints/MobileNet_224_checkpoints/202189956.h5",
          'Xception_448':"./model_checkpoints/Xception_448_checkpoints/2021810951.h5",
          'EfficientNet_B0_320':"./model_checkpoints/EfficientNetB0_320_checkpoints/2021871045.h5",
          'DenseNet121_448':"./model_checkpoints/DenseNet121_448_checkpoints/2021891655.h5"}

"""
Load model
"""
df = pd.read_excel('./AI-Physician Comparasion Dataset.xlsx')
# df = pd.read_csv('/home/joe/Project/Breast_new/20210805_b_m_Xception_train/df_test_small.csv')

"""
Eval each model
"""
for key in Weight.keys():
    if key == 'Resnet50_448':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        backbone_model= keras.applications.resnet50.ResNet50(include_top=False, weights=None, input_tensor=None,
                                                        input_shape=(448, 448, 3), pooling=None, classes=2)
    elif key == 'MobileNet_224':
        from tensorflow.keras.applications.mobilenet import preprocess_input
        backbone_model=  keras.applications.mobilenet.MobileNet(include_top=False, weights=None, input_tensor=None,
                                                        input_shape=(224, 224, 3), pooling=None, classes=2)
    elif key == 'Xception_448':
        from tensorflow.keras.applications.xception import preprocess_input
        backbone_model= keras.applications.xception.Xception(include_top=False, weights=None, input_tensor=None,
                                                        input_shape=(448, 448, 3), pooling=None, classes=2)
    elif key == 'EfficientNet_B0_320':
        from tensorflow.keras.applications.efficientnet import preprocess_input
        backbone_model= keras.applications.efficientnet.EfficientNetB0(include_top=False, weights=None, input_tensor=None,
                                                        input_shape=(320, 320, 3), pooling=None, classes=2)
    elif key == 'DenseNet121_448':
        from tensorflow.keras.applications.densenet import preprocess_input

        backbone_model = keras.applications.densenet.DenseNet121(include_top=False, weights="imagenet",
                                                                 input_tensor=None,
                                                                 input_shape=(448, 448, 3), pooling=None, classes=2)
    else:
        print('Error: No model weight find')
    test_model = Sequential()
    test_model.add(backbone_model)
    test_model.add(GlobalAveragePooling2D())
    test_model.add(Dense(2, activation='softmax', name='fc1'))
    test_model.load_weights(Weight[key])

    test_model.summary()

    y_true = []
    y_pred = []

    for i in range(len(df)):
        y_true.append(df['malignancy'][i])
        x = Image.open(df['path'][i])
        x = np.array(x)
        x = zero_pad(x,int(key.split('_')[-1]))
        x = preprocess_input(x)
        x = x.reshape(1,x.shape[0],x.shape[1],x.shape[2])
        y_pred.append(test_model.predict(x))
    

    y_pred = np.array(y_pred)
    y_pred = y_pred.reshape(y_pred.shape[0],2)
    y_pred_1 = y_pred[:,1]

    thresh_0=get_auc(0, np.array(y_true), np.array(y_pred_1), 'Malignancy', plot=False)
    y_pred_comp_lvl=[1 if y>thresh_0 else 0 for y in y_pred_1]
    cm_comp=confusion_matrix(y_true, y_pred_comp_lvl)

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout(pad=2, w_pad=2.)
    fig.set_figheight(8)
    fig.set_figwidth(7)
    thresh_0=get_auc(axes[0, 0], np.array(y_true), np.array(y_pred_1), 'Performance of {}'.format(key))
    thresh_AP=get_precision_recall(axes[0, 1], np.array(y_true), np.array(y_pred_1), 'Malignancy=0 vs 1')
    plot_confusion_matrix(axes[1, 0], cm_comp, ["0", "1"], title='Malignancy', normalize=False)
    plot_confusion_matrix(axes[1, 1], cm_comp, ["0", "1"], title='Malignancy (normalized)')
    print('f1 score is: {:.3f}'.format(f1_score(y_true, y_pred_comp_lvl)))
