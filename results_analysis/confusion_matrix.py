import numpy as np
import matplotlib.pyplot as plt

RESULT_FILE_PATH = 'results/'
# RESULT_FILE_ID = 'classification_LandingGear_LightTS_UEA_ftM_sl3000_ll48_pl0_dm128_nh8_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_0'
# RESULT_FILE_ID = 'classification_LandingGearFull_RandomForest_UEA_ftM_sl3000_ll48_pl0'
RESULT_FILE_ID = 'classification_LandingGearFull_RandomForest_UEA_ftM_sl7990_ll48_pl0'
CLASS_NUM = 7
if __name__ == "__main__":
    true = np.load(RESULT_FILE_PATH + RESULT_FILE_ID + '/true.npy')
    pred = np.load(RESULT_FILE_PATH + RESULT_FILE_ID + '/pred.npy')

    # 计算混淆矩阵
    confusion_matrix = np.zeros((CLASS_NUM, CLASS_NUM), dtype=int)
    for t, p in zip(true, pred):
        confusion_matrix[t][p] += 1

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(CLASS_NUM)
    plt.xticks(tick_marks, [f'Class {i}' for i in range(CLASS_NUM)], rotation=45)
    plt.yticks(tick_marks, [f'Class {i}' for i in range(CLASS_NUM)])

    # 在格子中添加数字
    thresh = confusion_matrix.max() / 2.0
    for i in range(CLASS_NUM):
        for j in range(CLASS_NUM):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()