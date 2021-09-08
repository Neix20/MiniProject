# Load Library
import os
import cv2
import math
import pickle
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.measure
import matplotlib.pyplot as plt

from glob import glob
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, accuracy_score

# Set the screen width to 120 pixels
screen_size = 118

# Default Dataset Folder Path
default_dataset_folder_path = "Dataset"

# Default Model Folder Path
default_model_folder_path = "Models"

# Default Prediction Folder Path
default_prediction_folder_path = "To_Predict"

def draw_line(num = 120):
    print("="*num)

def align_left(str, num = 45):
    print(" " * num, str)
    
menu_options = [
    ["Generate Dataset", "Model Result Analysis", "Predict Fruit Image", "Exit"],
    ["Generate Value for Single Image", "Show 300 Images", "Generate Dataset", "Exit"],
    ["Train with All Features", "Train with Color Features", "Train with Texture Features", "Train with Shape Features", "Precision-Recall Curve", "Roc Curve", "Feature Importance", "Save Model", "Exit"],
    ["Predict Fruit Image From To_Predict Folder", "Exit"]
]

# Get all types of fruit
def get_fruit_type(default_dataset_folder_path = "Dataset"):
    arr = glob(f'{default_dataset_folder_path}/*/')
    arr = [tmp_str.split("\\")[1] for tmp_str in arr]
    arr += ["Exit"]
    return arr

# Get all Fruit Image of Selected Fruit
def get_fruit_img_name(fruit_img, default_dataset_folder_path = "Dataset"):
    folder_path = f"{default_dataset_folder_path}\\{fruit_img}\\"
    arr = glob(folder_path + "*.jpg")
    arr = [tmp_str.split("\\")[2][:-4] for tmp_str in arr]
    return arr

# Remove White Background
def remove_background(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]

    mask = cv2.drawContours(threshed, cnt, 0, (0, 255, 0), 0)
    masked_data = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(cnt)
    dst = masked_data[y: y + h, x: x + w]

    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(dst_gray, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(dst)

    rgba = [r, g, b, alpha]
    dst = cv2.merge(rgba, 4)
    
    dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2RGB)

    return dst

# Convert Value to RGB
def convert_val_to_rgb(x):
    if x >= 0 and x < 64:
        return "00"
    elif x >= 64 and x < 128:
        return "55"
    elif x >= 128 and x < 192:
        return "AA"
    else:
        return "FF"

# Convert Value to Bin
def convert_val_to_bin(x):
    if x >= 0 and x < 64:
        return 0
    elif x >= 64 and x < 128:
        return 85
    elif x >= 128 and x < 192:
        return 170
    else:
        return 255
    
# Generate Color Features Pipeline
def pipeline_color(cv_img):
    img_arr = np.array(cv_img)
    img_flatten = img_arr.reshape(1, -1).T
    img_squeeze = np.squeeze(img_flatten)
    img_convert = np.vectorize(convert_val_to_rgb)(img_squeeze)
    img_2d_arr = img_convert.reshape(-1, 3)
    img_list_arr = img_2d_arr.tolist()
    convert_to_str = lambda x:"#"+"".join(list(map(str, x)))
    img_str_arr = [convert_to_str(x) for x in img_list_arr]
    new_arr = np.array(img_str_arr)
    tmp_dict = {a:b for (a,b) in zip(np.unique(new_arr, return_counts=True)[0], np.unique(new_arr, return_counts=True)[1])}
    return tmp_dict

# Generate Texture Features
def pipeline_texture(img_gray):
    tmp_dict = {}
    
    # Get Mean of image
    tmp_dict["Mean"] = np.mean(img_gray)
    
    # Get Variance of image
    tmp_dict["Variance"] = np.var(img_gray)
    
    # Get Entropy of Image
    tmp_dict["Entropy"] = skimage.measure.shannon_entropy(img_gray)
    
    glcm = greycomatrix(img_gray, [2], [0], 256, symmetric = True, normed = True)
    
    # Get Contrast of Image
    tmp_dict["Contrast"] = greycoprops(glcm, prop="contrast").item()
    
    # Get Homogeneity of Image
    tmp_dict["Homogeneity"] = greycoprops(glcm, prop="homogeneity").item()
    
    # Get Correlation of Image
    tmp_dict["Correlation"] = greycoprops(glcm, prop="correlation").item()
    
    # Get Energy of Image
    tmp_dict["Energy"] = greycoprops(glcm, prop="energy").item()
    return tmp_dict

# Generate Shape Features
def pipeline_shape(img_gray):
    # Apply Gaussian Blur to Image
    img_blur = cv2.GaussianBlur(img_gray, (7,7), 1)

    # Canny Edge Detection
    img_canny = cv2.Canny(img_blur, 50, 150)

    # Edge Detector
    kernel = np.ones((5, 5), dtype = np.uint8)
    img_dilate = cv2.dilate(img_canny, kernel, iterations = 2)

    contours, hier = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area, peri = 0, 0
    mu, huMoments = [], []
    
    for cnt in contours:
        
        # Get Area of Image from Contour
        area = cv2.contourArea(cnt)
        
        # Get Perimeter of Image From Contour
        peri = cv2.arcLength(cnt, True)
        mu = cv2.moments(cnt)
        
        # Get 7 Humoments From Contour
        huMoments = cv2.HuMoments(mu)
        
    for i in range(0,7):
        huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]) if abs(huMoments[i]) > 0 else 1)
    
    tmp_dict = {}
    tmp_dict["Area"] = area
    tmp_dict["Perimeter"] = peri
    for (i, huMoment) in enumerate(huMoments):
        tmp_dict[f"huMoment {(i + 1)}"] = huMoment[0]
    return tmp_dict

# Pipeline for Converting Image Path to OpenCV Image
def pipeline_img(img_path, threshold = 225, h = 100, w = 100):
    cv_img = cv2.imread(img_path)
    
    # Resize Image
    cv_img = cv2.resize(cv_img, (w,h),interpolation = cv2.INTER_AREA)
    
    # Remove White Background
    cv_img = remove_background(cv_img, threshold)
    
    # Convert Image from RGB to BGR
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # Transform CV Image from unint8 to float32
    img_tmp = cv_img / 255.0
    
    # Color Enhancement - Enhance the R, G, B Colors of the image
    r, g, b = cv2.split(img_tmp)
    img_sum = r + g + b
    CR, CG, CB = cv2.divide(r, img_sum), cv2.divide(g, img_sum), cv2.divide(b, img_sum)
    img_tmp = cv2.merge((CR, CG, CB))
    
    # Convert the image from float32 to uint8
    img_tmp = np.uint8(img_tmp * 255)
    
    # Convert the image to Gray Image
    img_gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to Image
    img_blur = cv2.GaussianBlur(img_gray, (7,7), 1)
    
    # Canny Edge Detection
    img_canny = cv2.Canny(img_blur, 50, 150)
    
    # Edge Detector
    kernel = np.ones((5, 5), dtype = np.uint8)
    img_dilate = cv2.dilate(img_canny, kernel, iterations = 1)
    
    contours, hier = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crop the Image
    roi = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = cv_img[y: y + h, x: x + w]
        
    # Darken Image
    roi = cv2.convertScaleAbs(roi, alpha=0.75, beta=20)
    return roi

# Pipeline for combining Color, Texture and Shape Features into one dictionary
def pipeline_all_dict(cv_img):
    final_dict = {}
    
    # Get Color Features
    color_feature_dict = pipeline_color(cv_img)
    final_dict.update(color_feature_dict)
    
    # Convert Image to Gray Image
    img_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

    # Get Texture Features
    texture_feature_dict = pipeline_texture(img_gray)
    final_dict.update(texture_feature_dict)

    # Get Shape Features
    shape_feature_dict = pipeline_shape(img_gray)
    final_dict.update(shape_feature_dict)
    
    return final_dict

# Final Pipeline
def pipeline_final(img_path):
    cv_img = pipeline_img(img_path)
    final_dict = pipeline_all_dict(cv_img)
    return final_dict

# Plot Image grid
def plot_img_grid(img_arr, nb_rows, nb_cols, figsize=(5, 5)):
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=figsize)
    
    n = 0
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].imshow(img_arr[n], interpolation = 'nearest')
            axs[i, j].axis('off')
            n += 1  

# Display 300 Images
def show_300(fruit_str, max_num = 300, default_dataset_folder_path = "Dataset"):
    folder_path = f"{default_dataset_folder_path}\\\\{fruit_str}\\\\"
    all_image_in_folder = glob(folder_path + "*.jpg")
    all_image_in_folder = all_image_in_folder[0:max_num]
    img_arr = [pipeline_img(img_path) for img_path in all_image_in_folder]
    plot_img_grid(img_arr, 15, 15, (10,10))
    
# Menu Options
def gen_dataset():
    print("")
    draw_line(screen_size)
    # Get Minimum Number of Images
    num_image_arr = []
    for folder_path in glob(f'{default_dataset_folder_path}/*/'):
        folder_name = folder_path.split("\\")[1]
        all_images = glob(folder_path + "*.jpg")
        num_image_arr.append(len(all_images))
    min_num_of_img = min(num_image_arr)

    # Generate Feature Dictionary
    feature_arr = []
    cc = ["00", "55", "AA", "FF"]

    for i in cc:
        for j in cc:
            for k in cc:
                feature_arr.append(f"#{i}{j}{k}")

    feature_arr += ["Mean", "Variance", "Entropy", "Contrast", "Homogeneity", "Correlation", "Energy"]
    feature_arr += [f"huMoment {(i + 1)}" for i in range(7)]
    feature_arr += ["Area", "Perimeter"]

    print("")
    align_left(f"Minimum Number of Images: {min_num_of_img}", 25)
    
    print("")
    align_left(f"Total Number of Images: {min_num_of_img * len(get_fruit_type())}", 25)

    print("")
    align_left(f"Total number of features: {len(feature_arr) + 1}", 25)

    print("")
    align_left("Generating New Features Dataset", 25)

    print("")
    align_left("Name of new features dataset shall be Image_Dataset_Color_Texture_Shape_Features.csv", 25)

    final_df = pd.DataFrame(columns = feature_arr)

    # Generate CSV (Color, Texture and Shape Features)
    for folder_path in glob('Dataset/*/'):
        folder_name = folder_path.split("\\")[1]
        all_images = glob(folder_path + "*.jpg")
        all_images = all_images[0:min_num_of_img]
        for img_path in all_images:
            feature_dict = pipeline_final(img_path)
            final_df = final_df.append(feature_dict, ignore_index = True)

    # Fill in NAN Values
    final_df.fillna(0, inplace=True)

    # Add Label to Each Record
    folder_name_arr = [folder_path.split("\\")[1] for folder_path in glob('Dataset/*/')]
    label_arr = list(itertools.chain.from_iterable(itertools.repeat(x, min_num_of_img) for x in folder_name_arr))
    final_df["Label"] = label_arr

    # Shuffle Dataset
    df = final_df
    df = df.sample(frac=1).reset_index(drop=True)

    # Output CSV Name
    df.to_csv("Image_Dataset_Color_Texture_Shape_Features.csv", index = False)

# Get Dataframe [Precision, Recall and F1-Score]
def get_df_type(model_name, clf_report1, clf_report2, clf_report3, col_name):
    df = pd.DataFrame()
    df[model_name[0]] = clf_report1[col_name][:-3]
    df[model_name[1]] = clf_report2[col_name][:-3]
    df[model_name[2]] = clf_report3[col_name][:-3]
    
    # Replace NA value with 0
    df.fillna(0, inplace=True)
    return df 

# Load Datasets
def load_dataset():
    df = pd.read_csv("Image_Dataset_Color_Texture_Shape_Features.csv")
    align_left("Image_Dataset_Color_Texture_Shape_Features.csv has been loaded successfully!", 20)
    return df

# Plot Confusion Matrix and Classification Report in One Graph
def plot_clf_report_conf_mat(model_title, y_test, y_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    fig.suptitle(model_title)
    
    clf_report = pd.DataFrame(classification_report(y_test, y_pred, output_dict = True)).T
    ax1.table(cellText = clf_report.values, colLabels = clf_report.columns, loc = 'center')
    ax1.axis("off")
    ax1.set_title("Classification Report")

    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index = sorted(df_Y.unique()), columns = sorted(df_Y.unique()))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='g',  cmap="YlGnBu_r", ax = ax2)
    ax2.set_title("Confusion matrix")
    return clf_report

# Plot Precision, Recall and F1-Score in one graph
def plot_prec_rec_f1_graph(model_name_arr, clf_report_1, clf_report_2, clf_report_3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
    fig.suptitle('Precision, Recall and F1-Score')
    
    term_arr = ["precision", "recall", "f1-score"]
    
    df_arr = [get_df_type(model_name, clf_report_1, clf_report_2, clf_report_3, term) for (model_name, term) in zip(model_name_arr, term_arr)]
    ax_arr = [ax1, ax2, ax3]
    
    for (tmp_df, tmp_ax, term) in zip(df_arr, ax_arr, term_arr):
        tmp_df.plot.bar(ax = tmp_ax, legend = None)
        tmp_ax.set_title(term.capitalize())

    ax1.set_ylabel("Score")
    ax2.set_xlabel("Types of Fruit")
    ax3.legend(bbox_to_anchor=(1, 1))
    
# Plot Overall Accuracy
def acc_graph(model_name_arr ,acc_score_arr, x_label, y_label, title):
    acc_dict = {
        model_name: acc_score for model_name, acc_score in zip(model_name_arr, acc_score_arr)
    }
    df = pd.Series(acc_dict)
    fig, ax = plt.subplots(figsize=(12, 2))
    for ind, val in enumerate(df):
        ax.barh(ind, val, 0.5, label=df.index[ind])
        ax.text(val,ind,f"{val:.3f}%")
    plt.yticks(np.arange(len(df)),df.index)
    plt.legend(bbox_to_anchor=(1, 1.05))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

def train_features(txt, df_X, df_Y):
    rf_model, svm_model, knn_model = "", "", ""
    
    os.system("cls")
    print("")
    draw_line(screen_size)
    print(f"Train Machine Learning Model on {txt} features".center(screen_size))
    draw_line(screen_size)
    print("")
    
    # Train Test Split based on 80-20 Ratio
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=0)
    
    # Train Random Forest Model
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train,y_train)
    y_pred = rf_model.predict(X_test)
    clf_report_1 = plot_clf_report_conf_mat("Random Forest Model", y_test, y_pred)
    
    # Train SVM Model
    svm_model = svm.SVC(decision_function_shape='ovo')
    svm_model.fit(X_train,y_train)
    y_pred = svm_model.predict(X_test)
    clf_report_2 = plot_clf_report_conf_mat("Support Vector Machine Model", y_test, y_pred)
    
    # Train KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train,y_train)
    y_pred = knn_model.predict(X_test)
    clf_report_3 = plot_clf_report_conf_mat("K-Nearest Neighbor Model", y_test, y_pred)
    
    # Get Precision, Recall, F1-Score Graph
    model_name_arr = ["Random Forest", "Support Vector Machine", "K-Nearest Neighbor"]
    plot_prec_rec_f1_graph(model_name_arr, clf_report_1, clf_report_2, clf_report_3)
    
    # Get Overall Accuracy
    acc_score_arr = [accuracy_score(y_test, model.predict(X_test)) * 100.0 for model in [rf_model, svm_model, knn_model]]
    acc_graph(model_name_arr, acc_score_arr, "Accuracy Score", "Types of Model", "Accuracy Score Comparison")
    
    accuracy_report = pd.DataFrame(columns=["Accuracy"])
    for (ind, acc_score) in enumerate(acc_score_arr):
        accuracy_report.loc[ind] = acc_score
    accuracy_report.index = model_name_arr
    
    print("")
    draw_line(screen_size)
    print("")
    align_left("Result Generated: ")
    print("")
    print(accuracy_report)
    
    return rf_model, svm_model, knn_model
    
def gen_all_feature(df):
    return df.iloc[:, 0:-1]
    
def gen_color_feature(df):
    return df.iloc[:, :64]
    
def gen_texture_feature(df):
    return df.iloc[:, 64:-10]

def gen_shape_feature(df):
    return df.iloc[:, -10:-1]

# Binarize Fruit label to Numbers
def map_fruit_val(x):
    map_fruit_val_dict = {key:val for val, key in enumerate(sorted(df_Y.unique()))}
    arr = [0 for i in range(6)]
    arr[map_fruit_val_dict[x]] = 1
    return arr

# Plot Precision-Recall Curve
def plot_prec_rec_curve(y_test, y_pred):
    
    y_test_arr = [map_fruit_val(key) for key in y_test.to_list()]
    y_test_arr = np.array(y_test_arr)
    
    y_pred_arr = [map_fruit_val(key) for key in y_pred]
    y_pred_arr = np.array(y_pred_arr)
    
    precision, recall = {}, {}
    for ind, val in enumerate(sorted(df_Y.unique())):
        precision[ind], recall[ind], _ = precision_recall_curve(y_test_arr[:, ind], y_pred_arr[:, ind])
        plt.plot(recall[ind], precision[ind], lw=2, label=f"Random Forest - {val}")
        plt.plot(np.linspace(0, 1, 20), np.linspace(1, 0, 20), 'k--')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc = "best")
    plt.title("Precision vs. Recall curve")

# Plot ROC Curve
def plot_roc_curve(y_test, y_pred):
    
    y_test_arr = [map_fruit_val(key) for key in y_test.to_list()]
    y_test_arr = np.array(y_test_arr)

    y_pred_arr = [map_fruit_val(key) for key in y_pred]
    y_pred_arr = np.array(y_pred_arr)
    
    fpr, tpr = {}, {}
    for ind, val in enumerate(sorted(df_Y.unique())):
        fpr[ind], tpr[ind], _ = roc_curve(y_test_arr[:, ind],y_pred_arr[:, ind])
        plt.plot(fpr[ind], tpr[ind], lw=2, label=f"Random Forest - {val}")
        plt.plot([0,1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="best")
    plt.title("ROC curve")
    
# Plot Feature Importance
def plot_feat_importance(model, df_X):
    feats = {key:val for (key,val) in zip(df_X.columns, rf_model.feature_importances_)}
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    importances.sort_values(by='Gini-importance', inplace = True, ascending = False)
    importances = importances.iloc[:10]
    importances.plot(kind='barh')

def menu_2_choice_5(model, df_X, df_Y):
    os.system("cls")
    print("")
    draw_line(screen_size)
    print("Precision Recall Curve".center(screen_size))
    draw_line(screen_size)
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=0)
    y_pred = model.predict(X_test)
    plot_prec_rec_curve(y_test, y_pred)
    
def menu_2_choice_6(model, df_X, df_Y):
    os.system("cls")
    print("")
    draw_line(screen_size)
    print("ROC Curve".center(screen_size))
    draw_line(screen_size)
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=0)
    y_pred = model.predict(X_test)
    plot_roc_curve(y_test, y_pred)
    
def menu_2_choice_7(model, df_X):
    os.system("cls")
    print("")
    draw_line(screen_size)
    print("Feature Importance of Random Forest Model".center(screen_size))
    draw_line(screen_size)
    plot_feat_importance(model, df_X)
    
# Save Model
def menu_2_choice_8(rf_model, svm_model, knn_model, default_model_folder_path = "Models"):
    os.system("cls")
    print("")
    draw_line(screen_size)
    print("Save Model to Folder".center(screen_size))
    draw_line(screen_size)
    model_arr = [rf_model, svm_model, knn_model]
    filename_arr = ["rf", "svm","knn"]
    for (model, filename) in zip(model_arr, filename_arr):
        pickle.dump(model, open(f"{default_model_folder_path}\\{filename}_model.sav", 'wb'))

# Load Model
def load_model(default_model_folder_path = "Models"):
    filename_arr = ["rf", "svm","knn"]
    model_arr = [pickle.load(open(f"{default_model_folder_path}\\{filename}_model.sav", 'rb')) for filename in filename_arr]
    return model_arr

# Get List of Image from To_Predict
def get_list_of_image_from_to_predict(default_prediction_folder_path = "To_Predict"):
    arr = glob(default_prediction_folder_path + "\\*.jpg")
    arr = [tmp_str.split("\\")[1][:-4] for tmp_str in arr]
    arr += ["Exit"]
    return arr

# Predict Individual Image
def pred_img(model, img_path):
    # Feature Array
    feature_arr = []
    cc = ["00", "55", "AA", "FF"]

    for i in cc:
        for j in cc:
            for k in cc:
                feature_arr.append(f"#{i}{j}{k}")

    feature_arr += ["Mean", "Variance", "Entropy", "Contrast", "Homogeneity", "Correlation", "Energy"]
    feature_arr += [f"huMoment {(i + 1)}" for i in range(7)]
    feature_arr += ["Area", "Perimeter"]
    
    # Generate DataFrame
    final_df = pd.DataFrame(columns = feature_arr)
    final_dict = pipeline_final(img_path)
    final_df = final_df.append(final_dict, ignore_index = True)
    final_df.fillna(0, inplace=True)
    final_df = final_df.iloc[:,:64]
    
    pred_label = model.predict(final_df)
    return pred_label[0]

# Use MatplotLib to Show Image
def display_img(img):
    plt.imshow (img, interpolation = 'nearest')
    _ = plt.axis(False)

# Menu
choice = 0
while choice != len(menu_options[0]):
    os.system("cls")
    print("")
    draw_line(screen_size)
    print("Fruit Image Classification".center(screen_size))
    draw_line(screen_size)
    print("")
    align_left("Menu: ")
    
    for (ind,str_2) in enumerate(menu_options[0]):
        align_left(f"{ind + 1}. {str_2}")
    
    print("")
    choice = int(input(" " * 45 + ">>  "))
    
    if choice == 1:
        choice_1 = 0
        while choice_1 != len(menu_options[1]):
            os.system("cls")
            print("")
            draw_line(screen_size)
            print(menu_options[0][0].center(screen_size))
            draw_line(screen_size)
            print("")
        
            align_left("Options: ")
            for (ind,str_2) in enumerate(menu_options[1]):
                align_left(f"{ind + 1}. {str_2}")
        
            print("")
            choice_1 = int(input(" " * 45 + ">>  "))
            
            if choice_1 == 1:
                choice_2 = 0
                fruit_str, fruit_img_path = "", ""
                option_arr = get_fruit_type()
                while choice_2 != len(option_arr):
                    os.system("cls")
                    print("")
                    draw_line(screen_size)
                    print("Please select the type of fruit".center(screen_size))
                    draw_line(screen_size)
                    print("")
                    
                    align_left("Options: ")
                    for (ind,str_2) in enumerate(option_arr):
                        align_left(f"{ind + 1}. {str_2}")

                    print("")
                    choice_2 = int(input(" " * 45 + ">>  "))
                    if choice_2 == len(option_arr):
                        break
                    
                    fruit_str = option_arr[choice_2 - 1]
                    
                    print("")
                    draw_line(screen_size)
                    print("")
                    
                    fruit_option_arr = get_fruit_img_name(fruit_str)
                    
                    fruit_img_name = ""
                    
                    while fruit_img_name not in fruit_option_arr:
                        
                        fruit_img_name = input(" " * 45 + "Please enter the image name: ")
                    
                        fruit_img_path = f"{default_dataset_folder_path}\\\\{fruit_str}\\\\{fruit_img_name}.jpg"
                        
                        if fruit_img_name not in fruit_option_arr:
                            print("")
                            align_left("Error! The fruit Image does not exist....")
                            print("")
                    
                    print("")
                    draw_line(screen_size)
                    print("")
                    align_left("Generating Pandas Series...")
                    print("")
                    print(pd.Series(pipeline_final(fruit_img_path)))
                    print("")
                    print("Generate Complete! Task Completed Successfully!")
                    print("")
                    draw_line(screen_size)
                    print("")
                    os.system("PAUSE")
                    
            elif choice_1 == 2:
                choice_2 = 0
                fruit_str = ""
                option_arr = get_fruit_type()
                while choice_2 != len(option_arr):
                    os.system("cls")
                    print("")
                    draw_line(screen_size)
                    print("Please select the type of fruit".center(screen_size))
                    draw_line(screen_size)
                    print("")
                    
                    align_left("Options: ")
                    for (ind,str_2) in enumerate(option_arr):
                        align_left(f"{ind + 1}. {str_2}")
                        
                    print("")
                    choice_2 = int(input(" " * 45 + ">>  "))
                    if choice_2 == len(option_arr):
                        break
                    
                    fruit_str = option_arr[choice_2 - 1]
                    
                    os.system("cls")
                    print("")
                    draw_line(screen_size)
                    print(f"Showing First 300 Images of {fruit_str}".center(screen_size))
                    draw_line(screen_size)
                    print("")
                    show_300(fruit_str)
                    plt.show()
                    print("")
                    align_left("Task Completed Successfully!")
                    print("")
                    draw_line(screen_size)
                    print("")
                    os.system("PAUSE")
                    
            elif choice_1 == 3:
                gen_dataset()
                
                print("")
                align_left("Generate Complete! Task Completed Successfully!", 25)
                print("")
                draw_line(screen_size)
                print("")
                os.system("PAUSE")
                
    elif choice == 2:
        choice_1 = 0
        rf_model, svm_model, knn_model = "", "", ""
        while choice_1 != len(menu_options[2]):
            os.system("cls")
            print("")
            draw_line(screen_size)
            print(menu_options[0][1].center(screen_size))
            draw_line(screen_size)
            print("")
            
            df = load_dataset()
            df_Y = df.iloc[:, -1]
            print("")
            draw_line(screen_size)
            print("")
        
            align_left("Options: ")
            for (ind,str_2) in enumerate(menu_options[2]):
                align_left(f"{ind + 1}. {str_2}")

            print("")
            choice_1 = int(input(" " * 45 + ">>  "))
            
            if choice_1 == len(menu_options[2]):
                break
            
            if choice_1 == 1:
                df_X = gen_all_feature(df)
                rf_model, svm_model, knn_model = train_features("All", df_X, df_Y)
            elif choice_1 == 2:
                df_X = gen_color_feature(df)
                rf_model, svm_model, knn_model = train_features("Color", df_X, df_Y)
            elif choice_1 == 3:
                df_X = gen_texture_feature(df)
                rf_model, svm_model, knn_model = train_features("Texture", df_X, df_Y)
            elif choice_1 == 4:
                df_X = gen_shape_feature(df)
                rf_model, svm_model, knn_model = train_features("Shape", df_X, df_Y)
            elif choice_1 == 5:
                if rf_model == "":
                    print("")
                    draw_line(screen_size)
                    print("")
                    align_left("Please train the model before selecting this option!", 25)
                    print("")
                    draw_line(screen_size)
                    print("")
                    os.system("PAUSE")
                    break
                menu_2_choice_5(rf_model, df_X, df_Y)
                
            elif choice_1 == 6:
                if rf_model == "":
                    print("")
                    draw_line(screen_size)
                    print("")
                    align_left("Please train the model before selecting this option!", 25)
                    print("")
                    draw_line(screen_size)
                    print("")
                    os.system("PAUSE")
                    break
                menu_2_choice_6(rf_model, df_X, df_Y)
            
            elif choice_1 == 7:
                if rf_model == "":
                    print("")
                    draw_line(screen_size)
                    print("")
                    align_left("Please train the model before selecting this option!", 25)
                    print("")
                    draw_line(screen_size)
                    print("")
                    os.system("PAUSE")
                    break
                menu_2_choice_7(rf_model, df_X)
                
            elif choice_1 == 8:
                if rf_model == "":
                    print("")
                    draw_line(screen_size)
                    print("")
                    align_left("Please train the model before selecting this option!", 25)
                    print("")
                    draw_line(screen_size)
                    print("")
                    os.system("PAUSE")
                    break
                menu_2_choice_8(rf_model, svm_model, knn_model)
                
            plt.show()
            print("")
            align_left("Task Completed Successfully!")
            print("")
            draw_line(screen_size)
            print("")
            os.system("PAUSE")
            
    elif choice == 3:
        choice_1 = 0
        rf_model, svm_model, knn_model = load_model()
        option_arr = get_list_of_image_from_to_predict()
        selected_img = ""
        while choice_1 != len(menu_options[3]):
            os.system("cls")
            print("")
            draw_line(screen_size)
            print(menu_options[3][0].center(screen_size))
            draw_line(screen_size)
            print("")
            
            align_left("Options: ")
            for (ind,str_2) in enumerate(option_arr):
                align_left(f"{ind + 1}. {str_2}")
            
            print("")
            choice_2 = int(input(" " * 45 + ">>  "))
            if choice_2 == len(option_arr):
                break
            
            selected_img = option_arr[choice_2 - 1]
            
            selected_img_path = f"{default_prediction_folder_path}\\{selected_img}.jpg"
            
            os.system("cls")
            print("")
            draw_line(screen_size)
            print(f"Predicting {selected_img}...".center(screen_size))
            draw_line(screen_size)
            print("")
            
            # Show Original Image
            ori_img = cv2.imread(selected_img_path)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            
            # Show Preprocessed Images
            preprocessed_img = pipeline_img(selected_img_path)
            
            # Resize Preprocessed Image
            preprocessed_img = cv2.resize(preprocessed_img, (100, 100), interpolation = cv2.INTER_AREA)
            
            img_arr = [ori_img, preprocessed_img]
            img_name_arr = ["Original Image", "Preprocessed Image"]
            
            # Show Predicted Result
            pred_label = pred_img(rf_model, selected_img_path)
            
            fig, axs = plt.subplots(1, 2, figsize=(5,5))
            fig.suptitle(f"Prediction: {pred_label}")
            
            for i in range(0, 2):
                axs[i].imshow(img_arr[i], interpolation = 'nearest')
                axs[i].axis('off')
                axs[i].set_title(img_name_arr[i])
            
            
            align_left(f"Prediction: {pred_label}")
            
            plt.show()
            print("")
            align_left("Task Completed Successfully!")
            print("")
            draw_line(screen_size)
            print("")
            os.system("PAUSE")
        
# Exit Menu
os.system("cls")
print("")
draw_line(screen_size)
print("Thank you for using our program".center(screen_size))
draw_line(screen_size)
print("")
os.system("PAUSE")


