# Mini Project
UCCC 2513 Mini Project\
For this semester, our mini project was on introduction of Image Processing. For our project, we have proposed a novel fruit classification machine learning program, that is able to identify fruits based on color features. The novelty of the program is that it can be trained on low computation time and simple digital camera captured images. The fruits can be classified under any angles and any environmental conditions.

## Teammates
1. Foong Xi Wan (Leader) 
2. Muck Wein Yong
3. Tan Xi En
4. Yeong Su Yen

## Project Objectives
1. To develop a fruit classification program that can classify fruits images
   * The system must be able to identify and classify multiple types of fruits
   * The system must be able to classify fruits under different angles and multiple environmental conditions
   * The system has to be able to make predictions using real time input images
2. To Identify the important features that helps in classifying fruit images
   * The system has to be able to provide analysis and visualizations on important features. The types of features used in this project are colour, shape and texture features.
   * The system must be able to make comparisons on different types of features.

## Dataset Used
Fruits 360 dataset: A dataset of images containing fruits and vegetables ([Link](https://www.kaggle.com/moltean/fruits))

## Google Drive Link
[Google Drive Repository](https://drive.google.com/drive/folders/1nwDUyBuLq8jlWzUven5MUV2RDDWYCW4N?usp=sharing)

## Github Repo (For This Assignment):
[Github Repo](https://github.com/Neix20/MiniProject)

## Presentation Video Link
[Presentation Video](https://drive.google.com/file/d/1nuWo8pfAkyzwoR7Bqsmfz51vcURxgTba/view?usp=sharing)

## Demo Video Link
[Demo Video](https://drive.google.com/file/d/1dWEvkKYy-jKbiF_8hRZXsSdsYrCOKt-v/view?usp=sharing)

## Libraries Used
opencv_python 4.5.2.54\
pandas 1.2.1\
numpy 1.19.5\
matplotlib 3.4.2\
seaborn 0.11.1\
scikit_image 0.18.2\
scikit_learn 0.24.2\
jupyter notebook

## How to Use:
1. There will be twelve files and folders in this project. That is:
    * Dataset Folder
    * Models Folder
    * To_Predict Folder
    * requirements.txt
    * Generate_Dataset.ipynb
    * Model_Result_Analysis (All Features).ipynb
    * Model_Result_Analysis (Color Features).ipynb
    * Model_Result_Analysis (Texture Features).ipynb
    * Model_Result_Analysis (Shape Features).ipynb
    * Demo_Make_Predictions.ipynb
    * install_necessary_libraries.py
    * Fruit_Image_Classifier_Program.py
2. Before you run this program, please use anaconda or ensure that you can run python in path. Please load the directory in cmd or anaconda prompt to the same directory as the folder.
3. To run this program, start by installing the necessary libraries listed above. You can install all the libraries by running the following command ```python install_necessary_libraries.py```
4. To run the cli program, please run the following command ```python Fruit_Image_Classifier_Program.py```. It shall list out three options, that is, generate dataset, model result analysis and predicting images.
5. To generate the dataset, please download the fruits-360 dataset from the kaggle link provided above. Please put the following fruits - Apple Red 1, Grapes Blue, Lemon, Lychee, Limes and Pear from the Training Folder into the Dataset Folder. Please Rename Apple Red 1 as Apple and Grapes Blue as Grapes. Then, please geenrate dataset using the cli program or Generate_Dataset.ipynb. It will generate a new csv file named Image_Dataset_Color_Texture_Shape_Features.csv
7. If you would like to train and analyze the models, please run model result analysis using the cli program or the following 4 ipynb files - Model_Result_Analysis (All Features).ipynb, Model_Result_Analysis (Color Features).ipynb, Model_Result_Analysis (Texture Features).ipynb, Model_Result_Analysis (Shape Features).ipynb respectively. After you have run the cli program, please make sure to **run the 8th option: save the model**. If you do not run this option, Models Folder will be empty and this will result in the program unable to load the random forest model.
8. If you would like to make predictions, please put the image you would like predicted into the To_Predict Folder. Please then use the cli program 3rd option or run Demo_Make_Predictions.ipynb
9. If there are any issues, please follow the demo video. A detailed step-by-step guide is listed and shown in the demo video.
10. If there are any queries, feel free to contact me Tan Xi En over MSTeams.
