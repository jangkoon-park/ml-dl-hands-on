@echo off
REM =========================================
REM 각 프로젝트 폴더 및 하위 폴더에 README.md 생성
REM =========================================

set "BASE_PATH=C:\Users\Owner\Documents\ml-dl-hands-on"

setlocal enabledelayedexpansion
for %%F in (
"Unsupervised_Learning_with_Clustering_Algorithms"
"Autoencoders_for_Data_Compression_and_Anomaly_Detection"
"Convolutional_Neural_Networks_for_Image_Recognition"
"Custom_Model_Building_with_Keras_Functional_API"
"Decision_Trees_for_Predictive_Modeling"
"Deep_Neural_Network_Optimization_and_Regularization"
"Deploying_Machine_Learning_Models_to_Production"
"Dimensionality_Reduction_with_PCA_and_t-SNE"
"Ensemble_Learning_with_Random_Forests_and_Gradient_Boosting"
"Generative_Models_with_GANs"
"Image Digit Classification Pipeline"
"Introduction_to_Artificial_Neural_Networks"
"Logistic_Regression_for_Classification_Tasks"
"Natural_Language_Processing_with_RNN_and_Word_Embeddings"
"Predictive Modeling with Linear, Polynomial, and Regularized Regression"
"Real Estate Price Prediction Pipeline"
"Reinforcement_Learning_with_Deep_Q-Networks"
"Sequence_Modeling_with_RNN_LSTM_and_GRU"
) do (
    set "FOLDER=%%~F"
    if exist "%BASE_PATH%\!FOLDER!" (
        echo [!FOLDER!] 폴더 및 하위 README 생성 중...
        
        for %%S in (notebooks src data results) do (
            mkdir "%BASE_PATH%\!FOLDER!\%%S" 2>nul
            if not exist "%BASE_PATH%\!FOLDER!\%%S\README.md" (
                echo # %%S folder in !FOLDER! > "%BASE_PATH%\!FOLDER!\%%S\README.md"
                echo This folder contains files related to %%S. >> "%BASE_PATH%\!FOLDER!\%%S\README.md"
            )
        )

        if not exist "%BASE_PATH%\!FOLDER!\README.md" (
            echo # !FOLDER! > "%BASE_PATH%\!FOLDER!\README.md"
            echo Project description goes here. >> "%BASE_PATH%\!FOLDER!\README.md"
        )
    ) else (
        echo [WARN] 폴더 없음: !FOLDER!
    )
)

echo.
echo 작업 완료!
pause
