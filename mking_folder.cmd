@echo off
chcp 65001 >nul
REM =========================================
REM 챕터 목록 기반으로 2자리 번호 붙인 새 폴더 생성 + 하위 README 생성
REM =========================================
setlocal EnableExtensions EnableDelayedExpansion

set "BASE_PATH=C:\Users\Owner\Documents\ml-dl-hands-on"

set /a COUNT=1
for %%F in (
"Logistic_Regression_for_Classification_Tasks"
"Decision_Trees_for_Predictive_Modeling"
"Ensemble_Learning_with_Random_Forests_and_Gradient_Boosting"
"Dimensionality_Reduction_with_PCA_and_t-SNE"
"Unsupervised_Learning_with_Clustering_Algorithms"
"Introduction_to_Artificial_Neural_Networks"
"Deep_Neural_Network_Optimization_and_Regularization"
"Custom_Model_Building_with_Keras_Functional_API"
"Convolutional_Neural_Networks_for_Image_Recognition"
"Sequence_Modeling_with_RNN_LSTM_and_GRU"
"Natural_Language_Processing_with_RNN_and_Word_Embeddings"
"Reinforcement_Learning_with_Deep_Q-Networks"
"Autoencoders_for_Data_Compression_and_Anomaly_Detection"
"Generative_Models_with_GANs"
"Deploying_Machine_Learning_Models_to_Production"
) do (
    set "FOLDER=%%~F"
    set "NUM=0!COUNT!"
    set "NUM=!NUM:~-2!"
    set "NEW=!NUM!-!FOLDER!"

    REM 메인 프로젝트 폴더 생성
    if not exist "%BASE_PATH%\!NEW!" (
        echo [CREATE] !NEW!
        mkdir "%BASE_PATH%\!NEW!"
    ) else (
        echo [SKIP] 이미 존재: !NEW!
    )

    REM 하위 폴더 및 README 생성
    for %%S in (notebooks src data results) do (
        mkdir "%BASE_PATH%\!NEW!\%%S" 2>nul
        if not exist "%BASE_PATH%\!NEW!\%%S\README.md" (
            > "%BASE_PATH%\!NEW!\%%S\README.md" (
                echo # %%S folder in !NEW!
                echo This folder contains files related to %%S.
            )
        )
    )

    if not exist "%BASE_PATH%\!NEW!\README.md" (
        > "%BASE_PATH%\!NEW!\README.md" (
            echo # !NEW!
            echo Project description goes here.
        )
    )

    set /a COUNT+=1
)

echo(
echo 작업 완료!
pause
