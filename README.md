## Getting Started

Disclaimer: For better and more consistent performance, an NVIDIA GPU is recommended while also installing latest Cuda Toolkit.
Will run on device CPU otherwise.
This will take some time to get started at first.

## Run the development server:

```
powershell 1 (Backend)

1.
cd .\py-backend\

2.
py -m venv .venv

3.
.\.venv\Scripts\activate

4.
pip install -r requirements.txt

4.1(Only if host device has Cuda Toolkit installed & has eligible NVIDIA GPU; If not then skip this)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

5.
.\run.bat
```

## Run the Frontend server:

```
powershell 2 (Frontend)

npm install

npm run dev
```

## Once everything is set up

Open [http://localhost:3000](http://localhost:3000) with your browser to run Scout AI.

# R code used to build Statistical Model

```
install.packages("randomForest")
install.packages("pROC")
install.packages("caret")

library(MASS)
library(caret)
library(randomForest)
library(klaR)

library(pROC)


data <- all_data_with_ada_embeddings_will_be_splitted_into_train_test_set

# Get the total number of columns
n_cols <- ncol(data)

# Select 4th column and last 6 columns
data <- data[, c(4, (n_cols-5):n_cols)]

set.seed(123)
data <- data[sample(nrow(data)), ]
data$LLM <- ifelse(data$LLM != "Human", "AI", data$LLM)
attach(data)
newdat <- data.frame(
  LLM = data$LLM,
  label = data$label,
  L = data$code_lines,
  CM = data$comments,
  CD = data$code_lines,
  FN =  data$functions,
  B = data$blank_lines,
  CD.L = data$code_lines / data$lines,
  CM.L= data$comments / data$lines,
  F.L = data$functions / data$lines,
  B.L = data$blank_lines / data$lines,
  CM.CD = data$comments / data$code_lines
)

train_index <- createDataPartition(newdat$LLM, p = 0.80, list = FALSE)
temp_train <- newdat[train_index, ]
TestData <- newdat[-train_index, ]
min_class_size <- min(nrow(ai_data), nrow(human_data))
ai_data <- temp_train[temp_train$LLM == "AI", ]
human_data <- temp_train[temp_train$LLM == "Human", ]

ai_balanced <- ai_data[sample(nrow(ai_data), min_class_size), ]
human_balanced <- human_data[sample(nrow(human_data), min_class_size), ]



TrainData <- rbind(ai_balanced, human_balanced)
TrainData <- TrainData[sample(nrow(TrainData)), ]

#first and last 190 rows for 5% split to test
TestData <- rbind(data[1:190, ], data[(nrow(data)-189):nrow(data), ])

TestData <- data.frame(
  LLM = TestData$LLM,
  label = TestData$label,
  L = TestData$code_lines,
  CM = TestData$comments,
  CD = TestData$code_lines,
  FN =  TestData$functions,
  B = TestData$blank_lines,
  CD.L = TestData$code_lines / TestData$lines,
  CM.L= TestData$comments / TestData$lines,
  F.L = TestData$functions / TestData$lines,
  B.L = TestData$blank_lines / TestData$lines,
  CM.CD = TestData$comments / TestData$code_lines
)


#rest of the dataset for training
TrainData <- data[191:(nrow(data)-190), ]


TrainData <- data.frame(
  LLM = TrainData$LLM,
  label = TrainData$label,
  L = TrainData$code_lines,
  CM = TrainData$comments,
  CD = TrainData$code_lines,
  FN =  TrainData$functions,
  B = TrainData$blank_lines,
  CD.L = TrainData$code_lines / TrainData$lines,
  CM.L= TrainData$comments / TrainData$lines,
  F.L = TrainData$functions / TrainData$lines,
  B.L = TrainData$blank_lines / TrainData$lines,
  CM.CD = TrainData$comments / TrainData$code_lines
)

CD.L= (code_lines/lines)
CM.L = (comments/lines)
F.L = (functions/lines)
B.L = (blank_lines/lines)
L = data$code_lines
CM = data$comments
CD = data$code_lines
FN =  data$functions
B = data$blank_lines
CM.CD = data$comments / data$code_lines


formulaAll=( LLM ~ CM.CD+ CD.L+CM.L+ F.L + B.L + L + CM + CD + FN + B)
greedy.wilks(formulaAll,data=newdat, niveau = 0.05) 
formulaStepwise=(LLM ~ F.L + FN + CM.CD + CM.L + B)

fit <- lda(formulaStepwise,data=TrainData, method="moment")

plot(fit)


glm_model <- glm(label ~ F.L + FN+ CM.CD+ CM.L +B,data = TrainData, family = binomial)
summary(glm_model)
coef(glm_model)
pred_probs <- predict(glm_model, newdat = TrainData, type = "response")
roc_curve <- roc(TrainData$label, pred_probs)
auc(roc_curve)
best_thresh <- coords(roc_curve, "best", ret="threshold", best.method="youden")
best_thresh
pred_class <- ifelse(pred_probs >= 0.397477, "AI", "Human")
table(Predicted = pred_class, Actual = TrainData$label)
plot(roc_curve, print.thres="best", print.thres.best.method="youden", col="red", lwd=2)
```

# HuggingFace Resources

```
https://huggingface.co/desklib/ai-text-detector-v1.01
https://huggingface.co/haywoodsloan/ai-image-detector-deploy
https://huggingface.co/datasets/basakdemirok/AIGCodeSet
```
