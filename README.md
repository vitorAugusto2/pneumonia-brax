# Modelo híbrido baseado em CNN e classificadores para detecção de pneumonia em imagens de raio-x

A detecção de pneumonia em imagens de raio X de tórax foi relevante para apoiar o
diagnóstico clínico e reduzir riscos associados ao atraso na identificação da doença. Este
estudo teve como objetivo desenvolver e avaliar um modelo híbrido para a detecção de
pneumonia em radiografias torácicas. Para isso, a metodologia utilizou o conjunto de dados
BRAX, após etapas de seleção, balanceamento, particionamento em treino, validação e
teste, além da aplicação de técnicas de aumento de dados. As arquiteturas VGG16 e
ResNet50 pré-treinadas no “ImageNet” atuaram como extratoras de características com
ajuste fino parcial e foram combinadas aos classificadores “Random Forest”, XGBoost e
“Support Vector Machine”. O estudo também realizou busca de hiperparâmetros com
“GridSearchCV” para otimizar o desempenho dos modelos híbridos. Os resultados indicaram
desempenho consistente entre as combinações avaliadas, com destaque para o modelo
híbrido VGG16 + XGBoost.

Palavras-chave: detecção de pneumonia; aprendizagem por transferência; extração de
características; classificadores; inteligência artificial.

### **Metodologia**

<img width="800" height="600" alt="TCC USP - Metodologia" src="https://github.com/user-attachments/assets/f62dfc42-c0f0-4783-a4ab-505a46e1222b" />

### **Resultados**

| CNN      | Classificador | Conjunto | Acurácia | Precisão | Revocação | F1-score | AUC   |
| -------- | ------------- | -------- | -------- | -------- | --------- | -------- | ----- |
| VGG16    | RF            | Val      | 0.659    | 0.565    | 0.840     | 0.675    | 0.760 |
| VGG16    | RF            | Test     | 0.683    | 0.628    | 0.887     | 0.736    | 0.782 |
| VGG16    | XGB           | Val      | 0.669    | 0.569    | 0.897     | 0.697    | 0.755 |
| VGG16    | XGB           | Test     | 0.720    | 0.650    | 0.950     | 0.772    | 0.771 |
| VGG16    | SVM           | Val      | 0.683    | 0.591    | 0.814     | 0.685    | 0.758 |
| VGG16    | SVM           | Test     | 0.708    | 0.663    | 0.838     | 0.740    | 0.771 |
| ResNet50 | RF            | Val      | 0.672    | 0.570    | 0.917     | 0.703    | 0.776 |
| ResNet50 | RF            | Test     | 0.677    | 0.623    | 0.887     | 0.732    | 0.769 |
| ResNet50 | XGB           | Val      | 0.661    | 0.559    | 0.942     | 0.702    | 0.779 |
| ResNet50 | XGB           | Test     | 0.658    | 0.605    | 0.900     | 0.724    | 0.761 |
| ResNet50 | SVM           | Val      | 0.696    | 0.596    | 0.872     | 0.708    | 0.783 |
| ResNet50 | SVM           | Test     | 0.702    | 0.660    | 0.825     | 0.733    | 0.754 |
