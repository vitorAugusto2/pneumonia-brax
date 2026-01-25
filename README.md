
### **Distribuição de pacientes**
| Categoria | Descrição                    | Quantidade |
| --------- | ---------------------------- | ---------- |
| Dataset   | Pacientes com pneumonia      | 380        |
| Dataset   | Outras doenças / Sem achados | 18.062     |

**Divisão estratificada**
| Conjunto | Target=0 | Target=1 | 
| -------- | -------- | -------- |
| Train    | 373      | 266      |
| Val      | 104      | 76       |
| Test     | 45       | 38       |

### **Resultados preliminares**

| CNN | Classificador | Conjunto | Accuracy | Precision | Recall | F1-score | AUC   | Tempo Total |
| -------- | ------------- | ----- | -------- | --------- | ------ | -------- | ----- | ----------- |
| VGG16    | RF            | Val   | 0.669    | 0.600     | 0.212  | 0.313    | 0.658 | 00:45       |
| VGG16    | RF            | Test  | 0.655    | 0.667     | 0.300  | 0.414    | 0.763 | 00:45       |
| VGG16    | XGB           | Val   | 0.655    | 0.521     | 0.404  | 0.455    | 0.642 | 02:05       |
| VGG16    | XGB           | Test  | 0.695    | 0.652     | 0.537  | 0.589    | 0.757 | 02:05       |
| VGG16    | SVM           | Val   | 0.685    | 0.562     | 0.526  | 0.543    | 0.658 | 01:04       |
| VGG16    | SVM           | Test  | 0.670    | 0.593     | 0.600  | 0.596    | 0.741 | 01:04       |
| ResNet50 | RF            | Val   | 0.689    | 0.656     | 0.269  | 0.382    | 0.704 | 00:47       |
| ResNet50 | RF            | Test  | 0.695    | 0.750     | 0.375  | 0.500    | 0.742 | 00:47       |
| ResNet50 | XGB           | Val   | 0.678    | 0.568     | 0.404  | 0.472    | 0.722 | 02:02       |
| ResNet50 | XGB           | Test  | 0.690    | 0.656     | 0.500  | 0.567    | 0.757 | 02:02       |
| ResNet50 | SVM           | Val   | 0.626    | 0.476     | 0.506  | 0.491    | 0.657 | 00:52       |
| ResNet50 | SVM           | Test  | 0.665    | 0.583     | 0.613  | 0.598    | 0.721 | 00:52       |
