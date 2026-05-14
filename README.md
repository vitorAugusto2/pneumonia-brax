
# Modelo híbrido baseado em CNN e classificadores para detecção de pneumonia em imagens de raio-x
## Metodologia

![TCC USP (4)](https://github.com/user-attachments/assets/d35c4845-a555-41f4-b83a-c1473f317178)

## Resultados preliminares
**Distribuição de pacientes**

| Descrição      | Quantidade |
| -------------- | ---------- |
| Pneumonia      | 380        |
| Normal         | 13.907     |

**Divisão**
| Conjunto | Target=0 | Target=1 | 
| -------- | -------- | -------- |
| Train    | 323      | 266      |
| Val      | 93       | 76       |
| Test     | 43       | 38       |

### **Resultados**

| CNN      | Classificador | Conjunto | Acurácia  | Precisão  | Revocação | F1-score  | AUC       | Tempo Execução |
| -------- | ------------- | -------- | --------- | --------- | --------- | --------- | --------- | -------------- |
| VGG16    | RF            | Val      | 0.691     | 0.710     | 0.455     | 0.555     | 0.733     | 00:43          |
| VGG16    | RF            | Test     | 0.627     | 0.667     | 0.500     | 0.571     | 0.721     | 00:43          |
| VGG16    | XGB           | Val      | 0.672     | 0.634     | 0.532     | 0.578     | 0.730     | 01:49          |
| VGG16    | XGB           | Test     | 0.615     | 0.625     | 0.562     | 0.592     | 0.675     | 01:49          |
| VGG16    | SVM           | Val      | 0.626     | 0.550     | 0.635     | 0.589     | 0.670     | 00:55          |
| VGG16    | SVM           | Test     | 0.615     | 0.594     | 0.713     | 0.648     | 0.669     | 00:55          |
| ResNet50 | RF            | Val      | 0.694     | 0.681     | 0.519     | 0.589     | 0.738     | 00:42          |
| ResNet50 | RF            | Test     | 0.696     | 0.731     | 0.613     | 0.667     | 0.739     | 00:42          |
| ResNet50 | XGB           | Val      | 0.696     | 0.639     | 0.647     | 0.643     | 0.762     | 01:41          |
| ResNet50 | XGB           | Test     | 0.689     | 0.674     | **0.725** | 0.699     | 0.747     | 01:41          |
| ResNet50 | SVM           | Val      | 0.659     | 0.600     | 0.577     | 0.588     | 0.704     | 00:45          |
| ResNet50 | SVM           | Test     | 0.683     | 0.671     | 0.713     | 0.691     | 0.725     | 00:45          |

