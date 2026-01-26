
# Modelo híbrido baseado em CNN e classificadores para detecção de pneumonia em imagens de raio-x
## Resultados preliminares
**Distribuição de pacientes**

| Descrição      | Quantidade |
| -------------- | ---------- |
| Pneumonia      | 380        |
| Normal         | 18.062     |

**Divisão**
| Conjunto | Target=0 | Target=1 | 
| -------- | -------- | -------- |
| Train    | 323      | 266      |
| Val      | 93       | 76       |
| Test     | 43       | 38       |

### **Resultados**

| CNN      | Classificador | Conjunto | Acurácia  | Precisão  | Revocação | F1-score  | AUC       | Tempo Execução |
| -------- | ------------- | -------- | --------- | --------- | --------- | --------- | --------- | -------------- |
| VGG16    | RF            | VAL      | 0.691     | 0.710     | 0.455     | 0.555     | 0.733     | 00:43          |
| VGG16    | RF            | TEST     | 0.627     | 0.667     | 0.500     | 0.571     | 0.721     | 00:43          |
| VGG16    | XGB           | VAL      | 0.672     | 0.634     | 0.532     | 0.578     | 0.730     | 01:49          |
| VGG16    | XGB           | TEST     | 0.615     | 0.625     | 0.562     | 0.592     | 0.675     | 01:49          |
| VGG16    | SVM           | VAL      | 0.626     | 0.550     | 0.635     | 0.589     | 0.670     | 00:55          |
| VGG16    | SVM           | TEST     | 0.615     | 0.594     | 0.713     | 0.648     | 0.669     | 00:55          |
| ResNet50 | RF            | VAL      | 0.694     | 0.681     | 0.519     | 0.589     | 0.738     | 00:42          |
| ResNet50 | RF            | TEST     | 0.696     | 0.731     | 0.613     | 0.667     | 0.739     | 00:42          |
| ResNet50 | XGB           | VAL      | 0.696     | 0.639     | 0.647     | 0.643     | 0.762     | 01:41          |
| ResNet50 | XGB           | TEST     | 0.689     | 0.674     | **0.725** | 0.699     | 0.747     | 01:41          |
| ResNet50 | SVM           | VAL      | 0.659     | 0.600     | 0.577     | 0.588     | 0.704     | 00:45          |
| ResNet50 | SVM           | TEST     | 0.683     | 0.671     | 0.713     | 0.691     | 0.725     | 00:45          |

