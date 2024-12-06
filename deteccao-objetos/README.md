### YOLOv3 com OpenCV em Python

Este repositório contém um exemplo de uso do modelo YOLOv3 para detecção de objetos em tempo real utilizando OpenCV e Python. O código inclui o carregamento do modelo pré-treinado YOLOv3, a configuração do OpenCV DNN, e a realização da detecção de objetos em tempo real a partir de uma captura de vídeo (por exemplo, webcam).

### Links para Download de Modelos YOLO

#### YOLOv3:

- Arquivo de configuração: [yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- Arquivo de pesos: [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- Arquivo de nomes das classes: [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

#### YOLOv3-tiny:

- Arquivo de configuração: [yolov3-tiny.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
- Arquivo de pesos: [yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)
- Arquivo de nomes das classes: [coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

### Descrição do Código

Este código faz o seguinte:

1. **Carrega o modelo YOLOv3 pré-treinado**:
   Utiliza a função `cv2.dnn.readNetFromDarknet` para carregar a configuração (`.cfg`) e os pesos (`.weights`) do modelo YOLOv3.

2. **Pré-processa os frames de vídeo**:
   Redimensiona e normaliza os frames capturados para que possam ser usados como entrada para o modelo YOLOv3.

3. **Detecta objetos nos frames**:
   Utiliza o modelo carregado para detectar objetos nos frames pré-processados.

4. **Desenha as detecções nos frames**:
   Desenha caixas delimitadoras ao redor dos objetos detectados, juntamente com a classe e a confiança da detecção.

### O que foi adaptado

Usando o arquivo disponibilizado em aula detecção_objetos, para execução da ideia deste projeto foram implementadas às funções cor_predominante e desenhar_deteccoes.


### Função cor_predominante
Analisa a cor predominante em uma região específica de uma imagem (chamada de ROI - Região de Interesse).

**Extrai a região de interesse (ROI):** Usa as coordenadas (x, y, largura, altura) para selecionar a área da imagem onde a cor será analisada.

**Verifica se a ROI é válida:** Retorna "indefinida" se a ROI não contém pixels.

**Converte para espaço de cor HSV:** Transforma a região de cor BGR (usada pelo OpenCV) em HSV, que facilita a análise de tonalidades (Hue), saturação e brilho.

**Calcula a média dos canais HSV:** Obtém os valores médios de tonalidade (Hue), saturação (Saturation) e brilho (Value) na região.

**Classifica a cor predominante:** Se a saturação for baixa, classifica como branco, preto ou cinza dependendo do brilho. Caso contrário, usa o valor de Hue para identificar cores como vermelho, azul, amarelo, etc.

**Retorna a cor identificada:** Baseada nos valores médios de HSV.


### Função desenhar_deteccoes

Esta função detecta objetos do tipo "copo" em um frame e desenha uma caixa ao redor deles, indicando a cor predominante.

**Inicializa listas para dados de detecções:** Prepara estruturas para armazenar caixas delimitadoras, níveis de confiança e IDs de classes detectadas.

**Percorre as detecções:** Obtém a classe mais provável de cada objeto e sua confiança. Filtra para considerar apenas objetos classificados como "copo" e que superam o limiar de confiança especificado.

**Calcula as caixas delimitadoras:** Converte as coordenadas normalizadas do modelo YOLO para pixels, gerando as dimensões (x, y, largura, altura) de cada copo.

**Aplica supressão de não-máximos:** Remove sobreposições excessivas entre caixas para manter apenas as melhores detecções.

**Determina a cor predominante:** Para cada copo detectado, usa a função cor_predominante para identificar a cor da região correspondente.

### COCO

Common Objects in Context: https://cocodataset.org/#overview

### DNN (Deep Neural Network) no OpenCV

A DNN (Deep Neural Network) é uma biblioteca no OpenCV que permite o uso de redes neurais profundas para diversas tarefas de visão computacional, como classificação de imagens, detecção de objetos, e segmentação semântica. A API DNN do OpenCV oferece uma interface para carregar e executar modelos treinados em diferentes frameworks de deep learning, como Caffe, TensorFlow, PyTorch e Darknet. Ela suporta tanto a CPU quanto a GPU, tornando-a flexível e eficiente para aplicações em tempo real.

### Darknet

Darknet é uma estrutura de rede neural de código aberto escrita em C e CUDA. Ela é utilizada principalmente para a detecção de objetos em tempo real. YOLO (You Only Look Once) é uma das implementações mais conhecidas que utilizam Darknet. YOLO é altamente eficiente e capaz de detectar objetos em uma única passagem pela rede neural, ao contrário de métodos tradicionais que requerem múltiplas passagens. Darknet é conhecido por seu desempenho rápido e capacidade de ser executado em tempo real em GPUs.
