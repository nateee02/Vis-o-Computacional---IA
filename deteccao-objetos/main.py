import cv2
import numpy as np

TINY = False

ARQUIVO_CFG = "deteccao-objetos/yolov3{}.cfg".format("-tiny" if TINY else "")
ARQUIVO_PESOS = "deteccao-objetos/yolov3{}.weights".format("-tiny" if TINY else "")
ARQUIVO_CLASSES = "deteccao-objetos/coco{}.names".format("-tiny" if TINY else "")

with open(ARQUIVO_CLASSES, "r") as arquivo:
    CLASSES = [linha.strip() for linha in arquivo.readlines()]

CORES = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def carregar_modelo_pretreinado():
    modelo = cv2.dnn.readNetFromDarknet(ARQUIVO_CFG, ARQUIVO_PESOS)
    modelo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    modelo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    if modelo.empty():
        raise IOError("Não foi possível carregar o modelo de detecção de objetos.")
    return modelo

def preprocessar_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    return blob

def detectar_objetos(frame, modelo):
    blob = preprocessar_frame(frame)
    modelo.setInput(blob)
    nomes_camadas = modelo.getLayerNames()
    camadas_saida = [nomes_camadas[i - 1] for i in modelo.getUnconnectedOutLayers()]
    saidas = modelo.forward(camadas_saida)
    return saidas

def cor_predominante(frame, x, y, largura, altura):
    """Identifica a cor predominante em uma região do frame."""
    roi = frame[y:y+altura, x:x+largura]
    if roi.size == 0:
        return "indefinida"

    # Converter a ROI para HSV
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calcular a média de cada canal HSV na ROI
    media_h = int(np.mean(hsv_roi[:, :, 0]))  # Hue
    media_s = int(np.mean(hsv_roi[:, :, 1]))  # Saturação
    media_v = int(np.mean(hsv_roi[:, :, 2]))  # Brilho

    # Definir cores com base no valor médio de Hue
    if media_s < 40:  # Saturação muito baixa
        if media_v > 200:
            return "branco"
        elif media_v < 50:
            return "preto"
        else:
            return "cinza"

    if 0 <= media_h < 15 or 165 <= media_h <= 180:
        return "vermelho"
    elif 15 <= media_h < 25:
        return "laranja"
    elif 25 <= media_h < 35:
        return "amarelo"
    elif 35 <= media_h < 85:
        return "verde"
    elif 85 <= media_h < 105:
        return "ciano"
    elif 105 <= media_h < 135:
        return "azul"
    elif 135 <= media_h < 165:
        return "roxo"
    else:
        return "indefinida"
    
    

def desenhar_deteccoes(frame, deteccoes, limiar=0.5):
    """Desenha caixas e identifica cores apenas para objetos classificados como 'copo'."""
    (altura, largura) = frame.shape[:2]
    caixas = []
    confiancas = []
    ids_classes = []

    for saida in deteccoes:
        for deteccao in saida:
            pontuacoes = deteccao[5:]
            id_classe = np.argmax(pontuacoes)
            confianca = pontuacoes[id_classe]

            # Verifica se o objeto é um copo e atende ao limite de confiança
            if confianca > limiar and CLASSES[id_classe] == "copo":  # Verifica se a classe é copo
                caixa = deteccao[0:4] * np.array([largura, altura, largura, altura])
                (centroX, centroY, largura_caixa, altura_caixa) = caixa.astype("int")
                x = int(centroX - (largura_caixa / 2))
                y = int(centroY - (altura_caixa / 2))

                caixas.append([x, y, int(largura_caixa), int(altura_caixa)])
                confiancas.append(float(confianca))
                ids_classes.append(id_classe)

    indices = cv2.dnn.NMSBoxes(caixas, confiancas, limiar, limiar - 0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (caixas[i][0], caixas[i][1])
            (largura_caixa, altura_caixa) = (caixas[i][2], caixas[i][3])

            # Determinar a cor predominante na ROI
            cor = cor_predominante(frame, x, y, largura_caixa, altura_caixa)

            # Desenhar a caixa ao redor do copo
            cv2.rectangle(frame, (x, y), (x + largura_caixa, y + altura_caixa), (0, 255, 0), 2)  # Verde
            texto = f"Copo: {cor}"
            cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Verde

def main():
    print("Inicializando o detector de objetos...")
    modelo = carregar_modelo_pretreinado()
    captura_video = cv2.VideoCapture(0) # 0 para webcam padrão

    if not captura_video.isOpened():
        raise Exception("Não foi possível abrir a webcam.")

    captura_video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    captura_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    limiar_confianca = 0.5 

    def ajustar_limiar(valor):
        nonlocal limiar_confianca
        limiar_confianca = valor / 100

    cv2.namedWindow('Detecta Objetos')
    if TINY:
        cv2.createTrackbar('Limiar de Confiança', 'Detecta Objetos', int(limiar_confianca * 100), 100, ajustar_limiar)
    try:
        while True:
            ret, frame = captura_video.read()
            if not ret:
                break
            
            deteccoes = detectar_objetos(frame, modelo)
            desenhar_deteccoes(frame, deteccoes, limiar_confianca)

            cv2.imshow('Detecta Objetos', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        captura_video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
