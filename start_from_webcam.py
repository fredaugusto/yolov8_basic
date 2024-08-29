import cv2
from ultralytics import YOLO

# Carrega o modelo YOLOv8m a partir do arquivo 'yolov8m.pt'. Se não houver na raiz da pasta, será automaticamente baixado.
model = YOLO('yolov8m.pt')

# Configura a confiança mínima para considerar uma detecção válida
confidence_threshold = 0.4  # Quanto menor, mais detecta mas com menos confiabilidade

# Inicia a captura de vídeo da webcam (índice 0 geralmente se refere à webcam padrão)
cap = cv2.VideoCapture(0)

while True:
    # Captura um frame da webcam
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar a imagem.")  # Mensagem de erro se não for possível capturar o frame
        break

    # Envia o frame para o modelo YOLO para realizar a detecção de objetos
    results = model(frame, conf=confidence_threshold)

    # Itera sobre os resultados das detecções
    for result in results:
        # Obtém os nomes das classes detectadas dentro do dicionário
        class_names = result.names

        # Itera sobre cada detecção no resultado
        for detection in result.boxes:

            # Verifica se a confiança da detecção é maior ou igual ao limite definido
            if detection.conf[0] >= confidence_threshold:

                # Extrai as coordenadas da caixa delimitadora da detecção
                x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Coordenadas a partir do canto superior esquerdo (x1, y1) e inferior direito (x2, y2)
                confidence = detection.conf[0]  # Captura a confiança da detecção
                class_id = int(detection.cls[0])  # ID da classe detectada

                # Obtém o nome da classe com base no ID da classe
                class_name = class_names[class_id]
                
                # Define a cor para desenhar a caixa delimitadora
                myColor = (255, 0, 0)

                # Desenha a caixa delimitadora ao redor do objeto detectado
                cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 2)

                # Insere o rótulo com o nome da classe e a confiança
                label = f'{class_name} ({confidence:.2f})'

                # Adiciona o rótulo ao frame próximo ao topo da caixa delimitadora
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, myColor, 2)

    # Exibe o frame com as caixas delimitadoras e rótulos no frame do vído
    cv2.imshow('YOLOv8m Object Detection', frame)

    # Fecha a janela e parar a captura se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fechar todas as janelas quando o loop terminar
cap.release()
cv2.destroyAllWindows()
