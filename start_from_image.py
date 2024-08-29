import cv2
from ultralytics import YOLO
import ctypes

# Função para obter a altura da tela
def get_screen_height():
    user32 = ctypes.windll.user32
    screen_height = user32.GetSystemMetrics(1)  # 1 é para a altura da tela
    return screen_height

# Carrega o modelo YOLOv8m a partir do arquivo 'yolov8m.pt'
model = YOLO('yolov8m.pt')

# Configura a confiança mínima para considerar uma detecção válida
confidence_threshold = 0.4  # Quanto menor, mais detecta mas com menos confiabilidade

# Caminho para a imagem de entrada
image_path = 'detect.jpg'  # Define o caminho da imagem como 'detect.jpg'

# Carrega a imagem de entrada
frame = cv2.imread(image_path)

# Verifica se a imagem foi carregada com sucesso
if frame is None:
    print("Erro ao carregar a imagem.")  # Mensagem de erro se não for possível carregar a imagem
else:
    # Obtém a altura da tela
    screen_height = get_screen_height()
    
    # Calcula a nova altura e largura
    new_height = screen_height - 100  # Nova altura da imagem
    (original_height, original_width) = frame.shape[:2]
    
    # Calcula a nova largura mantendo a proporção
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)
    
    # Redimensiona a imagem
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Envia o frame redimensionado para o modelo YOLO para realizar a detecção de objetos
    results = model(resized_frame, conf=confidence_threshold)

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
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), myColor, 2)

                # Insere o rótulo com o nome da classe e a confiança
                label = f'{class_name} ({confidence:.2f})'

                # Adiciona o rótulo ao frame próximo ao topo da caixa delimitadora
                cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, myColor, 2)

    # Exibe a imagem redimensionada com as caixas delimitadoras e rótulos
    cv2.imshow('YOLOv8m Object Detection', resized_frame)

    # Aguarda até que uma tecla seja pressionada para fechar a janela
    cv2.waitKey(0)  # 0 significa aguardar indefinidamente

# Fecha todas as janelas abertas pelo OpenCV
cv2.destroyAllWindows()
