import cv2
from ultralytics import YOLO
import ctypes

# Função para obter a altura da tela
def get_screen_height():
    user32 = ctypes.windll.user32
    screen_height = user32.GetSystemMetrics(1)  # 1 é para a altura da tela
    return screen_height

# Carregar o modelo YOLOv8m a partir do arquivo 'yolov8m.pt'
model = YOLO('yolov8m.pt')

# Configurar a confiança mínima para considerar uma detecção válida
confidence_threshold = 0.4  # Quanto menor, mais detecta mas com menos confiabilidade

# Caminho para o vídeo de entrada
video_path = 'videos/01.MP4'  # Substitua pelo caminho para o seu vídeo

# Captura o vídeo a partir do arquivo
cap = cv2.VideoCapture(video_path)

# Verificar se o vídeo foi carregado com sucesso
if not cap.isOpened():
    print("Erro ao carregar o vídeo.")  # Mensagem de erro se não for possível carregar o vídeo
else:
    # Obter a altura da tela
    screen_height = get_screen_height()
    
    while True:
        # Capturar um frame do vídeo
        ret, frame = cap.read()
        if not ret:
            break  # Sai do loop quando o vídeo terminar

        # Calcular a nova altura e largura
        new_height = screen_height - 100  # Nova altura da imagem
        (original_height, original_width) = frame.shape[:2]
        
        # Calcular a nova largura mantendo a proporção
        aspect_ratio = original_width / original_height
        new_width = int(new_height * aspect_ratio)
        
        # Redimensionar o frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Enviar o frame redimensionado para o modelo YOLO para realizar a detecção de objetos
        results = model(resized_frame, conf=confidence_threshold)

        # Iterar sobre os resultados das detecções
        for result in results:
            # Obter os nomes das classes detectadas dentro do dicionário
            class_names = result.names

            # Iterar sobre cada detecção no resultado
            for detection in result.boxes:
                # Verificar se a confiança da detecção é maior ou igual ao limite definido
                if detection.conf[0] >= confidence_threshold:

                    # Extrair as coordenadas da caixa delimitadora da detecção
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Coordenadas a partir do canto superior esquerdo (x1, y1) e inferior direito (x2, y2)
                    confidence = detection.conf[0]  # Captura a confiança da detecção
                    class_id = int(detection.cls[0])  # ID da classe detectada

                    # Obter o nome da classe com base no ID da classe
                    class_name = class_names[class_id]
                    
                    # Definir a cor para desenhar a caixa delimitadora
                    myColor = (255, 0, 0)

                    # Desenhar a caixa delimitadora ao redor do objeto detectado
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), myColor, 2)

                    # Inserir o rótulo com o nome da classe e a confiança
                    label = f'{class_name} ({confidence:.2f})'

                    # Adicionar o rótulo ao frame próximo ao topo da caixa delimitadora
                    cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, myColor, 2)

        # Exibir o frame redimensionado com as caixas delimitadoras e rótulos
        cv2.imshow('YOLOv8m Object Detection', resized_frame)

        # Pressionar 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Fechar a captura de vídeo e todas as janelas abertas pelo OpenCV
cap.release()
cv2.destroyAllWindows()
