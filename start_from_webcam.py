import cv2
from ultralytics import YOLO

# Carregue o modelo YOLOv8
model = YOLO('yolov8n.pt')  # Substitua pelo caminho para o seu modelo YOLOv8

# Abra a webcam
cap = cv2.VideoCapture(0)  # Índice 0 para a webcam padrão

while True:
    # Capture frame a frame
    ret, frame = cap.read()
    
    if not ret:
        print("Não foi possível capturar o vídeo.")
        break

    # Execute a detecção de objetos
    results = model(frame)

    # Processar os resultados
    # `results` pode ser um objeto com atributos como `boxes` e `names`
    for result in results:
        # `boxes` é um atributo que contém as caixas delimitadoras e outras informações
        boxes = result.boxes.xyxy.cpu().numpy()  # Obtém as coordenadas das caixas em formato NumPy
        confidences = result.boxes.conf.cpu().numpy()  # Obtém as confianças
        classes = result.boxes.cls.cpu().numpy()  # Obtém as classes
        
        for i in range(len(boxes)):
            # Extraia as coordenadas da caixa delimitadora
            x1, y1, x2, y2 = boxes[i]
            conf = confidences[i]
            cls = int(classes[i])
            
            # Desenhe a caixa delimitadora e o rótulo
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Class {cls} {conf:.2f}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exiba o frame anotado
    cv2.imshow('YOLOv8 Webcam', frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
cap.release()
cv2.destroyAllWindows()
