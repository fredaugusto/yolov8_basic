# YOLOv8 Object Detection

Este projeto demonstra como utilizar o modelo YOLOv8 para detecção de objetos em tempo real a partir da webcam e também para detecção em imagens estáticas. O código captura o vídeo da webcam, aplica o modelo de detecção de objetos YOLOv8 e exibe o vídeo com caixas delimitadoras e rótulos anotados. Além disso, o projeto inclui um script para realizar detecções em uma imagem estática.

## Requisitos

- Python 3.8 ou superior
- `opencv-python`: Biblioteca para captura e manipulação de vídeo e imagens.
- `ultralytics`: Biblioteca para o modelo YOLOv8.

## Instalação

1. Clone este repositório:

   ```
   git clone https://github.com/fredaugusto/yolov8_basic.git
   cd yolov8_basic
   ```

2. Instale as dependências:

   ```
   pip install opencv-python ultralytics
   ```

   **Nota:** Se você encontrar problemas com detecções incorretas, aparecer muitas detecções incorretas (dezenas, centenas e até milhares - poisé, acredite) ou se o modelo não estiver funcionando como esperado, pode ser útil tentar uma versão diferente do pacote `ultralytics`. Você pode trocar a versão do pacote com o seguinte comando:

   ```
   pip install ultralytics==8.1.0
   ```

   Substitua `8.1.0` pela versão desejada. A 8.2 foi um completo desastre para mim. :)

3. Baixe o modelo YOLOv8:

   O modelo pré-treinado pode ser baixado diretamente usando o código ou manualmente do repositório do YOLOv8. Substitua `'yolov8m.pt'` pelo caminho para o seu modelo YOLOv8.

## Uso

### Detecção em Tempo Real com Webcam

1. Substitua o caminho do modelo YOLOv8 no código:

   No arquivo `start_from_webcam.py`, substitua `'yolov8m.pt'` pelo caminho do modelo YOLOv8 que você deseja usar.

2. Execute o script:

   ```
   python start_from_webcam.py
   ```

3. Uma janela será exibida mostrando o vídeo da webcam com as detecções de objetos anotadas. O nome da classe e a confiança serão exibidos ao lado das caixas delimitadoras. Pressione a tecla 'q' para sair da aplicação.

### Detecção em Imagem Estática

1. Adicione sua imagem ao diretório do projeto e defina o caminho da imagem no código. No arquivo `start_from_image.py`, o caminho padrão da imagem é `'detect.jpg'`.

2. Execute o script para detectar objetos na imagem:

   ```
   python start_from_image.py
   ```

3. A imagem redimensionada com as caixas delimitadoras e rótulos será exibida em uma janela. O nome da classe e a confiança serão exibidos ao lado das caixas delimitadoras. Pressione qualquer tecla para fechar a janela.

## Arquivos

- `start_from_webcam.py`: Script para detecção de objetos em tempo real a partir da webcam.
- `start_from_image.py`: Script para detecção de objetos em uma imagem estática.
