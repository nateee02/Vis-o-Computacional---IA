# Projeto de Visão Computacional

Projeto criado para disciplina Fundamentos de Inteligência Artificial (FIA) - Graduação. Prof. Pablo De Chiaro

O projeto configura um ambiente virtual em python e instala dependências necessárias para p Projeto de Visão Computacional usando o exemplo 'detecção de objetos' que vai utilizar a webcam do computador para detectar copos e informar a cor predominante.

## Configuração do Ambiente Virtual

### Passos para criar e ativar um ambiente virtual:

1. **Criar o ambiente virtual:**

   ```bash
   python -m venv env-visao
   ```

2. **Ativar o ambiente virtual:**

   No macOS e Linux:

   ```bash
   source ./env-visao/bin/activate
   ```

   No Windows:

   ```bash
   .\env-visao\Scripts\activate
   ```

## Instalação de Dependências

Certifique-se de que seu ambiente virtual esteja ativado. Instale as dependências listadas no arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Conteúdo do arquivo `requirements.txt`:

```text
numpy==2.0.0
opencv-python==4.10.0.84
```

## Verificação da Instalação

Para verificar se as bibliotecas foram instaladas corretamente, você pode executar o seguinte comando em um terminal Python:

```python
import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")
print(f"NumPy version: {np.__version__}")
```

## Desativação do Ambiente Virtual

Quando terminar de trabalhar no projeto, você pode desativar o ambiente virtual com o comando:

```bash
deactivate
```
