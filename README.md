# Configurar Node + Python con GPU para el modelo de reconocimiento facial **`buffalo_s`**

Sigue estos pasos para crear un entorno de ejecución en Node.js (con posibilidad de GPU) y preparar la instalación de ONNX Runtime:

1. **Crear el directorio del proyecto e ingresar en él**

```bash
mkdir -p js_runtime && cd js_runtime
```

2. **Inicializar un proyecto de Node**

```bash
npm init -y
```

3. **Instalar ONNX Runtime para Node**

```bash
npm i onnxruntime-node
```