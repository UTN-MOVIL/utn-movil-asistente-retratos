from sanic import Sanic
from sanic.response import text

# from sanic.server.protocols.websocket_protocol import WebSocketProtocol # Probablemente ya no necesites esto

app = Sanic("MiAppHttpWebSocket")

# Endpoint WebSocket (el que ya tenías)
@app.websocket('/ws')
async def websocket_handler(request, ws):
    print(">>> Conexión WebSocket establecida. Esperando mensajes...")
    while True:
        try:
            data = await ws.recv()
            print(f"Mensaje recibido del WebSocket: {data}")

            if not data:
                print(">>> Datos vacíos recibidos o cliente WebSocket cerró la conexión. Terminando bucle.")
                break

            respuesta = f"Recibido vía WebSocket: {data}"
            await ws.send(respuesta)
        except Exception as e:
            print(f">>> ERROR en el manejador WebSocket: {e}")
            break 
    print(">>> Manejador WebSocket finalizado para esta conexión.")

# NUEVO: Endpoint HTTP
@app.route('/http', methods=['GET', 'POST']) # Puedes especificar los métodos HTTP que quieres soportar
async def http_handler(request):
    print(f">>> Solicitud HTTP recibida en /http con método: {request.method}")
    if request.method == 'GET':
        # Para solicitudes GET, simplemente devolvemos un mensaje
        return text("Hola desde el endpoint HTTP (GET)! Si quieres enviar datos, prueba con un POST.")
    elif request.method == 'POST':
        # Para solicitudes POST, podemos acceder a los datos del cuerpo
        # Sanic maneja diferentes tipos de cuerpo (json, form, etc.)
        data_recibida = request.json if request.json else request.form if request.form else request.body
        print(f"Datos recibidos en POST HTTP: {data_recibida}")
        return text(f"Datos recibidos vía HTTP (POST): {data_recibida}")

# NUEVO: Ruta raíz para verificar que el servidor HTTP funciona
@app.route('/', methods=['GET'])
async def root_handler(request):
    print(">>> Solicitud HTTP recibida en / (raíz)")
    return text("¡El servidor Sanic está funcionando! Prueba /ws para WebSocket o /http para HTTP.")

if __name__ == "__main__":
    # Modificado para usar dev=True y sin el parámetro protocol
    app.run(host="0.0.0.0", port=8000, dev=True, debug=True) # Añadido debug=True para más información en la consola
