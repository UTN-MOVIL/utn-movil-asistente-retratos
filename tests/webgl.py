from pythreejs import *
from IPython.display import display

cube = Mesh(
    BoxGeometry(1,1,1),
    MeshStandardMaterial(color='orange')
)
scene = Scene(children=[cube, DirectionalLight(position=[3,5,1], intensity=0.8)])
camera = PerspectiveCamera(position=[3,3,3], up=[0,0,1], lookAt=[0,0,0])
renderer = Renderer(camera=camera, scene=scene, controls=[OrbitControls(controlling=camera)])
display(renderer)
