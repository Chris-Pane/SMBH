import pygame as pg
import sys
from object import *
from camera import *
from projection import *
import pathlib

# ------------------------------------------------------------------

class Simulation():
    
    def __init__(self, dim = (1280,720)) -> None:
        """
        Creates the Frame of the Simulation

        dim : Dimensions of the Frame
        """

        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = dim[0], dim[1]
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        self.camera = Camera(self, [0, 0, -10])
        self.projection = Projection(self)

        self.running = True
        self.objects = []


    def handle_event(self, event):
        """
        Handle Quit and Simulation Pause Event
        """

        if event.type == pg.QUIT:
            self.running = False
            pg.quit()
            sys.exit()

        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                self.running = not self.running


    def AddSphere(self, pos, Color="Orange"):
        """
        Adds a Sphere to the Draw Call at Position = pos. Color can be changed
        """

        pa = pathlib.Path(__file__).parent.resolve()
        tempObj = self.get_object_from_file(str(pa) + "/Sphere.obj", Color)
        tempObj.translate(pos)
        self.objects.append(tempObj)


    def AddCoord(self):
        """
        Adds a Coordinate System indicator in the middle of the screen.
        """
        tempObj = Axes(self)
        self.objects.append(tempObj)


    def get_object_from_file(self, filename, Color):
        """
        Load the Sphere Model from disk and create an Object class from it. Internal use
        """

        vertex, faces = [], []
        with open(filename) as f:
            for line in f:
                if line.startswith('v '):
                    vertex.append([float(i) for i in line.split()[1:]] + [1])
                elif line.startswith('f'):
                    faces_ = line.split()[1:]
                    faces.append([int(face_.split('/')[0]) - 1 for face_ in faces_])
        return Object3D(self, vertex, faces, Color, rotate=False)


    def draw(self):
        """
        Draw all Objects to screen
        """

        self.screen.fill(pg.Color('Black'))
        for x in self.objects:
            x.draw()

    # ------------------------------------------------------------------

    def start(self):
        """
        Start the Simulation.
        """

        while True:

            for event in pg.event.get():
                self.handle_event(event)

            if self.running:

                self.draw()
                self.camera.control()

                #self.camera.camera_pitch(pg.time.get_ticks() % 0.005)

                #pg.display.set_caption(str(self.clock.get_fps()))
                pg.display.set_caption(str(self.camera.position))
                pg.display.flip()
                self.clock.tick(self.FPS)


    def stop(self):
        """
        Pause the Simulation.
        """

        self.running = False
        
# ------------------------------------------------------------------

s = Simulation()

s.AddCoord()

s.AddSphere((0,0,0), Color="lightblue")
s.AddSphere((10,0,0))

s.start()