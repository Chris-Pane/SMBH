import pygame as pg
from matrix_functions import *
from numba import njit


@njit(fastmath=True)
def any_func(arr, a, b):
    return np.any((arr == a) | (arr == b))


class Object3D:

    def __init__(self, render, vertexes='', faces='', Color="Orange", rotate=False):
        self.render = render
        self.vertexes = np.array([np.array(v) for v in vertexes])
        self.faces = np.array([np.array(face) for face in faces])
        self.center = np.array([0,0,0,1])

        self.font = pg.font.SysFont('Arial', 30, bold=True)
        self.color_faces = [(pg.Color(Color), face) for face in self.faces]
        self.movement_flag, self.draw_vertexes = rotate, False
        self.label = ''

        self.bShouldDrawTrace = False
        self.lineMaxElem = 50
        self.lineList = []

    def draw(self):
        self.screen_projection()
        self.movement()

    def movement(self):
        if self.movement_flag:
            self.rotate_y(-(pg.time.get_ticks() % 0.005))

    def screen_projection(self):
        vertexes = self.vertexes @ self.render.camera.camera_matrix()
        vertexes = vertexes @ self.render.projection.projection_matrix
        vertexes /= vertexes[:, -1].reshape(-1, 1)
        vertexes[(vertexes > 2) | (vertexes < -2)] = 0
        vertexes = vertexes @ self.render.projection.to_screen_matrix
        vertexes = vertexes[:, :2]

        cen = self.center @ self.render.camera.camera_matrix()
        cen = cen @ self.render.projection.projection_matrix
        cen /= cen[-1]
        cen[(cen > 2) | (cen < -2)] = 0
        cen = cen @ self.render.projection.to_screen_matrix
        cen = cen[:2]

        if self.bShouldDrawTrace and len(self.lineList) > 2:
            line = np.array(self.lineList)
            line = self.lineList @ self.render.camera.camera_matrix()
            line = line @ self.render.projection.projection_matrix
            line /= line[:,-1].reshape(-1, 1)
            line[(line > 2) | (line < -2)] = 0
            line = line @ self.render.projection.to_screen_matrix
            line = line[:, :2]

            #TraceColor = [pg.Color(i,i,i,i) for i in range(min( 255, len(line) ))]

            for i in range(len(line)-1):
                t = int((len(line) - 1 - i) / (len(line)-1) * 255)
                tCol = pg.Color(t,t,t)
                pg.draw.line(self.render.screen, tCol, line[i], line[i+1])

            #pg.draw.lines(self.render.screen, TraceColor, False, line)

        pg.draw.circle(self.render.screen, pg.Color('white'), cen, 2)

        for index, color_face in enumerate(self.color_faces):
            color, face = color_face
            polygon = vertexes[face]
            if not any_func(polygon, self.render.H_WIDTH, self.render.H_HEIGHT):
                pg.draw.polygon(self.render.screen, color, polygon, 1)
                if self.label:
                    text = self.font.render(self.label[index], True, pg.Color('white'))
                    self.render.screen.blit(text, polygon[-1])

        if self.draw_vertexes:
            for vertex in vertexes:
                if not any_func(vertex, self.render.H_WIDTH, self.render.H_HEIGHT):
                    pg.draw.circle(self.render.screen, pg.Color('white'), vertex, 2)

    def UpdateTrace(self):
        self.lineList.insert(0, self.center)
        if len(self.lineList) >= self.lineMaxElem:
            self.lineList.pop()

    def translate(self, pos):
        self.vertexes = self.vertexes @ translate(pos)
        self.center = self.center @ translate(pos)
        self.UpdateTrace()

    def scale(self, scale_to):
        self.vertexes = self.vertexes @ scale(scale_to)
        self.center = self.center @ scale(scale_to)
        self.UpdateTrace()

    def rotate_x(self, angle):
        self.vertexes = self.vertexes @ rotate_x(angle)
        self.center = self.center @ rotate_x(angle)
        self.UpdateTrace()

    def rotate_y(self, angle):
        self.vertexes = self.vertexes @ rotate_y(angle)
        self.center = self.center @ rotate_y(angle)
        self.UpdateTrace()

    def rotate_z(self, angle):
        self.vertexes = self.vertexes @ rotate_z(angle)
        self.center = self.center @ rotate_z(angle)
        self.UpdateTrace()


class Axes(Object3D):
    def __init__(self, render):
        super().__init__(render)
        self.vertexes = np.array([(0, 0, 0, 1), (2, 0, 0, 1), (0, 2, 0, 1), (0, 0, 2, 1)])
        self.faces = np.array([(0, 1), (0, 2), (0, 3)])
        self.colors = [pg.Color('red'), pg.Color('green'), pg.Color('blue')]
        self.color_faces = [(color, face) for color, face in zip(self.colors, self.faces)]
        self.draw_vertexes = False
        self.label = 'XYZ'
