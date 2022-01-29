import pygame as pg
import pygame.gfxdraw
import sys
import SMBH.lib.SMBH as SMBH
import numpy as np



pg.init()
screen = pg.display.set_mode((600, 400))
clock = pg.time.Clock()
buttons = pg.sprite.Group()
num = 1

buttons = []
class Button:
    def __init__(self,text,width,height,pos,elevation, fx=None):
        #Core attributes 
        self.func = fx

        self.pressed = False
        self.elevation = elevation
        self.dynamic_elecation = elevation
        self.original_y_pos = pos[1]
        # top rectangle 
        self.top_rect = pygame.Rect(pos,(width,height))
        self.top_color = '#475F77'
        
        # bottom rectangle 
        self.bottom_rect = pygame.Rect(pos,(width,height))
        self.bottom_color = '#354B5E'
        #text
        self.text = text
        self.text_surf = pygame.gfxdraw.gui_font.render(text,True,'#FFFFFF')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)
        buttons.append(self)

    def change_text(self, newtext):
        self.text_surf = pygame.gfxdraw.gui_font.render(newtext, True,'#FFFFFF')
        self.text_rect = self.text_surf.get_rect(center = self.top_rect.center)

    def draw(self):
    # elevation logic 
        self.top_rect.y = self.original_y_pos - self.dynamic_elecation
        self.text_rect.center = self.top_rect.center 

        self.bottom_rect.midtop = self.top_rect.midtop
        self.bottom_rect.height = self.top_rect.height + self.dynamic_elecation
        pygame.draw.rect(screen,self.bottom_color, self.bottom_rect,border_radius = 12)
        pygame.draw.rect(screen,self.top_color, self.top_rect,border_radius = 12)
        screen.blit(self.text_surf, self.text_rect)
        self.check_click() 

    def check_click(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.top_rect.collidepoint(mouse_pos):
            self.top_color = '#D74B4B'
            if pygame.mouse.get_pressed()[0]:
                self.dynamic_elecation = 0
                self.pressed = True
                self.change_text(f"{self.text}")
            else:
                self.dynamic_elecation = self.elevation
                if self.pressed == True:

                    if self.func:
                        self.func()
                    self.pressed = False
                    self.change_text(self.text)
        else:
        	self.dynamic_elecation = self.elevation
        	self.top_color = '#475F77'


class Sim2D():
    
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

        self.running = True
        self.objects = []
        self.zoom = 10
        self.origin = np.zeros(2)


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

            if event.key == pg.K_UP:
                self.origin[1] -= 10

            if event.key == pg.K_DOWN:
                self.origin[1] += 10

            if event.key == pg.K_LEFT:
                self.origin[0] -= 10

            if event.key == pg.K_RIGHT:
                self.origin[0] += 10

            if event.key == pg.K_w:
                self.zoom += 1

            if event.key == pg.K_s:
                self.zoom -= 1


    def getScreenCoord(self, AbsVec:np.ndarray):
        
        return (self.zoom*AbsVec + self.origin + np.array([self.H_WIDTH, self.H_HEIGHT]))

    def drawCoord(self):
        """
        Draw all Objects to screen
        """

        self.screen.fill(pg.Color(10,10,25))

        # x axis
        pg.draw.line( self.screen, pg.Color(255,0,0), (0,self.origin[1] + self.H_HEIGHT), (self.WIDTH, self.origin[1] + self.H_HEIGHT) )
        # y axis
        pg.draw.line( self.screen, pg.Color(0,255,0), (self.origin[0] + self.H_WIDTH,0), (self.origin[0] + self.H_WIDTH, self.HEIGHT) )

        #for i in range(10):
        #    pg.draw.line(self.screen, pg.Color(100,100,100), (0, self.HEIGHT/10*i), (self.WIDTH, self.HEIGHT/10*i))
        

    # ------------------------------------------------------------------

    def start(self):
        """
        Start the Simulation.
        """

        startVec = SMBH.ParamVec
        h = 10E-6
        self.traceList = []
        dim = 2
        for _ in range(dim):
            self.traceList.append([])

        while True:

            for event in pg.event.get():
                self.handle_event(event)

            if self.running:

                self.drawCoord()

                startVec = SMBH.StormerVerlet(startVec, h, 2)

                pos = SMBH.ParGetPos(startVec, 2)

                for i in range(dim):
                    self.traceList[i].insert(0, np.array([pos[3*i], pos[3*i+1]]) )
                    if len(self.traceList[i]) >= 150:
                        self.traceList[i].pop()


                pos1 = self.getScreenCoord(startVec[3:5])
                pos2 = self.getScreenCoord(startVec[6:8])

                pg.draw.circle(self.screen, pg.Color(255,0,255), (pos1[0], pos1[1]), 5)
                pg.draw.circle(self.screen, pg.Color(255,255,0), (pos2[0], pos2[1]), 5)

                for i in range(dim):
                    for j in range(len(self.traceList[i])-1):
                        t = int((len(self.traceList[i]) - 1 - j) / (len(self.traceList[i])-1) * 255)
                        tCol = pg.Color(t,t,t)
                        #print(self.traceList[i][j])
                        pg.draw.line(self.screen, tCol, self.getScreenCoord(self.traceList[i][j]), self.getScreenCoord(self.traceList[i][j+1]) )



                #self.camera.camera_pitch(pg.time.get_ticks() % 0.005)

                #pg.display.set_caption(str(self.clock.get_fps()))
                #pg.display.set_caption( ("%s, %s, %s, %s") % (pos1[0], pos1[1], pos2[0], pos2[1]) )
                pg.display.flip()
                self.clock.tick(self.FPS)


    def stop(self):
        """
        Pause the Simulation.
        """

        self.running = False
        

'''

def testFunction1():
    print("test1")
    
def testFunction2():
    print("test2")

def testFunction3():
    print("test3")

button1 = Button('Rome',200,40,(100,200),5, testFunction1)
button2 = Button('Milan',200,40,(100,250),5, testFunction2)
button3 = Button('Neaples',200,40,(100,300),5, testFunction3)

def buttons_draw():
	for b in buttons:
		b.draw()


while True:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
			sys.exit()

	screen.fill('#DCDDD8')
	buttons_draw()

	pygame.display.update()
	clock.tick(60)

'''

s = Sim2D()
s.start()