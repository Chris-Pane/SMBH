import pygame as pg
from matrix_functions import *

class Camera:

    def __init__(self, render, position):
        self.render = render
        self.position = np.array([*position, 1.0])
        self.forward = np.array([0, 0, 1, 1])
        self.up = np.array([0, 1, 0, 1])
        self.right = np.array([1, 0, 0, 1])

        self.GLOBforward = np.array([0, 0, 1, 1])
        self.GLOBup = np.array([0, 1, 0, 1])
        self.GLOBright = np.array([1, 0, 0, 1])

        self.PitchAngle = 0
        self.YawAngle = 0
        self.radius = 10

        self.h_fov = math.pi / 3
        self.v_fov = self.h_fov * (render.HEIGHT / render.WIDTH)
        self.near_plane = 0.1
        self.far_plane = 100
        self.moving_speed = 0.3
        self.rotation_speed = 0.05

    # ------------------------------------------------------------------

    def control(self):
        key = pg.key.get_pressed()
        if key[pg.K_a]:
            # move left - rotate right
            angle = np.pi / 100
            self.camera_yaw(-1 * self.YawAngle)
            self.YawAngle += angle
            self.camera_yaw( self.YawAngle )
            self.camera_pitch(-1 * self.PitchAngle)
            self.camera_pitch(self.PitchAngle)
            self.updateCameraPosition()

        if key[pg.K_d]:
            # move left - rotate right
            angle = np.pi / 100
            self.camera_yaw(-1 * self.YawAngle)
            self.YawAngle -= angle
            self.camera_yaw( self.YawAngle ) 
            self.camera_pitch(-1 * self.PitchAngle)
            self.camera_pitch(self.PitchAngle)
            self.updateCameraPosition()

        if key[pg.K_w]:
            # zoom in
            self.radius -= 0.1
            self.updateCameraPosition()

        if key[pg.K_s]:
            # zoom out
            self.radius += 0.1
            self.updateCameraPosition()

        if key[pg.K_c]:
            # move down - rotate up
            angle = np.pi / 100
            self.camera_pitch(-1 * self.PitchAngle)
            self.PitchAngle -= angle
            self.camera_pitch(self.PitchAngle)
            self.camera_yaw(-1 * self.YawAngle)
            self.camera_yaw( self.YawAngle )
            self.updateCameraPosition()
            
        if key[pg.K_e]:
            # move up - rotate down
            angle = np.pi / 100
            self.camera_pitch(-1 * self.PitchAngle)
            self.PitchAngle += angle
            self.camera_pitch(self.PitchAngle)
            self.camera_yaw(-1 * self.YawAngle)
            self.camera_yaw( self.YawAngle )
            self.updateCameraPosition()


        # if key[pg.K_LEFT]:
        #     self.camera_yaw(-self.rotation_speed * np.pi)
        # if key[pg.K_RIGHT]:
        #     self.camera_yaw(self.rotation_speed * np.pi)
        # if key[pg.K_UP]:
        #     self.camera_pitch(-self.rotation_speed * np.pi)
        # if key[pg.K_DOWN]:
        #     self.camera_pitch(self.rotation_speed * np.pi)

    # ------------------------------------------------------------------

    def updateCameraPosition(self):
        self.position = np.array([
            self.radius * np.sin(self.YawAngle) * np.cos(self.PitchAngle),
            self.radius * np.sin(self.PitchAngle),
            self.radius * np.cos(self.YawAngle) * np.cos(self.PitchAngle),
            1
            ])

        self.forward = np.array([-self.position[0]/self.radius, -self.position[1]/self.radius, -self.position[2]/self.radius, 1])
        self.right = np.array([-self.forward[2],0,self.forward[0],1])
        leng = np.linalg.norm(self.right[:3])
        self.right = np.array([self.right[0]/leng, 0, self.right[2]/leng, 1])
        self.up = np.array([
            -self.forward[0]*self.forward[1],
            self.forward[0]*self.forward[0] - self.forward[1]*self.forward[2],
            -self.forward[1]*self.forward[1],
            1
        ])
        leng = np.linalg.norm(self.up[:3])
        self.up = np.array([self.up[0]/leng, self.up[1]/leng, self.up[2]/leng, 1])

        print(np.linalg.norm(self.up[:3]))

    def updateCameraRotation(self):

        leng = np.sqrt(self.position[2]**2 + self.position[1]**2)
        self.camera_yaw_GLOB( -1*np.arctan( self.position[0] / leng ) )

        leng = np.sqrt(self.position[0]**2 + self.position[2]**2)
        self.camera_pitch_GLOB( np.arctan( self.position[1] / leng ) )

    def camera_pitch_GLOB(self, angle):
        #reverse old rotation
        self.camera_pitch(-1 * self.PitchAngle)
        #apply new rotation
        self.camera_pitch(angle)

    def camera_yaw_GLOB(self, angle):
        #reverse old rotation
        self.camera_yaw(-1 * self.YawAngle)
        #apply new rotation
        self.camera_yaw(angle)

    def camera_yaw(self, angle):
        rotate = rotate_y(angle)
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate

    def camera_pitch(self, angle):
        rotate = rotate_x(angle)
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate
        
    # ------------------------------------------------------------------

    def translate_matrix(self):
        x, y, z, w = self.position
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [-x, -y, -z, 1]
        ])

    def rotate_matrix(self):
        rx, ry, rz, w = self.right
        fx, fy, fz, w = self.forward
        ux, uy, uz, w = self.up
        return np.array([
            [rx, ux, fx, 0],
            [ry, uy, fy, 0],
            [rz, uz, fz, 0],
            [0, 0, 0, 1]
        ])

    def camera_matrix(self):
        return self.translate_matrix() @ self.rotate_matrix()