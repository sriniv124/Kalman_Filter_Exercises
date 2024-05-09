import numpy as np
from sim.sim2d import sim_run

# Simulator options.
options = {}
options['FIG_SIZE'] = [8,8]

options['DRIVE_IN_CIRCLE'] = True
# If False, measurements will be x,y.
# If True, measurements will be x,y, and current angle of the car.
# Required if you want to pass the driving in circle.
options['MEASURE_ANGLE'] = True
options['RECIEVE_INPUTS'] = True

class KalmanFilter:
    def __init__(self):
        # Initial State
        self.prev_time=0
        self.x = np.matrix([[0.],
                            [0.],
                            [0.],
                            [0.],
                            [0.]])

        # Uncertainity Matrix
        self.P = np.matrix([[100., 0.,0.,0.,0.],
                            [0., 100.,0.,0.,0.],
                            [0., 0.,100.,0.,0.],
                            [0., 0.,0.,100.,0.],
                            [0., 0.,0.,0.,100.]])

        # Next State Function
        self.F = np.matrix([[1., 0.,1.,0.,0.],
                            [0., 1.,1.,0.,0.],
                            [0., 0.,1.,0.,0.],
                            [0., 0.,0.,1.,0.],
                            [0., 0.,0.,0.,1.]])
        
        # External force
        self.u= np.matrix([[0.],
                          [0.],
                          [0.],
                          [0.],
                          [0.]])
        
        # Measurement Function
        self.H = np.matrix([[1., 0.,0.,0.,0.],
                            [0., 1.,0.,0.,0.],
                            [0., 0.,1.,0.,0.]])

        # Measurement Uncertainty
        self.R = np.matrix([[0.1,0.0,0.],
                            [0.0,0.1,0.],
                            [0.,0.,0.1 ]])

        # Identity Matrix
        self.I = np.matrix([[1., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0.],
                            [0., 0., 1., 0., 0.],
                            [0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 1.]])
    def predict(self, dt):
        # dt=t-self.prev_time
        x_dot=np.cos(self.x[3])
        y_dot=np.sin(self.x[3])
        self.F[0,2]*=dt*x_dot
        self.F[1,2]*=dt*y_dot
        # self.F[1,3]*=dt*(v+self.u[2])
        self.F[3,4]=dt
        self.P[0,0]+=0.1
        self.P[1,1]+=0.1
        self.P[2,2]+=0.1
        self.P[3,3]+=0.1
        self.x=self.F*self.x+self.u
        self.P=self.F*self.P*np.transpose(self.F)
        return
    
    def measure_and_update(self,measurements, dt):
        # dt=t-self.prev_time
        self.F[0,2]=dt
        self.F[1,3]=dt

        Z=np.matrix(measurements)
        y=np.transpose(Z)-(self.H*self.x)
        S=(self.H*self.P*np.transpose(self.H))+self.R
        k=self.P*np.transpose(self.H)*np.linalg.inv(S)
        self.x=self.x+(k*y)
        self.P=(self.I-(k*self.H))*self.P
        # self.prev_time=t
        return [self.x[0], self.x[1]]

    def recieve_inputs(self, u_steer, u_pedal):
        self.u[2]=u_pedal
        self.u[3]=u_steer
        return

sim_run(options,KalmanFilter)
