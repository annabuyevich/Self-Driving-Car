# Builds the Self Driving Car

# Libraries
import numpy as np
from random import random,randint
import matplotlib.pyplot as plt
import time

# Kivy Packages 
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Import the Artificial Intelligence
from ai import Dqn


# if user clicks right - no red point will be added
Config.set('input','mouse','mouse,multitouch_on_demand')

"""
introducing the prev_x and prev_y, keeps the previous point in memory where
the sand was drawn
the total points and the length of the previous drawing
"""
prev_x = 0
prev_y = 0
total_points = 0 
length = 0

"""
The AI, the 'brain', that represent the Q function and contains our neural 
network. 
"""
brain = Dqn(5,3,0.9) # 5 sensors, 3 actions, gama - 0.9
action2rotation = [0,20,-20] # action - 0 => 0, action - 1 = > rotate 20, etc
prev_reward = 0
scores = []

first_update = True # Initialize map only once
def init():
    global sand # an array of sand if sand value 1 otherwise 0
    global goal_x # where the car has to go
    global goal_y 
    global first_update
    sand = np.zeros((longueur,largeur))
    goal_x = 20 # the goal  to reach is the upper left of the map not 0 don't
                # want to touch the wal
    goal_y = largeur - 20
    first_update = False # used to initialise map only once

prev_distance = 0

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0) # either 0,20,-20
    
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y) # velocity vector
    
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) # sensor vector
    
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y) # sensor vector
    
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y) # sensor vector
    
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    
    def move(self, rotation):
        # updating position of the car according to the previous position and velocity
        self.pos = Vector(*self.velocity) + self.pos 
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        # updating the position of the sensors
        self.sensor1 = Vector(30,0).rotate(self.angle) + self.pos 
        self.sensor2 = Vector(30,0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30,0).rotate((self.angle-30)%360) + self.pos
        
        # getting signal recieved by the sensors
        """ TODO - sand might cause ERROR"""
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10,int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10,int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
    
        # sensor out of the map, detects full sand
        if self.sensor1_x > longueur-10 or self.sensor1_x < 10 or self.sensor1_y > largeur-10 or self.sensor1_y < 10:
                self.signal1 = 1.
                
        if self.sensor2_x > longueur-10 or self.sensor2_x < 10 or self.sensor2_y > largeur-10 or self.sensor2_y < 10:
                self.signal2 = 1.
                
        if self.sensor3_x > longueur-10 or self.sensor3_x < 10 or self.sensor3_y > largeur-10 or self.sensor3_y < 10:
                self.signal3 = 1.
                
        
"""The sensors"""
class Ball1(Widget):
    pass

class Ball2(Widget):
    pass

class Ball3(Widget):
    pass


class Game(Widget):
        
    # getting our objects from our kivy file
    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    
    # starting the car when run the game
    def serve_car(self):
        self.car.center = self.center
        
        # horizonal to the right with speed 6
        self.car.velocity = Vector(6,0)
    
    """Updates everything that needs to be updated at each discrete time t 
        when reaching a new stating
    """
    def update(self, discreteTime):
        
        global brain
        global prev_reward
        global scores
        global prev_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        
        longueur = self.width
        largeur = self.height
        
        # inititalize only once
        if first_update:
            init()
        
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        
        # direction of the car with respect to the goal if the car is 
        # heading towards the goal then the orientation is 0
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.
        
        # state vector
        prev_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        
        # playing the action from the ai
        action = brain.update(prev_reward, prev_signal)
        
        # adding the last 100 rewards from the window
        scores.append(brain.score())
        
        # converting the action played 0,1,2 to 0,20,-20
        rotation = action2rotation[action]
        
        self.car.move(rotation)
        
        # getting the new distance from the car to the goal
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        
        # updating the positions
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3
        
        # penalize the car if it goes onto some sand
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1,0).rotate(self.car.angle)
            prev_reward = -1
        else:
            self.car.velocity = Vector(6,0).rotate(self.car.angle)
            prev_reward = -0.2
            
            if distance < prev_distance:
                prev_reward = 0.5
        
        # left edge of the frame
        if self.car.x < 10:
            self.car.x = 10
            prev_reward = -1
            
        # right edge
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            prev_reward = -1
        
        # bottom edge
        if self.car.y < 10:
            self.car.y = 10
            prev_reward = -1
        
        #  uppder edge
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            prev_reward = -1
        
        # when the car reaches the goal
        if distance < 100:
            # the goal becomes the bottom right corner and vis versa
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y
        
        prev_distance = distance

class MyPaintWidget(Widget):
    
  
    
    # drawing the sand when clicked
    def on_touch_down(self, touch):
        
        global length
        global total_points
        global prev_x
        global prev_y
        
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            prev_x = int(touch.x)
            prev_y = int(touch.y)
            total_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
    
    def on_touch_move(self, touch):
        
        global length
        global total_points
        global prev_x
        global prev_y
        
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            
            length += np.sqrt(max((x - prev_x)**2 + (y - prev_y)**2,2))
            total_points += 1
            density = total_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) -10 
                     : int (touch.y) + 10] = 1
            prev_x = x
            prev_y = y

# Building the App
class CarApp(App):
    
    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 60.0)
        self.painter = MyPaintWidget()
        
        goal_one = Button(text = 'Town', pos = (20, parent.height  + 950),
                   background_color = (1.0, 0.0, 0.0, 1.0))
        goal_two = Button(text = 'Airport', pos = (parent.width + 1350, 20),
                   background_color = (1.0, 0.0, 0.0, 1.0))
        clearbtn = Button(text='clear')
        savebtn = Button(text='save',pos=(parent.width,0))
        loadbtn = Button(text='load', pos=(2 * parent.width,0))
        
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        
        parent.add_widget(self.painter)
        parent.add_widget(goal_one)
        parent.add_widget(goal_two)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
    
    def save(self, obj):
        print("Saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()
    
    def load(self, obj):
        print("Loading last saved brain...")
        brain.load()
    
if __name__ == '__main__':
    CarApp().run()