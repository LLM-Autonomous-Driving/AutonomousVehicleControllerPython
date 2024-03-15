# /opt/homebrew/anaconda3/bin/python python
"""the_vehicle controller."""
import uuid

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor


# import math
from controller import Camera, Lidar
from vehicle import Driver
from flask import (Flask)
from threading import Thread
from publisher import Publisher
from numpy import array, inf

PORT = 5300
HOST = '0.0.0.0'
DEBUG = True
USE_RELOADER = False

TIME_STEP: int = 1000  # (in ms) / Specify the time step of the simulation
KAFKA_SERVER: str = 'localhost:29092'  # Kafka server address

#  Definition of Sensor Names
CAMERA_NAME: str = "camera"
LIDAR_NAME: str = "Sick LMS 291"

#  Definition of Topics
CAMERA_IMAGE_DATA_TOPIC: str = 'camera_image_data'
LIDAR_RANGE_IMAGE_DATA_TOPIC: str = 'lidar_range_image_data'
LIDAR_POINT_CLOUD_DATA_TOPIC: str = 'lidar_point_cloud_data'


#  Definition of Max Values


class Vehicle:
    # Indicator does not work in simulation, so only modifying local variables
    INDICATOR_OFF: int = 0
    INDICATOR_RIGHT: int = 1
    INDICATOR_LEFT: int = 2

    def __init__(self, driver: Driver, publisher: Publisher):
        self.driver: Driver = driver
        self.publisher: Publisher = publisher
        self.speed: float = 0.0
        self.steering_angle: float = 0.0
        self.indicator: int = 0

        self.camera = self.driver.getDevice(CAMERA_NAME)
        if self.camera:
            self.camera: Camera = Camera(CAMERA_NAME)
            self.camera.enable(TIME_STEP)
        else:
            print("Camera not found")

        self.lidar = self.driver.getDevice(LIDAR_NAME)
        if self.lidar:
            self.lidar: Lidar = Lidar(LIDAR_NAME)
            self.lidar.enable(TIME_STEP)
            self.lidar.enablePointCloud()

        else:
            print("Lidar not found")

    def create_app(self):
        # create and configure the api
        api = Flask(__name__)

        @api.route('/stop')
        def stop():
            self.speed = 0
            self.steering_angle = 0
            return 'stopped'

        @api.route('/start')
        def start():
            self.speed = 2
            return 'started'

        @api.route('/getSteeringAngle')  # GET /getSteeringAngle
        def get_steering_angle():
            self.steering_angle = self.driver.getSteeringAngle()
            return str(self.steering_angle)

        @api.route('/setSteeringAngle/<angle>')  # GET /setSteeringAngle/0.5
        def set_steering_angle(angle=0.0):
            # Limit the difference between the current and new angle
            if angle - self.steering_angle > 0.1:
                angle = self.steering_angle + 0.1
            elif angle - self.steering_angle < -0.1:
                angle = self.steering_angle - 0.1

            # Limit the angle to the range of -0.5 to 0.5
            if angle > 0.5:
                angle = 0.5
            elif angle < -0.5:
                angle = -0.5

            self.steering_angle = float(angle)
            return 'Steering angle set to' + str(angle)

        @api.route('/getBrakeIntensity')  # GET /getBrakeIntensity
        def get_brake_intensity():
            self.brake_intensity = self.driver.getBrakeIntensity()
            return str(self.brake_intensity)

        @api.route('/setBrakeIntensity/<intensity>')  # GET /setBrakeIntensity/0.5
        def set_brake_intensity(intensity=0.0):
            # Limit the brake intensity to the range of 0 to 0.4
            if intensity > 0.4:
                intensity = 0.4
            elif intensity < 0:
                intensity = 0

            self.brake_intensity = float(intensity)
            return 'Brake intensity set to' + str(intensity)

        @api.route('/getSpeed')  # GET /getSpeed
        def get_speed():
            self.speed = self.driver.getCurrentSpeed()
            return str(self.speed)

        @api.route('/setSpeed/<speed>')  # GET /setSpeed/2
        def set_speed(speed=0):
            self.speed = float(speed)
            return 'Speed set to' + str(speed)

        @api.route('/getIndicator')  # GET /getIndicator
        def get_indicator():
            match self.indicator:
                case self.INDICATOR_OFF:
                    return 'Off'
                case self.INDICATOR_RIGHT:
                    return 'Right'
                case self.INDICATOR_LEFT:
                    return 'Left'
                case _:
                    # Reset to Off
                    self.indicator = self.INDICATOR_OFF
                    return 'Off'

        @api.route('/setIndicatorOff')  # GET /setIndicatorOff
        def set_indicator_off():
            self.indicator = self.INDICATOR_OFF
            return 'Indicator set to Off'

        @api.route('/setIndicatorRight')  # GET /setIndicatorRight
        def set_indicator_right():
            self.indicator = self.INDICATOR_RIGHT
            return 'Indicator set to Right'

        @api.route('/setIndicatorLeft')  # GET /setIndicatorLeft
        def set_indicator_left():
            self.indicator = self.INDICATOR_LEFT
            return 'Indicator set to Left'

        return api

    def adjust_speed(self):
        if self.speed > 250:
            self.speed = 250  # Max Speed
        self.driver.setCruisingSpeed(self.speed)

    def adjust_steering_angle(self):
        self.driver.setSteeringAngle(self.steering_angle)

    def publish_camera_data(self, time_step):
        if self.camera:
            image = self.camera.getImage()
            # print("Image: ", image)
            self.publisher.publish(CAMERA_IMAGE_DATA_TOPIC, str(time_step), image)

    def publish_lidar_data(self, time_step):
        if self.lidar:
            scan = self.lidar.getRangeImage()
            point_cloud_data = self.lidar.getPointCloud()

            scan = array(scan)
            scan[scan == inf] = 0
            scan = scan.tobytes()

            point_cloud = []
            for point in point_cloud_data: # Extract Lidar Points from Object
                point_cloud.append([point.x, point.y, point.z, point.time, point.layer])

            point_cloud = array(point_cloud)
            point_cloud[point_cloud == inf] = 0
            point_cloud = point_cloud.tobytes()

            self.publisher.publish(LIDAR_RANGE_IMAGE_DATA_TOPIC, str(time_step), scan)
            self.publisher.publish(LIDAR_POINT_CLOUD_DATA_TOPIC, str(time_step), point_cloud)

    def __del__(self):
        if self.camera:
            self.camera.disable()
        if self.lidar:
            self.lidar.disable()
        self.driver.__del__()


def run_server():
    current_time_step = 0

    driver: Driver = Driver()
    publisher: Publisher = Publisher(KAFKA_SERVER)
    vehicle: Vehicle = Vehicle(driver, publisher)
    app = vehicle.create_app()

    #  Start the server in a separate thread
    server_thread = Thread(target=app.run, kwargs={
        'host': HOST, 'port': PORT, 'debug': DEBUG, 'use_reloader': USE_RELOADER
    })
    server_thread.start()

    i = 0  # TimeStep counter
    current_time_step = driver.getBasicTimeStep()
    while driver.step() != -1:


        vehicle.adjust_speed()
        vehicle.adjust_steering_angle()


        # Publish camera and lidar data on every TimeStep ms
        if (i % int(TIME_STEP / current_time_step)) == 0:
            vehicle.publish_camera_data(i)
            vehicle.publish_lidar_data(i)
            print(f"Published camera and lidar data at {i} ms")

        i += 1  # Increment TimeStep counter

    vehicle.__del__()

if __name__ == "__main__":
    run_server()
