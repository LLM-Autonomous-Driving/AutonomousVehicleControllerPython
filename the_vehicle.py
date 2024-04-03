# /opt/homebrew/anaconda3/bin/python python
"""the_vehicle controller."""
import json
import math
import os

from cv2 import Mat
from numpy import uint8, array, ndarray

from mlx.generate import load_model, prepare_inputs, generate_text

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

#  Definition of Program Mode
MANUAL_MODE: int = 0
AUTO_MODE: int = 1
PROGRAM_MODE: int = MANUAL_MODE  # 0: Manual, 1: Auto

LLM_STEERING_DISABLED: bool = True


# import math
from controller import Camera, Lidar
from vehicle import Driver
from flask import (Flask)
from threading import Thread
from publisher import Publisher
import cv2
# import PIL.Image as Image
import numpy as np

PORT = 5300
HOST = '0.0.0.0'
DEBUG = True
USE_RELOADER = False

TIME_STEP: int = 50  # (in ms) / Specify the time step of the simulation
KAFKA_SERVER: str = 'localhost:39093'  # Kafka server address

UNKNOWN: float = 99999.99  # Unknown value for Lidar data
FILTER_SIZE: int = 3  # Size of the filter for Camera data

#  Definition of Sensor Names
CAMERA_NAME: str = "camera"
LIDAR_NAME: str = "Sick LMS 291"

CAR_SENSOR_TOPIC: str = 'CarSensorData'



#  Definition of Max Values
MAX_BRAKE_INTENSITY: float = 0.4  # (in percentage)
MAX_SPEED: float = 30.0  # (in km/h)
MAX_STEERING_ANGLE: float = 0.5  # (in radians)
MIN_STEERING_ANGLE: float = -0.5  # (in radians)
MAX_STEERING_ANGLE_DIFFERENCE: float = 0.1  # (in radians)
MIN_STEERING_ANGLE_DIFFERENCE: float = -0.1  # (in radians)

# PID Constants
Kp = 0.25
Ki = 0.006
Kd = 2

processor, model = load_model("llava-hf/llava-1.5-7b-hf")
max_tokens, temperature = 128, 0.0
prompt = "USER: <image>\nWhat are these?\nASSISTANT:"


def color_diff(a, b):
    diff = 0
    for i in range(3):
        d = a[i] - b[i]
        if d > 0:
            diff += d
        else:
            diff -= d

    # print(diff)
    return diff


class Vehicle:
    # Indicator does not work in simulation, so only modifying local variables
    INDICATOR_OFF: int = 0
    INDICATOR_RIGHT: int = 1
    INDICATOR_LEFT: int = 2

    def __init__(self, driver: Driver, publisher: Publisher):
        self.driver: Driver = driver
        self.publisher: Publisher = publisher
        self.speed: float = 50.0 if PROGRAM_MODE == AUTO_MODE else 0.0
        self.steering_angle: float = 0.0
        self.indicator: int = 0
        self.brake_intensity: float = 0.0

        self.first_call = True
        self.old_value = [0.0 for i in range(FILTER_SIZE)]

        self.pid_needs_reset = False
        self.integral = 0.0
        self.old_value_pid = 0.0

        self.image = None
        self.camera = self.driver.getDevice(CAMERA_NAME)
        if self.camera:
            self.camera: Camera = Camera(CAMERA_NAME)
            self.camera.enable(TIME_STEP)

            self.camera_width = self.camera.getWidth()
            self.camera_height = self.camera.getHeight()
            self.camera_fov = self.camera.getFov()

        else:
            print("Camera not found")

        self.lidar = self.driver.getDevice(LIDAR_NAME)
        self.lidar_width: int = -1
        self.lidar_range: float = -1.0
        self.lidar_fov: float = -1.0
        if self.lidar:
            self.lidar: Lidar = Lidar(LIDAR_NAME)
            self.lidar.enable(TIME_STEP)
            self.lidar.enablePointCloud()

            self.lidar_width = self.lidar.getHorizontalResolution()
            self.lidar_range = self.lidar.getMaxRange()
            self.lidar_fov = self.lidar.getFov()
        else:
            print("Lidar not found")

    def create_app(self):
        # create and configure the api
        api = Flask(__name__)

        @api.route('/stop')
        def stop():
            self.speed = 0.0
            self.steering_angle = 0.0
            return 'stopped'

        @api.route('/start')
        def start():
            self.speed = 2.0
            return 'started'

        @api.route('/getSteeringAngle')  # GET /getSteeringAngle
        def get_steering_angle():
            self.steering_angle = self.driver.getSteeringAngle()
            return str(self.steering_angle)

        @api.route('/setSteeringAngle/<angle>')  # GET /setSteeringAngle/0.5
        def set_steering_angle(angle: float = 0.0):
            angle = float(angle)
            # Limit the difference between the current and new angle
            # if angle - self.steering_angle > MAX_STEERING_ANGLE_DIFFERENCE:
            #     angle = self.steering_angle + MAX_STEERING_ANGLE_DIFFERENCE
            # elif angle - self.steering_angle < MIN_STEERING_ANGLE_DIFFERENCE:
            #     angle = self.steering_angle + MIN_STEERING_ANGLE_DIFFERENCE
            #
            # # Limit the angle to the range of -0.5 to 0.5
            # if angle > MAX_STEERING_ANGLE:
            #     angle = MAX_STEERING_ANGLE
            # elif angle < MIN_STEERING_ANGLE:
            #     angle = MIN_STEERING_ANGLE

            self.steering_angle = float(angle)
            return 'Steering angle set to' + str(angle)

        @api.route('/getBrakeIntensity')  # GET /getBrakeIntensity
        def get_brake_intensity():
            self.brake_intensity = self.driver.getBrakeIntensity()
            return str(self.brake_intensity)

        @api.route('/setBrakeIntensity/<intensity>')  # GET /setBrakeIntensity/0.5
        def set_brake_intensity(intensity: float = 0.0):
            intensity = float(intensity)
            # Limit the brake intensity to the range of 0 to 0.4
            if intensity > MAX_BRAKE_INTENSITY:
                intensity = MAX_BRAKE_INTENSITY
            elif intensity < 0:
                intensity = 0

            self.brake_intensity = float(intensity)
            return 'Brake intensity set to' + str(intensity)

        @api.route('/getSpeed')  # GET /getSpeed
        def get_speed():
            self.speed = self.driver.getCurrentSpeed()
            return str(self.speed)

        @api.route('/setSpeed/<speed>')  # GET /setSpeed/2
        def set_speed(speed: float = 0):
            speed = float(speed)
            # Limit the speed to the range of 0 to MAX_SPEED
            if speed > MAX_SPEED:
                self.speed = MAX_SPEED  # Max Speed
            elif speed < 0:
                self.speed = 0.0
            else:
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
        self.driver.setCruisingSpeed(self.speed)

    def adjust_steering_angle(self):
        angle = float(self.steering_angle)

        # Limit the difference between the current and new angle
        if angle - self.steering_angle > MAX_STEERING_ANGLE_DIFFERENCE:
            angle = self.steering_angle + MAX_STEERING_ANGLE_DIFFERENCE
        elif angle - self.steering_angle < MIN_STEERING_ANGLE_DIFFERENCE:
            angle = self.steering_angle + MIN_STEERING_ANGLE_DIFFERENCE

        # Limit the angle to the range of -0.5 to 0.5
        if angle > MAX_STEERING_ANGLE:
            angle = MAX_STEERING_ANGLE
        elif angle < MIN_STEERING_ANGLE:
            angle = MIN_STEERING_ANGLE

        self.steering_angle = float(angle)
        self.driver.setSteeringAngle(self.steering_angle)

    def adjust_brake_intensity(self):
        self.driver.setBrakeIntensity(self.brake_intensity)

    def apply_pid(self, angle):
        if self.pid_needs_reset:
            self.pid_needs_reset = False
            self.old_value_pid = angle
            self.integral = 0.0

        if angle != self.old_value_pid:
            self.integral = 0.0

        diff = angle - self.old_value_pid



        if self.integral < 30 and self.integral > -30:
            self.integral += angle

        self.old_value_pid = angle
        print(f"Angle: {angle}, Old Value: {self.old_value_pid}, Diff: {diff}")
        return Kp * angle + Ki * self.integral + Kd * diff

    def filter_angle(self, new_value):

        if self.first_call | (new_value == UNKNOWN):
            self.first_call = False
            for i in range(FILTER_SIZE):
                self.old_value[i] = 0.0
        else:
            for i in range(FILTER_SIZE - 1):
                self.old_value[i] = self.old_value[i + 1]

        if new_value == UNKNOWN:
            return UNKNOWN
        else:
            self.old_value[FILTER_SIZE - 1] = new_value
            sum = 0.0
            for i in range(FILTER_SIZE):
                sum += self.old_value[i]
            return sum / FILTER_SIZE

    def process_camera_image(self, image: bytes):
        num_pixels = self.camera_width * self.camera_height
        ref = [95, 187, 203]
        sum_of_x = 0
        pixel_count = 0
        pixel = 0

        for x in range(num_pixels):
            pixel = image[x * 4: x * 4 + 3]  # Extract BGR values from image
            if color_diff(pixel, ref) < 30:
                sum_of_x += x % self.camera_width
                pixel_count += 1

        if pixel_count == 0:
            return UNKNOWN

        return ((sum_of_x / pixel_count / self.camera_width) - 0.5) * self.camera_fov

    # Returns approximate (angle, distance) of obstacle
    def process_sick_lidar(self, lidar_data: list[float]):
        HALF_AREA = 20  # Check 20 degrees to the left and right of the vehicle
        sum_of_x = 0
        collision_count = 0
        obstacle_distance = 0.0

        for x in range(round(self.lidar_width / 2 - HALF_AREA), round(self.lidar_width / 2 + HALF_AREA)):
            if lidar_data[x] < 20.0:
                sum_of_x += x
                collision_count += 1
                obstacle_distance += lidar_data[x]

        if collision_count == 0:
            return UNKNOWN, obstacle_distance

        obstacle_distance /= collision_count

        return (sum_of_x / collision_count / self.lidar_width - 0.5) * self.lidar_fov, obstacle_distance

    def get_and_publish_camera_data(self, time_step: int):
        if self.camera:
            camera_image = self.camera.getImage()
            focal_length = self.camera.getFocalLength()
            # image = cv2.imread("image.png")
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.resize(image, (640, 480))
            # image = Image.fromarray(image)
            # image.show()

            self.image = np.frombuffer(camera_image, np.uint8).reshape(
                (self.camera.getHeight(), self.camera.getWidth(), 4)
            )
            # cv2.imshow(CAMERA_NAME, self.image)

            # self.publisher.publish_camera_data(time_step, camera_image)

    def get_and_publish_lidar_data(self, time_step: int):
        if self.lidar:
            scan = self.lidar.getRangeImage()
            point_cloud_data = self.lidar.getPointCloud()
            # self.publisher.publish_lidar_data(time_step, point_cloud_data, scan)

    def auto_steer(self, enable_collision_avoidance, i):
        camera_image = self.camera.getImage()
        sick_data = self.lidar.getRangeImage()
        yellow_line_angle = self.filter_angle(self.process_camera_image(camera_image))

        obstacle_angle, obstacle_dist = self.process_sick_lidar(sick_data)  # Returns (angle, distance)

        # print(f"Yellow Line Angle: {yellow_line_angle}")
        # print(f"Obstacle Angle: {obstacle_angle}")
        # print(f"Obstacle Distance: {obstacle_dist}")

        if enable_collision_avoidance and obstacle_angle != UNKNOWN:
            # An obstacle has been detected
            self.brake_intensity = 0.0
            self.adjust_brake_intensity()
            obstacle_steering = self.steering_angle
            if 0.0 < obstacle_angle < 0.04:
                obstacle_steering = self.steering_angle + (obstacle_angle - 0.25) / obstacle_dist
            elif obstacle_angle > -0.04:
                obstacle_steering = self.steering_angle + (obstacle_angle + 0.25) / obstacle_dist
            steer = self.steering_angle
            if yellow_line_angle != UNKNOWN:
                line_following_steering = self.apply_pid(yellow_line_angle)
                if obstacle_steering > 0 and line_following_steering > 0:
                    steer = max(obstacle_steering, line_following_steering)
                elif obstacle_steering < 0 and line_following_steering < 0:
                    steer = min(obstacle_steering, line_following_steering)
            else:
                self.pid_needs_reset = True
            self.steering_angle = steer
            self.adjust_steering_angle()
            # print("yellow line angle: %f, obstacle angle: %f, obstacle dist: %f, obstacle steering: %f, steer: %f\n" %
            #         (yellow_line_angle, obstacle_angle,
            #         obstacle_dist, obstacle_steering, steer))
            # print("Avoiding obstacle")
        elif yellow_line_angle != UNKNOWN and enable_collision_avoidance:
            # No obstacle has been detected, simply follow the line
            self.brake_intensity = 0.0
            self.steering_angle = self.apply_pid(yellow_line_angle)
            self.adjust_steering_angle()
            self.adjust_steering_angle()
            # print("Following the line")
        elif enable_collision_avoidance:
            # No obstacle has been detected but the line is lost => brake and hope to find the line again
            self.brake_intensity = 0.4
            self.pid_needs_reset = True
            print("Lost the line")

        if obstacle_angle != UNKNOWN:  # will always publish data if obstacle exists
            #  LLM Steering
            data = str({
                "yellow_line_angle": yellow_line_angle,
                "obstacle_distance": obstacle_dist,
                "obstacle_angle": obstacle_angle,
                "brake": self.brake_intensity,
                "speed": self.speed,
                "steering_angle": self.steering_angle,
            })
            self.publisher.publish(CAR_SENSOR_TOPIC, str(i), data)

        # print(f"Steering Angle: {self.steering_angle}")

    def llm_annotate_image(self, ):
        image_location = "image.jpg"

        self.camera.saveImage(image_location, 100)

        self.image = np.frombuffer(self.camera.getImage(), np.uint8).reshape(
            (self.camera.getHeight(), self.camera.getWidth(), 4)
        )
        cv2.imshow(CAMERA_NAME, self.image)


        input_ids, pixel_values = prepare_inputs(processor, image_location, prompt)

        reply = generate_text(
            input_ids, pixel_values, model, processor, max_tokens, temperature
        )
        print(reply)

        #Delete the saved image after processing
        os.remove("image.jpg")




    def __del__(self):
        if self.camera:
            self.camera.disable()
        if self.lidar:
            self.lidar.disable()
        self.driver.__del__()


def run_server():
    driver: Driver = Driver()
    publisher: Publisher = Publisher(KAFKA_SERVER)
    vehicle: Vehicle = Vehicle(driver, publisher)
    app = vehicle.create_app()

    cv2.startWindowThread()
    cv2.namedWindow(CAMERA_NAME, cv2.WINDOW_NORMAL)

    driver.step()  # Start the driver and make the first step

    #  Start the server in a separate thread
    server_thread = Thread(target=app.run, kwargs={
        'host': HOST, 'port': PORT, 'debug': DEBUG, 'use_reloader': USE_RELOADER
    })
    server_thread.start()

    i: int = 0  # TimeStep counter
    line_counter: int = 0  # Line counter
    current_time_step: float = driver.getBasicTimeStep()
    cv2.resizeWindow(CAMERA_NAME, vehicle.camera.getWidth() * 2, vehicle.camera.getHeight() * 2)
    while driver.step() != -1:
        # if i > 200:
        #     vehicle.auto_steer(LLM_STEERING_DISABLED, i)
        vehicle.adjust_brake_intensity()
        vehicle.adjust_speed()
        vehicle.adjust_steering_angle()

        #  Publish camera and lidar data on every TimeStep ms
        if (i % int(TIME_STEP / current_time_step)) == 0:
            if i > 200:
                if PROGRAM_MODE == AUTO_MODE:
                    vehicle.auto_steer(LLM_STEERING_DISABLED, i)
            vehicle.llm_annotate_image()

            # print(filter_angle(vehicle.process_camera_image()))
            # vehicle.get_and_publish_camera_data(i)
            # vehicle.get_and_publish_lidar_data(i)
            # print(f"Published camera and lidar data at {i} ms")
            # print(vehicle.camera.getImageArray())
            pass

        cv2.waitKey(1)

        i += 1  # Increment TimeStep counter

    vehicle.__del__()  # Cleanup


if __name__ == "__main__":
    run_server()
