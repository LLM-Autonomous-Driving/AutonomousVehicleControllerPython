import json

from confluent_kafka import Producer
from numpy import inf
import PIL.Image as Image
import io

from controller import LidarPoint


#  Definition of Topics
CAMERA_IMAGE_DATA_TOPIC: str = 'CameraImageRawData'
LIDAR_RANGE_IMAGE_DATA_TOPIC: str = 'LidarRangeImageRawData'
LIDAR_POINT_CLOUD_DATA_TOPIC: str = 'LidarPointCloudRawData'


#  Function to remove inf and nan from data
def remove_inf_and_nan(data):
    if data == inf:
        return 0.0
    elif data == -inf:
        return 0.0
    elif str(data) == "nan":
        return 0.0
    else:
        return data


def show_image(image: bytes):
    image = Image.open(io.BytesIO(image))
    image.show()


class Publisher:
    def __init__(self, server: str):
        self.publisher = Producer({'bootstrap.servers': server})
        self.i = 0

    def publish(self, topic: str, key: str, data: str):

        data = {"channel": topic, "data": {topic: data}, "topic": topic}
        data = json.dumps(data)

        try:
            self.publisher.produce(topic, key=key, value=data)
            self.publisher.flush()
        except Exception as e:
            print(f"Error publishing to {topic}: {e}")
        return f"Published to {topic} with key {key} and value {data}"

    def publish_lidar_data(self, time_step: int, point_cloud_data: list[LidarPoint], range_image: list[float]):

        # range_image = [remove_inf_and_nan(x) for x in range_image]
        range_image = str(range_image)

        point_cloud = []
        for point in point_cloud_data:  # Extract Lidar Points from Object
            # point_cloud.append({
            #     "x": remove_inf_and_nan(point.x),
            #     "y": remove_inf_and_nan(point.y),
            #     "z": remove_inf_and_nan(point.z),
            #     "time": remove_inf_and_nan(point.time),
            #     "layer": remove_inf_and_nan(point.layer)
            # })
            point_cloud.append({
                "x": point.x,
                "y": point.y,
                "z": point.z,
                "time": point.time,
                "layer": point.layer

            })
        point_cloud = str(point_cloud)

        self.publish(LIDAR_RANGE_IMAGE_DATA_TOPIC, str(time_step), range_image)
        self.publish(LIDAR_POINT_CLOUD_DATA_TOPIC, str(time_step), point_cloud)

    def publish_camera_data(self, time_step: int, image: bytes):
        image = str(image)
        self.publish(CAMERA_IMAGE_DATA_TOPIC, str(time_step), image)
