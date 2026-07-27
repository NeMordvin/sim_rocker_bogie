from dataclasses import dataclass
from math import cos, sin, radians

@dataclass
class Point:
    x: float
    y: float

def point_from(start: Point, length: float, angle_deg: float) -> Point:
    """Calculate a new point given a starting point, length, and angle in degrees."""
    angle_radians = radians(angle_deg)
    new_x = start.x + length * cos(angle_radians)
    new_y = start.y + length * sin(angle_radians)
    return Point(new_x, new_y)

@dataclass
class ChassisPoints:
    main_pivot: Point
    front_wheel: Point
    bogie_pivot: Point
    middle_wheel: Point
    rear_wheel: Point

def calculate_chassis_points(
        main_pivot: Point,
        front_length: float,
        front_angle: float,
        bogie_length: float,
        bogie_angle: float,
        middle_length: float,
        middle_angle: float,
        rear_length: float,
        rear_angle: float,
        rocker_rotation: float = 0.0,
        bogie_rotation: float = 0.0,
    ) -> ChassisPoints:

    """Calculate the positions of key points in the Rocker-Bogie chassis."""
    front_wheel = point_from(main_pivot, front_length, front_angle + rocker_rotation)

    bogie_pivot = point_from(main_pivot, bogie_length, bogie_angle + rocker_rotation)

    middle_wheel = point_from(bogie_pivot, middle_length, middle_angle + bogie_rotation)

    rear_wheel = point_from(bogie_pivot, rear_length, rear_angle + bogie_rotation)

    return ChassisPoints(
        main_pivot=main_pivot,  
        front_wheel=front_wheel,
        bogie_pivot=bogie_pivot,
        middle_wheel=middle_wheel,
        rear_wheel=rear_wheel
    )


# Расчёт высоты колес относительно нулевой линии (y=0)
def calculate_wheel_clearances(
    points: ChassisPoints,
    wheel_radius: float,
) -> dict[str, float]:
    return {
        "Переднее": points.front_wheel.y - wheel_radius,
        "Среднее": points.middle_wheel.y - wheel_radius,
        "Заднее": points.rear_wheel.y - wheel_radius,
    }


# Расчёт корпуса
@dataclass
class BodyPoints:
    bottom_left: Point
    bottom_right: Point
    top_right: Point
    top_left: Point


def calculate_body_points(
    main_pivot: Point,
    body_length: float,
    body_height: float,
    attachment_x: float,
    attachment_y: float,
) -> BodyPoints:

    bottom_left = Point(
        x=main_pivot.x - attachment_x,
        y=main_pivot.y - attachment_y,
    )

    bottom_right = Point(
        x=bottom_left.x + body_length,
        y=bottom_left.y,
    )

    top_right = Point(
        x=bottom_left.x + body_length,
        y=bottom_left.y + body_height,
    )

    top_left = Point(
        x=bottom_left.x,
        y=bottom_left.y + body_height,
    )

    return BodyPoints(
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        top_right=top_right,
        top_left=top_left,
    )

if __name__ == "__main__":
    points = calculate_chassis_points(
        main_pivot=Point(0.0, 400.0),

        front_length=500.0,
        front_angle=-36.87,

        bogie_length=335.41,
        bogie_angle=-153.43,

        middle_length=250.0,
        middle_angle=-36.87,

        rear_length=250.0,
        rear_angle=-143.13,
    )

    print("Главный шарнир:", points.main_pivot)
    print("Переднее колесо:", points.front_wheel)
    print("Шарнир bogie:", points.bogie_pivot)
    print("Среднее колесо:", points.middle_wheel)
    print("Заднее колесо:", points.rear_wheel)