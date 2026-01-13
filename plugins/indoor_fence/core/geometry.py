#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
几何核心库 - 电子围栏系统几何计算
基于《变电站多目标操作安全监测系统》方案实现
作者: G组 | 版本: 2.0.0
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Point2D:
    """2D点"""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def distance_to(self, other: 'Point2D') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def __add__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point2D') -> 'Point2D':
        return Point2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Point2D':
        return Point2D(self.x * scalar, self.y * scalar)


@dataclass
class LineSegment:
    """线段"""
    start: Point2D
    end: Point2D
    
    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)
    
    def to_vector(self) -> Point2D:
        return self.end - self.start


@dataclass
class Polygon:
    """多边形"""
    vertices: List[Point2D]
    name: str = ""
    
    @property
    def num_vertices(self) -> int:
        return len(self.vertices)
    
    @property
    def edges(self) -> List[LineSegment]:
        edges = []
        n = len(self.vertices)
        for i in range(n):
            edges.append(LineSegment(self.vertices[i], self.vertices[(i + 1) % n]))
        return edges
    
    @property
    def centroid(self) -> Point2D:
        if not self.vertices:
            return Point2D(0, 0)
        cx = sum(v.x for v in self.vertices) / len(self.vertices)
        cy = sum(v.y for v in self.vertices) / len(self.vertices)
        return Point2D(cx, cy)
    
    @property
    def bounding_box(self) -> Tuple[Point2D, Point2D]:
        if not self.vertices:
            return Point2D(0, 0), Point2D(0, 0)
        min_x = min(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        return Point2D(min_x, min_y), Point2D(max_x, max_y)
    
    @classmethod
    def from_list(cls, coords: List[Tuple[float, float]], name: str = "") -> 'Polygon':
        vertices = [Point2D(x, y) for x, y in coords]
        return cls(vertices, name)
    
    @classmethod
    def create_rectangle(cls, x: float, y: float, width: float, height: float, name: str = "") -> 'Polygon':
        vertices = [Point2D(x, y), Point2D(x + width, y), Point2D(x + width, y + height), Point2D(x, y + height)]
        return cls(vertices, name)


def cross_product_2d(v1: Point2D, v2: Point2D) -> float:
    return v1.x * v2.y - v1.y * v2.x


def dot_product_2d(v1: Point2D, v2: Point2D) -> float:
    return v1.x * v2.x + v1.y * v2.y


def point_in_polygon(point: Point2D, polygon: Polygon, on_edge_is_inside: bool = True) -> bool:
    """判断点是否在多边形内 (射线法)"""
    if polygon.num_vertices < 3:
        return False
    
    min_pt, max_pt = polygon.bounding_box
    if point.x < min_pt.x or point.x > max_pt.x or point.y < min_pt.y or point.y > max_pt.y:
        return False
    
    inside = False
    n = polygon.num_vertices
    j = n - 1
    
    for i in range(n):
        vi = polygon.vertices[i]
        vj = polygon.vertices[j]
        if ((vi.y > point.y) != (vj.y > point.y)) and \
           (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x):
            inside = not inside
        j = i
    
    return inside


def point_to_segment_distance(point: Point2D, segment: LineSegment) -> Tuple[float, Point2D]:
    """计算点到线段的最短距离"""
    v = segment.end - segment.start
    w = point - segment.start
    c1 = dot_product_2d(w, v)
    c2 = dot_product_2d(v, v)
    
    if c2 < 1e-10:
        return point.distance_to(segment.start), segment.start
    
    t = max(0, min(1, c1 / c2))
    closest = Point2D(segment.start.x + t * v.x, segment.start.y + t * v.y)
    return point.distance_to(closest), closest


def point_to_polygon_distance(point: Point2D, polygon: Polygon) -> Tuple[float, Point2D, bool]:
    """计算点到多边形边界的最短距离"""
    is_inside = point_in_polygon(point, polygon)
    min_dist = float('inf')
    closest_point = polygon.vertices[0] if polygon.vertices else Point2D(0, 0)
    
    for edge in polygon.edges:
        dist, closest = point_to_segment_distance(point, edge)
        if dist < min_dist:
            min_dist = dist
            closest_point = closest
    
    if is_inside:
        return -min_dist, closest_point, True
    return min_dist, closest_point, False


def lidar_to_ground(distance: float, angle_deg: float, lidar_height: float = 2.5, lidar_tilt_deg: float = 20.0) -> Point2D:
    """将激光雷达测量转换为地面坐标"""
    angle_rad = math.radians(angle_deg)
    tilt_rad = math.radians(lidar_tilt_deg)
    horizontal_distance = distance * math.cos(tilt_rad)
    x = horizontal_distance * math.sin(angle_rad)
    y = horizontal_distance * math.cos(angle_rad)
    return Point2D(x, y)


def pixel_to_ground(pixel_x: float, pixel_y: float, image_width: int, image_height: int,
                   homography: Optional[np.ndarray] = None, camera_height: float = 3.0, 
                   camera_tilt_deg: float = 40.0) -> Point2D:
    """将像素坐标转换为地面坐标"""
    if homography is not None:
        pt = np.array([pixel_x, pixel_y, 1.0])
        ground_pt = homography @ pt
        ground_pt = ground_pt / ground_pt[2]
        return Point2D(float(ground_pt[0]), float(ground_pt[1]))
    
    nx = (pixel_x - image_width / 2) / image_width
    ny = (pixel_y - image_height / 2) / image_height
    tilt_rad = math.radians(camera_tilt_deg)
    depth_factor = 1.0 / max(0.1, 1.0 - ny * 0.8)
    y_ground = camera_height * math.tan(tilt_rad) * depth_factor
    x_ground = nx * y_ground * 0.5
    return Point2D(x_ground, y_ground)


def fuse_coordinates(visual_pos: Optional[Point2D], lidar_pos: Optional[Point2D],
                    trust_visual_x: bool = True, trust_lidar_y: bool = True) -> Optional[Point2D]:
    """融合视觉和雷达坐标 - 横向信任视觉,纵向信任测距"""
    if visual_pos is None and lidar_pos is None:
        return None
    if visual_pos is None:
        return lidar_pos
    if lidar_pos is None:
        return visual_pos
    x = visual_pos.x if trust_visual_x else lidar_pos.x
    y = lidar_pos.y if trust_lidar_y else visual_pos.y
    return Point2D(x, y)


__all__ = ['Point2D', 'LineSegment', 'Polygon', 'point_in_polygon', 'point_to_segment_distance',
           'point_to_polygon_distance', 'lidar_to_ground', 'pixel_to_ground', 'fuse_coordinates']
