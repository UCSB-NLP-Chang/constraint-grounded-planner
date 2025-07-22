# We define some colors and a function to visualize the simulation here.

import os
import math
import pygame
from typing import List, Dict, Any
from .colors import *


class TileMap:
    def __init__(
        self,
        TILE_SIZE=60,
    ):
        """Initialize the tile map renderer"""
        self.TILE_SIZE = TILE_SIZE
        self.obj_colors = []

        # PAD = max(40, TILE_SIZE // 2)
        PAD = TILE_SIZE // 4
        self.UP_PAD = PAD
        self.LOW_PAD = PAD
        self.LEFT_PAD = PAD
        self.RIGHT_PAD = PAD
        # self.edge_thickness = 10
        self.edge_thickness = 5
        self.tile_images = self.load_tile_images()

    def load_tile_images(self):
        """Loads tile images into a dictionary"""
        assets_dir = os.path.join(os.path.dirname(__file__), "../assets")
        return {
            "robot": pygame.transform.smoothscale(
                pygame.image.load(os.path.join(assets_dir, "robotic-arm.png")),
                (self.TILE_SIZE // 2.5, self.TILE_SIZE // 2.5),
            ),
        }

    def draw_map(self, screen, N, M):
        # This function draws the background map grid
        width = M * self.TILE_SIZE
        height = N * self.TILE_SIZE

        screen.fill(WHITE)

        for i in range(N + 1):
            pygame.draw.rect(
                screen,
                GRAY,
                (
                    self.LEFT_PAD - self.edge_thickness // 2,
                    self.UP_PAD + i * self.TILE_SIZE - self.edge_thickness // 2,
                    width + self.edge_thickness,
                    self.edge_thickness,
                ),
            )
        for j in range(M + 1):
            pygame.draw.rect(
                screen,
                GRAY,
                (
                    self.LEFT_PAD + j * self.TILE_SIZE - self.edge_thickness // 2,
                    self.UP_PAD - self.edge_thickness // 2,
                    self.edge_thickness,
                    height + self.edge_thickness,
                ),
            )

    def draw_object(self, screen, object_dict: Dict):
        # print("Object dict", object_dict)
        for obj_id, obj_pos in object_dict.items():
            color = desaturated_colors[-obj_id % len(desaturated_colors)][1]
            box_size = self.TILE_SIZE // 4
            x, y = obj_pos[1], obj_pos[0]
            pygame.draw.rect(
                screen,
                color,
                (
                    x * self.TILE_SIZE + self.UP_PAD - box_size // 2,
                    y * self.TILE_SIZE + self.LEFT_PAD - box_size // 2,
                    box_size,
                    box_size,
                ),
            )

    def draw_robot(self, screen, robot_dicts: Dict[str, Any]):
        # print("Robot dict", robot_dicts)
        for robot_id, robot_pos in robot_dicts.items():
            robot_pos = robot_pos.base_pos
            screen.blit(
                self.tile_images["robot"],
                (
                    robot_pos[1] * self.TILE_SIZE + self.UP_PAD - self.TILE_SIZE // 6,
                    robot_pos[0] * self.TILE_SIZE + self.LEFT_PAD - self.TILE_SIZE // 6,
                ),
            )

    def draw_target_pos(self, screen, target_pos: Dict):
        # print("Target Pos", target_pos)

        def draw_box_circle(color, x, y):
            circle_center = (
                x * self.TILE_SIZE + self.UP_PAD,
                y * self.TILE_SIZE + self.LEFT_PAD,
            )
            pygame.draw.circle(
                screen,
                color,
                circle_center,
                self.TILE_SIZE // 5,
            )
            box_size = self.TILE_SIZE // 4
            box_rect = pygame.Rect(
                circle_center[0] - box_size // 2,
                circle_center[1] - box_size // 2,
                box_size,
                box_size,
            )
            pygame.draw.rect(screen, WHITE, box_rect)

        for obj_id, obj_pos in target_pos.items():
            color = desaturated_colors[-obj_id % len(desaturated_colors)][1]
            draw_box_circle(color, obj_pos[1], obj_pos[0])

    def draw_arm(
        self, screen, robots, color=BLUE + (160,), border_color=None, is_initial=False
    ):
        for robotid, robot in robots.items():
            start_y, start_x = robot.base_pos
            end_y, end_x = robot.arm_pos
            draw_arrow(
                screen,
                color,
                (
                    start_x * self.TILE_SIZE + self.UP_PAD,
                    start_y * self.TILE_SIZE + self.LEFT_PAD,
                ),
                (
                    end_x * self.TILE_SIZE + self.UP_PAD - 10 * int(is_initial),
                    end_y * self.TILE_SIZE + self.LEFT_PAD,
                ),
                border_color=border_color,
                arrow_head_angle=0,
                arrow_head_length=0,
            )

    def draw_action(self, screen, actions: List, color=RED + (128,), border_color=None):
        for action in actions:
            start_y, start_x = action.pos_s
            end_y, end_x = action.pos_e

            draw_arrow(
                screen,
                color,
                (
                    start_x * self.TILE_SIZE + self.UP_PAD,
                    start_y * self.TILE_SIZE + self.LEFT_PAD,
                ),
                (
                    end_x * self.TILE_SIZE + self.UP_PAD,
                    end_y * self.TILE_SIZE + self.LEFT_PAD,
                ),
                border_color=border_color,
            )

    def draw_invalid_action(self, screen, invalid_actions: List):
        self.draw_action(screen, invalid_actions, border_color=YELLOW + (128,))

    def draw_conflict(self, screen, conflict: tuple):
        #! We add border to the arrow to denote it conflict, and change color
        color = WHITE + (0,)
        border_color = YELLOW + (255,)

        def get_xy(pos):
            return (
                pos[1] * self.TILE_SIZE + self.UP_PAD,
                pos[0] * self.TILE_SIZE + self.LEFT_PAD,
            )

        if len(conflict) == 1:
            draw_arrow(
                screen,
                color,
                get_xy(conflict[0].pos_s),
                get_xy(conflict[0].pos_e),
                border_color=border_color,
            )
        else:
            draw_arrow(
                screen,
                color,
                get_xy(conflict[0].pos_s),
                get_xy(conflict[0].pos_e),
                border_color=border_color,
            )
            draw_arrow(
                screen,
                color,
                get_xy(conflict[1].pos_s),
                get_xy(conflict[1].pos_e),
                border_color=border_color,
            )

    def export_to_png(self, screen, out_path):
        pygame.image.save(screen, out_path)


def draw_arrow(
    screen,
    color,
    start,
    end,
    # arrow_width=5,
    arrow_width=8,
    arrow_head_length=15,
    arrow_head_angle=45,
    border_color=None,
    border_width=2,
):
    """
    Draw an arrow from start to end on the given surface, optionally with a border.

    Parameters:
      screen: The pygame surface to draw on.
      color: The color of the arrow.
      start: (x, y) tuple for the arrow’s starting position.
      end: (x, y) tuple for the arrow’s ending position.
      arrow_width: Width of the arrow shaft.
      arrow_head_length: Length of the arrow head lines.
      arrow_head_angle: Angle (in degrees) between the arrow shaft and each side of the head.
      border_color: The color of the arrow's border. If None, no border is drawn.
      border_width: Width of the border.
    """
    surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    surface.fill(WHITE + (0,))

    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    rad = math.radians(arrow_head_angle)  # Convert arrow head angle to radians

    # Calculate arrowhead points
    left_point = (
        end[0] - arrow_head_length * math.cos(angle - rad),
        end[1] - arrow_head_length * math.sin(angle - rad),
    )
    right_point = (
        end[0] - arrow_head_length * math.cos(angle + rad),
        end[1] - arrow_head_length * math.sin(angle + rad),
    )

    # Draw border if specified
    if border_color:
        pygame.draw.line(
            surface, border_color, start, end, arrow_width + 2 * border_width
        )  # Draw wider line for border
        pygame.draw.polygon(
            surface, border_color, [end, left_point, right_point], border_width
        )  # Border around arrowhead

    # Draw main arrow
    pygame.draw.line(surface, color, start, end, arrow_width)
    pygame.draw.polygon(surface, color, [end, left_point, right_point])

    screen.blit(surface, (0, 0))
