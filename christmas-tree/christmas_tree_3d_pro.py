"""A cinematic interactive 3D particle Christmas tree.

Stack: NumPy + Pygame

Install:
    python -m pip install -r requirements_christmas_tree_3d_pro.txt

Run:
    python christmas_tree_3d_pro.py

Controls:
    Hold LEFT mouse     Attract nearby particles toward the cursor
    SPACE               Disassemble / reassemble the tree
    RIGHT mouse         Launch a firework toward the cursor
    F                   Launch a firework at a random position
    D                   Toggle depth of field
    M                   Toggle mouse attraction mode (default ON while held)
    A                   Toggle automatic camera orbit
    LEFT / RIGHT        Rotate camera
    UP / DOWN           Change orbit speed
    S                   Toggle snowfall
    L                   Toggle tree lights
    1 / 2 / 3            Change color palette
    R                   Reset composition
    ESC                 Quit

Notes:
- The scene is genuinely 3D: particles retain XYZ world coordinates.
- The camera applies yaw and pitch before perspective projection.
- Depth of field is an inexpensive particle/bokeh approximation suitable for
  real-time Pygame rendering, not a physically based camera simulation.
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pygame


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WIDTH, HEIGHT = 1120, 820
FPS = 60
CENTER = np.array([0.0, 3.10, 0.0], dtype=np.float32)
TREE_HEIGHT = 6.15
TREE_RADIUS = 2.55
CAMERA_DISTANCE = 13.0
FOCAL_LENGTH = 1280.0
FOCUS_DISTANCE = 13.0
TREE_PARTICLE_COUNT = 300000  # very dense, hairline foliage point cloud

BG_TOP = (3, 8, 22)
BG_BOTTOM = (8, 25, 39)

PALETTES = {
    1: {
        "name": "EVERGREEN / CHAMPAGNE",
        "foliage": [(5, 47, 35), (8, 72, 48), (13, 97, 59), (24, 119, 72), (40, 139, 86)],
        "bulbs": [(255, 86, 104), (255, 207, 112), (96, 218, 255), (255, 248, 222)],
        "baubles": [(190, 33, 51), (235, 172, 73), (42, 132, 174), (231, 225, 203)],
        "garland": (255, 197, 111),
        "firework": [(255, 203, 118), (255, 109, 127), (117, 222, 255)],
    },
    2: {
        "name": "FROST / GLACIER",
        "foliage": [(7, 38, 58), (10, 65, 87), (17, 98, 125), (36, 133, 158), (76, 174, 190)],
        "bulbs": [(133, 229, 255), (255, 255, 255), (119, 177, 255), (255, 222, 165)],
        "baubles": [(156, 221, 245), (224, 246, 255), (91, 159, 224), (246, 233, 206)],
        "garland": (155, 226, 255),
        "firework": [(153, 228, 255), (255, 255, 255), (157, 183, 255)],
    },
    3: {
        "name": "RUBY / ROSE GOLD",
        "foliage": [(42, 17, 30), (73, 23, 40), (102, 31, 49), (137, 43, 59), (164, 59, 72)],
        "bulbs": [(255, 187, 113), (255, 91, 111), (255, 227, 174), (255, 245, 219)],
        "baubles": [(255, 188, 112), (191, 43, 62), (241, 218, 177), (121, 30, 47)],
        "garland": (255, 207, 143),
        "firework": [(255, 200, 135), (255, 106, 131), (255, 238, 193)],
    },
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def smoothstep(value: float) -> float:
    value = clamp01(value)
    return value * value * (3.0 - 2.0 * value)


def palette_array(count: int, colors: list[tuple[int, int, int]],
                  rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(colors), count)
    out = np.asarray(colors, dtype=np.float32)[idx]
    variation = rng.uniform(0.84, 1.13, (count, 1))
    return np.clip(out * variation, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# 3D geometry and camera
# -----------------------------------------------------------------------------
def make_tree_particles(count: int, rng: np.random.Generator) -> np.ndarray:
    """Volumetric fir-cone cloud with slightly layered branch silhouettes."""
    t = rng.uniform(0.008, 0.995, count)
    y = 0.12 + t * TREE_HEIGHT
    envelope = TREE_RADIUS * np.power(1.0 - t, 0.76)
    branch_phase = (t * 8.0) % 1.0
    scallop = 0.80 + 0.20 * np.power(np.sin(np.pi * branch_phase), 0.65)
    radius = envelope * scallop * np.sqrt(rng.random(count))
    angle = rng.uniform(0.0, math.tau, count)
    # A little extra outward foliage around branch tips creates a natural edge.
    radius *= rng.uniform(0.90, 1.08, count)
    x = np.cos(angle) * radius
    z = np.sin(angle) * radius
    return np.column_stack((x, y, z)).astype(np.float32)


def make_helix(turns: float, count: int, radius_factor: float = 0.93,
               phase: float = 0.35) -> np.ndarray:
    t = np.linspace(0.025, 0.975, count, dtype=np.float32)
    y = 0.16 + t * TREE_HEIGHT
    r = TREE_RADIUS * np.power(1.0 - t, 0.76) * radius_factor
    angle = t * math.tau * turns + phase
    return np.column_stack((np.cos(angle) * r, y, np.sin(angle) * r)).astype(np.float32)


def make_decorations(count: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(0.13, 0.91, count)
    a = rng.uniform(0, math.tau, count)
    r = TREE_RADIUS * np.power(1.0 - t, 0.76) * rng.uniform(0.86, 1.03, count)
    return np.column_stack((np.cos(a) * r, 0.15 + t * TREE_HEIGHT,
                            np.sin(a) * r)).astype(np.float32)


def make_burst_targets(points: np.ndarray, rng: np.random.Generator,
                       low: float = 2.1, high: float = 5.8) -> np.ndarray:
    center_relative = points - CENTER
    direction = center_relative.copy()
    direction[:, 1] *= 0.84
    direction += rng.normal(0.0, 0.85, direction.shape).astype(np.float32)
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-5)
    direction /= norm
    distance = rng.uniform(low, high, (len(points), 1)).astype(np.float32)
    noise = rng.normal(0, 0.32, points.shape).astype(np.float32)
    return points + direction * distance + noise


def camera_transform(points: np.ndarray, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Transform world XYZ into camera-oriented XYZ; returns view points + depth."""
    relative = points - CENTER
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    x1 = relative[:, 0] * cy + relative[:, 2] * sy
    z1 = -relative[:, 0] * sy + relative[:, 2] * cy
    y2 = relative[:, 1] * cp - z1 * sp
    z2 = relative[:, 1] * sp + z1 * cp

    view = np.column_stack((x1, y2, z2)).astype(np.float32)
    depth = CAMERA_DISTANCE - z2
    return view, depth


def project(points: np.ndarray, yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view, depth = camera_transform(points, yaw, pitch)
    safe_depth = np.maximum(2.0, depth)
    scale = FOCAL_LENGTH / safe_depth
    sx = WIDTH * 0.5 + view[:, 0] * scale
    sy = HEIGHT * 0.52 - view[:, 1] * scale
    return np.column_stack((sx, sy)), depth, scale


def unproject_mouse(mouse_xy: tuple[int, int], yaw: float, pitch: float) -> np.ndarray:
    """Map cursor to a 3D point on the camera-facing plane through scene center."""
    mx, my = mouse_xy
    vx = (mx - WIDTH * 0.5) * CAMERA_DISTANCE / FOCAL_LENGTH
    vy = -(my - HEIGHT * 0.52) * CAMERA_DISTANCE / FOCAL_LENGTH
    # Inverse pitch, then inverse yaw.
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    y1 = vy * cp
    z1 = -vy * sp
    wx = vx * cy - z1 * sy
    wz = vx * sy + z1 * cy
    return CENTER + np.array([wx, y1, wz], dtype=np.float32)


def make_background(rng: np.random.Generator) -> pygame.Surface:
    bg = pygame.Surface((WIDTH, HEIGHT))
    for y in range(HEIGHT):
        t = y / max(1, HEIGHT - 1)
        c = tuple(int(BG_TOP[i] * (1 - t) + BG_BOTTOM[i] * t) for i in range(3))
        pygame.draw.line(bg, c, (0, y), (WIDTH, y))

    # Deep-field stars.
    for _ in range(210):
        x = int(rng.integers(0, WIDTH))
        y = int(rng.integers(0, HEIGHT))
        r = int(rng.choice([1, 1, 1, 2]))
        col = (110, 165, 205) if r == 1 else (170, 207, 229)
        pygame.draw.circle(bg, col, (x, y), r)

    # Large soft bokeh in the distant background.
    bokeh = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    for _ in range(28):
        x = int(rng.integers(25, WIDTH - 25))
        y = int(rng.integers(35, HEIGHT - 55))
        r = int(rng.integers(7, 24))
        col = rng.choice([(105, 164, 194, 8), (225, 190, 125, 9), (120, 190, 170, 8)])
        pygame.draw.circle(bokeh, col, (x, y), r)
    bg.blit(bokeh, (0, 0))
    return bg.convert()


def draw_star(surface: pygame.Surface, x: float, y: float, outer: float,
              color: tuple[int, int, int], rotation: float = 0.0) -> None:
    pts = []
    for i in range(10):
        a = -math.pi / 2 + rotation + i * math.pi / 5
        r = outer if i % 2 == 0 else outer * 0.43
        pts.append((int(x + math.cos(a) * r), int(y + math.sin(a) * r)))
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (255, 247, 220), pts, 2)


def color_at_depth(color: np.ndarray, blur: float) -> tuple[int, int, int]:
    """Slightly desaturate and cool out-of-focus particles."""
    t = min(0.46, blur * 0.075)
    base = color.astype(np.float32)
    cool = np.array([125.0, 166.0, 190.0], dtype=np.float32)
    mixed = base * (1.0 - t) + cool * t
    return tuple(int(v) for v in np.clip(mixed, 0, 255))


@dataclass
class Firework:
    pos: np.ndarray
    vel: np.ndarray
    color: tuple[int, int, int]
    target_y: float
    age: float = 0.0
    exploded: bool = False
    particles: Optional[np.ndarray] = None
    particle_vel: Optional[np.ndarray] = None
    life: Optional[np.ndarray] = None
    max_life: Optional[np.ndarray] = None
    trail: Optional[list[np.ndarray]] = None

    @classmethod
    def launch(cls, target: np.ndarray, color: tuple[int, int, int]) -> "Firework":
        rng = np.random.default_rng()
        start = np.array([target[0] * 0.36, -0.32, target[2] * 0.30], dtype=np.float32)
        target_y = float(np.clip(target[1], 2.0, 6.7))
        delta = target - start
        delta[1] = max(2.8, target_y - start[1])
        velocity = delta / max(0.7, float(np.linalg.norm(delta)) / 5.0)
        velocity += np.array([rng.uniform(-0.35, 0.35), 0.6, rng.uniform(-0.35, 0.35)], dtype=np.float32)
        return cls(start, velocity, color, target_y, trail=[])

    def explode(self, rng: np.random.Generator) -> None:
        count = int(rng.integers(95, 155))
        u = rng.uniform(-1.0, 1.0, count)
        theta = rng.uniform(0.0, math.tau, count)
        radius = np.sqrt(1.0 - u * u)
        directions = np.column_stack((radius * np.cos(theta), u, radius * np.sin(theta))).astype(np.float32)
        speeds = rng.uniform(1.1, 4.2, (count, 1)).astype(np.float32)
        self.particles = np.repeat(self.pos[None, :], count, axis=0)
        self.particle_vel = directions * speeds
        self.life = rng.uniform(0.75, 1.65, count).astype(np.float32)
        self.max_life = self.life.copy()
        self.exploded = True

    def update(self, dt: float, rng: np.random.Generator) -> bool:
        self.age += dt
        if not self.exploded:
            self.vel[1] -= 0.75 * dt
            self.pos += self.vel * dt
            assert self.trail is not None
            self.trail.append(self.pos.copy())
            if len(self.trail) > 9:
                self.trail.pop(0)
            if self.pos[1] >= self.target_y or self.vel[1] <= 1.1 or self.age > 2.4:
                self.explode(rng)
            return True

        assert self.particles is not None and self.particle_vel is not None
        assert self.life is not None
        self.particle_vel[:, 1] -= 2.15 * dt
        self.particle_vel *= (1.0 - 0.28 * dt)
        self.particles += self.particle_vel * dt
        self.life -= dt
        return bool(np.any(self.life > 0))


def make_gift(surface: pygame.Surface, x: int, y: int, w: int, h: int,
              box: tuple[int, int, int], ribbon: tuple[int, int, int]) -> None:
    pygame.draw.ellipse(surface, (1, 5, 12), (x - 8, y + h - 2, w + 19, 18))
    pygame.draw.rect(surface, box, (x, y, w, h), border_radius=5)
    lighter = tuple(min(255, c + 25) for c in box)
    pygame.draw.rect(surface, lighter, (x + 4, y + 4, w - 8, 8), border_radius=3)
    pygame.draw.rect(surface, ribbon, (x + int(w * 0.43), y, max(5, int(w * 0.14)), h))
    pygame.draw.rect(surface, ribbon, (x, y + int(h * 0.28), w, max(5, int(h * 0.12))))
    pygame.draw.ellipse(surface, ribbon, (x + int(w * .18), y - 13, int(w * .34), 20))
    pygame.draw.ellipse(surface, ribbon, (x + int(w * .48), y - 13, int(w * .34), 20))
    pygame.draw.circle(surface, (255, 243, 207), (x + w // 2, y - 4), 5)


# -----------------------------------------------------------------------------
# Main real-time scene
# -----------------------------------------------------------------------------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("CHRISTMAS / PARTICLE ATELIER — NumPy × Pygame")
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont("georgia", 27)
    sub_font = pygame.font.SysFont("arial", 13)
    ui_font = pygame.font.SysFont("arial", 13)
    tiny_font = pygame.font.SysFont("arial", 11)

    rng = np.random.default_rng(2026)
    background = make_background(rng)
    # Native-resolution alpha layer for restrained depth-of-field bokeh.
    dof_layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    # A colorkey point layer lets NumPy stamp thousands of one-pixel foliage
    # particles in one vectorized operation instead of drawing large circles.
    foliage_layer = pygame.Surface((WIDTH, HEIGHT))
    foliage_layer.set_colorkey((0, 0, 0))

    # Primary foliage particles: each has a 3D home position, a burst destination,
    # and velocity for spring-based deconstruction/reassembly.
    leaf_home = make_tree_particles(TREE_PARTICLE_COUNT, rng)
    leaf_burst = make_burst_targets(leaf_home, rng)
    leaf_pos = leaf_home.copy()
    leaf_vel = np.zeros_like(leaf_pos)
    leaf_colors = palette_array(len(leaf_home), PALETTES[1]["foliage"], rng)
    # Only a small fraction scintillates; the rest form a dense, fine-grained
    # evergreen silhouette rather than a cloud of chunky dots.
    leaf_twinkle = rng.random(len(leaf_home)) < 0.014
    leaf_phase = rng.uniform(0, math.tau, len(leaf_home)).astype(np.float32)

    # Decorative 3D elements.
    garland_home = make_helix(8.4, 690, 0.92, 0.22)
    garland_burst = make_burst_targets(garland_home, rng, 2.0, 4.8)
    garland_pos = garland_home.copy()
    bauble_home = make_decorations(43, rng)
    bauble_burst = make_burst_targets(bauble_home, rng, 1.9, 4.4)
    bauble_pos = bauble_home.copy()
    bauble_colors = palette_array(len(bauble_home), PALETTES[1]["baubles"], rng)
    bauble_sizes = rng.integers(5, 10, len(bauble_home))
    bulb_home = make_helix(13.0, 155, 0.95, 1.30)
    bulb_burst = make_burst_targets(bulb_home, rng, 1.7, 4.1)
    bulb_pos = bulb_home.copy()
    bulb_colors = palette_array(len(bulb_home), PALETTES[1]["bulbs"], rng)
    bulb_phase = rng.uniform(0, math.tau, len(bulb_home))

    # Snow is also stored in 3D world coordinates.
    snow_count = 235
    snow_pos = np.column_stack((
        rng.uniform(-6.6, 6.6, snow_count),
        rng.uniform(-0.8, 8.4, snow_count),
        rng.uniform(-5.5, 5.5, snow_count),
    )).astype(np.float32)
    snow_speed = rng.uniform(0.7, 2.2, snow_count).astype(np.float32)
    snow_phase = rng.uniform(0, math.tau, snow_count).astype(np.float32)
    snow_size = rng.uniform(1.0, 3.1, snow_count)

    fireworks: list[Firework] = []
    firework_timer = 1.5
    palette_id = 1
    palette = PALETTES[palette_id]

    # State and camera controls.
    yaw = 0.0
    pitch = -0.055
    orbit_speed = 0.20
    auto_orbit = True
    assembled = True
    disassembly = 0.0
    magnet_enabled = True
    dof_enabled = True
    focus_distance = FOCUS_DISTANCE
    snow_enabled = True
    lights_enabled = True
    elapsed = 0.0
    running = True

    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.033)
        elapsed += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    assembled = not assembled
                elif event.key == pygame.K_a:
                    auto_orbit = not auto_orbit
                elif event.key == pygame.K_d:
                    dof_enabled = not dof_enabled
                elif event.key == pygame.K_LEFTBRACKET:
                    focus_distance = max(8.0, focus_distance - 0.45)
                elif event.key == pygame.K_RIGHTBRACKET:
                    focus_distance = min(18.0, focus_distance + 0.45)
                elif event.key == pygame.K_m:
                    magnet_enabled = not magnet_enabled
                elif event.key == pygame.K_s:
                    snow_enabled = not snow_enabled
                elif event.key == pygame.K_l:
                    lights_enabled = not lights_enabled
                elif event.key == pygame.K_f:
                    target = np.array([rng.uniform(-2.9, 2.9), rng.uniform(3.0, 7.0),
                                       rng.uniform(-1.8, 1.8)], dtype=np.float32)
                    fireworks.append(Firework.launch(target, random.choice(palette["firework"])))
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                    palette_id = int(event.unicode)
                    palette = PALETTES[palette_id]
                    leaf_colors = palette_array(len(leaf_home), palette["foliage"], rng)
                    bauble_colors = palette_array(len(bauble_home), palette["baubles"], rng)
                    bulb_colors = palette_array(len(bulb_home), palette["bulbs"], rng)
                elif event.key == pygame.K_r:
                    assembled = True
                    disassembly = 0.0
                    leaf_pos[:] = leaf_home
                    leaf_vel.fill(0)
                    yaw, pitch = 0.0, -0.055
                    auto_orbit = True
                    fireworks.clear()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    focus_distance = max(8.0, focus_distance - 0.45)
                elif event.button == 5:
                    focus_distance = min(18.0, focus_distance + 0.45)
                elif event.button == 3:
                    target = unproject_mouse(event.pos, yaw, pitch)
                    target[1] = np.clip(target[1], 2.0, 6.8)
                    fireworks.append(Firework.launch(target, random.choice(palette["firework"])))

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            yaw -= 0.95 * dt
        if keys[pygame.K_RIGHT]:
            yaw += 0.95 * dt
        if keys[pygame.K_UP]:
            orbit_speed = min(1.0, orbit_speed + 0.42 * dt)
        if keys[pygame.K_DOWN]:
            orbit_speed = max(0.0, orbit_speed - 0.42 * dt)
        if auto_orbit:
            yaw += orbit_speed * dt

        # Spring particles toward their assembled or nebula destinations.
        target_mix = 0.0 if assembled else 1.0
        response = 1.0 - math.exp(-2.8 * dt)
        disassembly += (target_mix - disassembly) * response
        eased = smoothstep(disassembly)
        shape_target = leaf_home + (leaf_burst - leaf_home) * eased
        acceleration = (shape_target - leaf_pos) * 12.0 - leaf_vel * 4.7

        # Cursor acts as a 3D magnetic field. Press and hold LEFT mouse.
        mouse_down = pygame.mouse.get_pressed(3)[0]
        if magnet_enabled and mouse_down:
            pointer_world = unproject_mouse(pygame.mouse.get_pos(), yaw, pitch)
            delta = pointer_world[None, :] - leaf_pos
            dist2 = np.sum(delta * delta, axis=1)
            radius = 3.25
            influence = np.exp(-dist2 / (2.0 * radius * radius)).astype(np.float32)
            acceleration += delta * (influence[:, None] * 8.5)

        leaf_vel += acceleration * dt
        leaf_vel *= math.exp(-2.25 * dt)
        leaf_pos += leaf_vel * dt

        # Decorations follow the same tree / nebula transition, with a soft
        # additional cursor displacement so the entire object responds together.
        garland_pos = garland_home + (garland_burst - garland_home) * eased
        bauble_pos = bauble_home + (bauble_burst - bauble_home) * eased
        bulb_pos = bulb_home + (bulb_burst - bulb_home) * eased
        if magnet_enabled and mouse_down:
            pointer = unproject_mouse(pygame.mouse.get_pos(), yaw, pitch)
            for array in (garland_pos, bauble_pos, bulb_pos):
                delta = pointer[None, :] - array
                d2 = np.sum(delta * delta, axis=1)
                inf = np.exp(-d2 / 18.0)[:, None]
                array += delta * inf * 0.19

        # Update 3D snow.
        if snow_enabled:
            snow_pos[:, 1] -= snow_speed * dt
            snow_pos[:, 0] += (np.sin(elapsed * 0.8 + snow_phase) * 0.45 + 0.10) * dt
            reset = snow_pos[:, 1] < -1.25
            snow_pos[reset, 1] = 8.3
            snow_pos[reset, 0] = rng.uniform(-6.6, 6.6, int(reset.sum()))
            snow_pos[reset, 2] = rng.uniform(-5.5, 5.5, int(reset.sum()))

        # Fireworks: rockets, trails, spherical bursts, gravity and fading.
        firework_timer -= dt
        if firework_timer <= 0 and len(fireworks) < 5:
            target = np.array([rng.uniform(-3.3, 3.3), rng.uniform(3.3, 6.8),
                               rng.uniform(-2.4, 2.4)], dtype=np.float32)
            fireworks.append(Firework.launch(target, random.choice(palette["firework"])))
            firework_timer = float(rng.uniform(2.4, 4.0))
        fireworks = [fw for fw in fireworks if fw.update(dt, rng)]

        # ------------------------------------------------------------------
        # Render background and depth layers.
        # ------------------------------------------------------------------
        screen.blit(background, (0, 0))
        dof_layer.fill((0, 0, 0, 0))

        # Low horizon floor / reflection pool.
        pygame.draw.ellipse(screen, (9, 25, 39), (WIDTH // 2 - 330, 725, 660, 55))
        pygame.draw.ellipse(screen, (20, 47, 60), (WIDTH // 2 - 275, 739, 550, 28), 1)

        # Dense foliage point cloud. Each leaf is now a single screen pixel;
        # NumPy writes the complete cloud into a transparent-color surface in
        # one batch, so increasing the count does not mean 30,000 draw calls.
        leaf_screen, leaf_depth, _ = project(leaf_pos, yaw, pitch)
        leaf_x = np.rint(leaf_screen[:, 0]).astype(np.int32)
        leaf_y = np.rint(leaf_screen[:, 1]).astype(np.int32)
        leaf_visible = (
            (leaf_x >= 0) & (leaf_x < WIDTH) &
            (leaf_y >= 55) & (leaf_y < HEIGHT)
        )
        visible_idx = np.flatnonzero(leaf_visible)
        foliage_layer.fill((0, 0, 0))

        if visible_idx.size:
            visible_depth = leaf_depth[visible_idx]
            if dof_enabled:
                blur_all = np.maximum(0.0, np.abs(leaf_depth - focus_distance) - 0.45)
            else:
                blur_all = np.zeros_like(leaf_depth)

            # Depth-aware color falloff gives the tiny points a little distance
            # separation without inflating their physical screen-space size.
            blur = blur_all[visible_idx]
            mix = np.minimum(0.46, blur * 0.075)[:, None]
            base = leaf_colors[visible_idx].astype(np.float32)
            cool = np.array([125.0, 166.0, 190.0], dtype=np.float32)
            rendered = base * (1.0 - mix) + cool[None, :] * mix
            rendered *= (1.0 / (1.0 + blur[:, None] * 0.10))

            twinkle_now = leaf_twinkle[visible_idx] & (
                np.sin(elapsed * 2.2 + leaf_phase[visible_idx]) > 0.60
            )
            rendered[twinkle_now] = np.clip(
                rendered[twinkle_now] * 1.42 + 20.0, 0, 255
            )
            rendered = np.clip(rendered, 2, 255).astype(np.uint8)

            # Draw far points first, then let nearer pixels overwrite them.
            depth_order = np.argsort(visible_depth)[::-1]
            px = leaf_x[visible_idx][depth_order]
            py = leaf_y[visible_idx][depth_order]
            rgb = rendered[depth_order]
            pixel_view = pygame.surfarray.pixels3d(foliage_layer)
            pixel_view[px, py] = rgb
            del pixel_view
            screen.blit(foliage_layer, (0, 0))

            # A small number of defocused points become subtle, native-resolution bokeh.
            # This is depth-of-field only: no low-resolution enlargement or glow pass.
            if dof_enabled:
                dof_idx = visible_idx[(blur_all[visible_idx] > 0.85) &
                                      ((visible_idx % 22) == 0)]
                for idx in dof_idx:
                    sx, sy = leaf_screen[idx]
                    blur_value = float(blur_all[idx])
                    radius = max(1, min(2, int(1 + blur_value * 0.035)))
                    col = tuple(int(v) for v in leaf_colors[idx])
                    alpha = min(30, int(blur_value * 5))
                    pygame.draw.circle(dof_layer, (*col, alpha),
                                       (int(sx), int(sy)), radius)

        # Subtle native-resolution depth-of-field marks; no upscaling or additive glow.
        if dof_enabled:
            screen.blit(dof_layer, (0, 0))

        # Garland: crisp 3D helix line without a glow layer.
        garland_screen, garland_depth, _ = project(garland_pos, yaw, pitch)
        garland_points = [(int(x), int(y)) for x, y in garland_screen
                          if -50 <= x <= WIDTH + 50 and 35 <= y <= HEIGHT]
        if eased < 0.96 and len(garland_points) > 2:
            pygame.draw.lines(screen, tuple(int(c * .68) for c in palette["garland"]),
                              False, garland_points, 2)
            pygame.draw.lines(screen, (255, 232, 186), False, garland_points, 1)

        # Baubles: small glossy ornaments, drawn without oversized halos.
        bauble_screen, bauble_depth, bauble_scale = project(bauble_pos, yaw, pitch)
        for i, (sx, sy) in enumerate(bauble_screen):
            if not (-15 <= sx <= WIDTH + 15 and 40 <= sy <= HEIGHT):
                continue
            radius = max(2, min(7, int(bauble_sizes[i] * bauble_scale[i] * 0.007)))
            col = tuple(int(v) for v in bauble_colors[i])
            pygame.draw.circle(screen, (17, 28, 38), (int(sx), int(sy)), radius + 2)
            pygame.draw.circle(screen, col, (int(sx), int(sy)), radius)
            pygame.draw.circle(screen, (255, 246, 225),
                               (int(sx - radius * .30), int(sy - radius * .34)), max(1, radius // 3))

        # Twinkling bulbs are rendered directly with crisp edges.
        if lights_enabled:
            bulb_screen, bulb_depth, bulb_scale = project(bulb_pos, yaw, pitch)
            for i, (sx, sy) in enumerate(bulb_screen):
                if not (-10 <= sx <= WIDTH + 10 and 35 <= sy <= HEIGHT):
                    continue
                pulse = 0.5 + 0.5 * math.sin(elapsed * 3.4 + i * 1.73)
                radius = max(1, min(3, int(0.8 + pulse * 1.5)))
                col = tuple(int(v) for v in bulb_colors[i])
                pygame.draw.circle(screen, col, (int(sx), int(sy)), radius)
                if radius >= 3:
                    pygame.draw.circle(screen, (255, 255, 250), (int(sx - 1), int(sy - 1)), 1)

        # Topper star and rays.
        topper = np.array([[0.0, TREE_HEIGHT + 0.48, 0.0]], dtype=np.float32)
        topper_screen, _, topper_scale = project(topper, yaw, pitch)
        tx, ty = topper_screen[0]
        star_radius = 24 * float(topper_scale[0]) * 0.012 * (1 + 0.10 * math.sin(elapsed * 2.4))
        for j in range(8):
            a = elapsed * 0.17 + j * math.tau / 8
            p1 = (int(tx + math.cos(a) * star_radius * 1.05),
                  int(ty + math.sin(a) * star_radius * 1.05))
            p2 = (int(tx + math.cos(a) * star_radius * (1.45 if j % 2 == 0 else 1.25)),
                  int(ty + math.sin(a) * star_radius * (1.45 if j % 2 == 0 else 1.25)))
            pygame.draw.line(screen, (255, 225, 158), p1, p2, 2)
        draw_star(screen, tx, ty, star_radius, (255, 210, 112), elapsed * .10)

        # Trunk and gifts anchor the floating particle tree.
        pygame.draw.rect(screen, (88, 52, 34), (WIDTH // 2 - 22, 664, 44, 78), border_radius=5)
        pygame.draw.line(screen, (151, 99, 62), (WIDTH // 2 - 8, 672), (WIDTH // 2 - 8, 735), 3)
        make_gift(screen, WIDTH // 2 - 330, 715, 82, 65, (151, 39, 57), (242, 200, 119))
        make_gift(screen, WIDTH // 2 - 235, 738, 64, 48, (37, 105, 143), (231, 224, 187))
        make_gift(screen, WIDTH // 2 + 158, 716, 95, 65, (133, 40, 63), (250, 212, 139))
        make_gift(screen, WIDTH // 2 + 262, 742, 62, 46, (39, 111, 80), (232, 207, 145))

        # Fireworks: draw rocket trails and crisp 3D particles.
        for fw in fireworks:
            if not fw.exploded:
                assert fw.trail is not None
                if len(fw.trail) > 1:
                    trail3d = np.asarray(fw.trail, dtype=np.float32)
                    trail_screen, _, _ = project(trail3d, yaw, pitch)
                    trail_points = [(int(x), int(y)) for x, y in trail_screen]
                    if len(trail_points) > 1:
                        pygame.draw.lines(screen, fw.color, False, trail_points, 2)
                rocket_screen, _, _ = project(fw.pos[None, :], yaw, pitch)
                rx, ry = rocket_screen[0]
                pygame.draw.circle(screen, (255, 246, 218), (int(rx), int(ry)), 3)
            else:
                assert fw.particles is not None and fw.life is not None
                fw_screen, fw_depth, fw_scale = project(fw.particles, yaw, pitch)
                for j, (sx, sy) in enumerate(fw_screen):
                    if fw.life[j] <= 0 or not (-8 <= sx < WIDTH + 8 and -8 <= sy < HEIGHT + 8):
                        continue
                    fade = clamp01(float(fw.life[j] / fw.max_life[j]))
                    r = max(1, int(2.2 * fw_scale[j] * 0.012))
                    col = tuple(int(c * (0.35 + 0.65 * fade)) for c in fw.color)
                    pygame.draw.circle(screen, col, (int(sx), int(sy)), r)

        # 3D snowfall projected through the same camera.
        if snow_enabled:
            snow_screen, snow_depth, snow_scale = project(snow_pos, yaw, pitch)
            for i, (sx, sy) in enumerate(snow_screen):
                if -5 <= sx < WIDTH + 5 and -5 <= sy < HEIGHT + 5:
                    radius = max(1, min(4, int(snow_size[i] * snow_scale[i] * 0.012)))
                    col = (205, 226, 241) if snow_depth[i] > CAMERA_DISTANCE else (243, 248, 255)
                    pygame.draw.circle(screen, col, (int(sx), int(sy)), radius)


        # Cursor feedback ring.
        mx, my = pygame.mouse.get_pos()
        if magnet_enabled:
            held = pygame.mouse.get_pressed(3)[0]
            ring_col = (255, 213, 137) if held else (115, 174, 193)
            pygame.draw.circle(screen, ring_col, (mx, my), 15 if held else 8, 1)
            if held:
                pygame.draw.circle(screen, (255, 228, 173), (mx, my), 3)

        # Editorial overlay and controls.
        title = title_font.render("CHRISTMAS / PARTICLE ATELIER", True, (239, 226, 198))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 22))
        subtitle = sub_font.render("A SMALL UNIVERSE OF LIGHT, MOTION & WINTER MAGIC", True,
                                   (131, 171, 190))
        screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 57))
        theme_label = tiny_font.render(palette["name"], True, (190, 204, 209))
        screen.blit(theme_label, (28, 26))
        mode = "ASSEMBLED TREE" if assembled else "DECONSTRUCTED NEBULA"
        mode_label = tiny_font.render(mode, True, (239, 197, 125))
        screen.blit(mode_label, (WIDTH - mode_label.get_width() - 28, 26))

        panel = pygame.Surface((WIDTH - 80, 46), pygame.SRCALPHA)
        panel.fill((2, 9, 20, 150))
        screen.blit(panel, (40, HEIGHT - 59))
        controls = "HOLD MOUSE  attract particles     SPACE  disassemble / assemble     RIGHT CLICK  firework     F  random firework"
        controls2 = "D  depth: " + ("ON" if dof_enabled else "OFF") + \
                    f"     FOCUS {focus_distance:.1f}  [ ] / wheel     A  orbit     S  snow     L  lights     1–3  palette     R  reset     ESC  quit"
        t1 = ui_font.render(controls, True, (206, 216, 218))
        t2 = tiny_font.render(controls2, True, (135, 165, 178))
        screen.blit(t1, (WIDTH // 2 - t1.get_width() // 2, HEIGHT - 49))
        screen.blit(t2, (WIDTH // 2 - t2.get_width() // 2, HEIGHT - 27))

        # Disassembly progress line.
        bx, by, bw = WIDTH // 2 - 74, 88, 148
        pygame.draw.rect(screen, (37, 58, 71), (bx, by, bw, 3), border_radius=2)
        pygame.draw.rect(screen, (237, 190, 118),
                         (bx, by, int(bw * (1.0 - eased)), 3), border_radius=2)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
