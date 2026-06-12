from .carla_wpt_fixed_env import CarlaWptFixedEnv
from .toolkit import get_vehicle_pos
import carla
import math

class CarlaLeftTurnEnv(CarlaWptFixedEnv):
    """
    Vehicle passes the crossing (turn left) and avoid collision.

    **Provided Tasks**: ``carla_left_turn_simple``, ``carla_left_turn_medium``, ``carla_left_turn_hard``
    """

    def on_step(self) -> None:
        if len(self.actor_flow) > 0:
            vehicle = self.actor_flow[0]
            x, y = get_vehicle_pos(self.actor_flow[0])
            if y > -99.4 or y < -171.4 or x > 57.6:
                self._world.destroy_actor(vehicle.id)
                self.actor_flow.popleft()
        self._update_spectator_follow()
    
        super().on_step()

    def _update_spectator_follow(self):
        spectator = self._world._world.get_spectator()  # carla.World
        ego_tf = self.get_ego_vehicle().get_transform()
        loc = ego_tf.location
        rot = ego_tf.rotation

        yaw = math.radians(rot.yaw)

        back_dist = 8.0
        height = 3.0

        cam_loc = carla.Location(
            x=loc.x - back_dist * math.cos(yaw),
            y=loc.y - back_dist * math.sin(yaw),
            z=loc.z + height,
        )

        cam_rot = carla.Rotation(
            pitch=-15.0,
            yaw=rot.yaw,
            roll=0.0,
        )

        spectator.set_transform(carla.Transform(cam_loc, cam_rot))