from .carla_wpt_env import CarlaWptEnv
from .toolkit import RandomPlanner


class CarlaNavigationEnv(CarlaWptEnv):
    """
    In this task, the ego vehicle needs to navigate through a set of random waypoints.

    **Provided Tasks**: ``carla_navigation``

    Available config parameters:

    * ``num_vehicles``: Number of vehicles to spawn in the environment

    """

    SUCCESS_DISTANCE_THRESHOLD = 200.0

    def on_reset(self) -> None:
        self.ego = self._world.spawn_actor()
        self._world.spawn_auto_actors(self._config.num_vehicles)
        self.ego_planner = RandomPlanner(vehicle=self.ego)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
        self.sum_travel_distance = self.planner_stats["travel_distance"]

    def on_step(self) -> None:
        super().on_step()
        self.sum_travel_distance += self.planner_stats["travel_distance"]

    def is_destination_reached(self):
        return self.sum_travel_distance >= self.SUCCESS_DISTANCE_THRESHOLD
