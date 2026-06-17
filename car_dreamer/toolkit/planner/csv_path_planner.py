import numpy as np
import pandas as pd
from pathlib import Path

from .base_planner import BasePlanner

class CsvPathPlanner(BasePlanner):
    """
    Read route directly from CSV and expose waypoints as (x, y, yaw).
    CSV must contain x,y columns. yaw is optional.
    """

    def __init__(
        self,
        vehicle,
        csv_path,
        x_col="x",
        y_col="y",
        yaw_col=None,
        max_waypoints=60,
        reach_threshold=0.8,
    ):
        super().__init__(vehicle, max_waypoints=max_waypoints, reach_threshold=reach_threshold)
        self._csv_path = csv_path
        self._x_col = x_col
        self._y_col = y_col
        self._yaw_col = yaw_col

        # 新增：保存完整全局路径，不会随着车辆前进被 pop 掉
        self._global_waypoints = []

    def get_global_waypoints(self):
        """
        Return full global path loaded from CSV.
        Each waypoint is (x, y, yaw).
        """
        return list(self._global_waypoints)

    def _compute_yaw_deg(self, xs, ys):
        n = len(xs)
        yaw = np.zeros(n, dtype=float)
        if n <= 1:
            return yaw

        dx = np.zeros(n, dtype=float)
        dy = np.zeros(n, dtype=float)

        dx[1:-1] = xs[2:] - xs[:-2]
        dy[1:-1] = ys[2:] - ys[:-2]

        dx[0] = xs[1] - xs[0]
        dy[0] = ys[1] - ys[0]

        dx[-1] = xs[-1] - xs[-2]
        dy[-1] = ys[-1] - ys[-2]

        yaw = np.degrees(np.arctan2(dy, dx))
        return yaw

    def init_route(self):
        csv_path = Path(self._csv_path)
        if not csv_path.is_absolute() and not csv_path.exists():
            repo_candidate = Path(__file__).resolve().parents[3] / csv_path
            if repo_candidate.exists():
                csv_path = repo_candidate
        df = pd.read_csv(csv_path)

        if self._x_col not in df.columns or self._y_col not in df.columns:
            raise ValueError(f"CSV must contain columns: {self._x_col}, {self._y_col}")

        xs = df[self._x_col].to_numpy(dtype=float)
        ys = df[self._y_col].to_numpy(dtype=float)

        if self._yaw_col is not None and self._yaw_col in df.columns:
            yaws = df[self._yaw_col].to_numpy(dtype=float)
        else:
            yaws = self._compute_yaw_deg(xs, ys)

        self._global_waypoints = [
            (float(x), float(y), float(yaw))
            for x, y, yaw in zip(xs, ys, yaws)
        ]

        for waypoint in self._global_waypoints:
            self.add_waypoint(waypoint)

    def extend_route(self):
        pass
