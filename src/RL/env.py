import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulator import EmergencyPolicySimulator
from policy_actions import (
    apply_policy,
    build_policy_actions,
    load_ambulance_increment_effects,
    policy_reward_bonus,
)


class CheonanEmergencyEnv(gym.Env):

    def __init__(
        self,
        data_dir="data/processed/RL",
        policy_budget=3,
        refined_ambulance_actions=True,
    ):
        super().__init__()

        self.sim = EmergencyPolicySimulator(
            data_dir=data_dir
        )

        self.policy_budget = policy_budget

        # =========================
        # Action 후보 생성
        # =========================

        self.refined_ambulance_actions = refined_ambulance_actions
        self.actions = build_policy_actions(
            self.sim,
            refined_ambulance_actions=refined_ambulance_actions,
        )
        self.ambulance_increment_effects = load_ambulance_increment_effects(
            data_dir
        )

        self.action_space = spaces.Discrete(
            len(self.actions)
        )

        # =========================
        # State 구성
        # =========================

        self.state_cols = [
            "ems_duration_min",
            "hospital_duration_min",
            "alternative_score_01",
            "traffic_score_01",
            "vulnerable_score_01",
            "occurrence_score_01",

            "type_dispatch",
            "type_transfer",
            "type_alternative",
            "type_traffic",
            "type_demand",
        ]

        # 31개 지역 × 11개 변수
        base_state_size = (
            len(self.sim.state)
            * len(self.state_cols)
        )

        # Action 사용 여부
        action_mask_size = len(self.actions)

        # 남은 정책 예산 1개
        total_state_size = (
            base_state_size
            + action_mask_size
            + 1
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_state_size,),
            dtype=np.float32,
        )

        self.used_actions = None
        self.steps = 0
        self.cumulative_dispatch_improvement = 0.0


    # =========================================================
    # State 생성
    # =========================================================
    def _get_state(self):

        df = self.sim.state.copy()

        # 정책 적용 후 바뀐 이동시간을 0~1로 정규화
        ems_max = self.sim.base_state[
            "ems_duration_min"
        ].max()

        hospital_max = self.sim.base_state[
            "hospital_duration_min"
        ].max()

        df["ems_state"] = (
            df["ems_duration_min"] / ems_max
        ).clip(0, 1)

        df["hospital_state"] = (
            df["hospital_duration_min"] / hospital_max
        ).clip(0, 1)

        region_state = (
            df[
                [
                    "ems_state",
                    "hospital_state",
                    "alternative_score_01",
                    "traffic_score_01",
                    "vulnerable_score_01",
                    "occurrence_score_01",
                    "type_dispatch",
                    "type_transfer",
                    "type_alternative",
                    "type_traffic",
                    "type_demand",
                ]
            ]
            .astype(float)
            .to_numpy()
            .flatten()
        )

        used_action_state = (
            self.used_actions.astype(np.float32)
        )

        remaining_budget = (
            self.policy_budget
            - self.steps
        ) / self.policy_budget

        state = np.concatenate([
            region_state,
            used_action_state,
            np.array(
                [remaining_budget],
                dtype=np.float32
            )
        ])

        return state.astype(np.float32)


    # =========================================================
    # Reset
    # =========================================================
    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        self.sim.reset()

        self.used_actions = np.zeros(
            len(self.actions),
            dtype=np.float32
        )

        self.steps = 0
        self.cumulative_dispatch_improvement = 0.0

        return (
            self._get_state(),
            {}
        )


    # =========================================================
    # Step
    # =========================================================
    def step(
        self,
        action
    ):

        action = int(action)

        # 이미 사용한 정책이면 penalty
        if self.used_actions[action] == 1:

            reward = -1.0
            travel_time_reward = 0.0
            dispatch_reward = 0.0

        else:

            before = (
                self.sim
                .calculate_metrics()
            )

            policy = self.actions[
                action
            ]

            # =========================
            # 정책 적용
            # =========================

            apply_policy(self.sim, policy)

            after = (
                self.sim
                .calculate_metrics()
            )

            # =========================
            # Reward
            # =========================

            travel_time_reward = (
                before[
                    "weighted_total_time"
                ]
                -
                after[
                    "weighted_total_time"
                ]
            )
            dispatch_reward = policy_reward_bonus(
                policy,
                self.ambulance_increment_effects,
            )
            reward = travel_time_reward + dispatch_reward
            self.cumulative_dispatch_improvement += dispatch_reward

            self.used_actions[
                action
            ] = 1

        self.steps += 1

        terminated = (
            self.steps
            >= self.policy_budget
        )

        truncated = False

        metrics = self.sim.calculate_metrics()
        metrics["dispatch_response_improvement_min"] = (
            self.cumulative_dispatch_improvement
        )

        info = {
            "selected_action":
                self.actions[action],

            "steps":
                self.steps,

            "remaining_budget":
                self.policy_budget
                - self.steps,

            "reward_components": {
                "travel_time_reward": float(travel_time_reward),
                "dispatch_reward": float(dispatch_reward),
            },

            "metrics": metrics,
        }

        return (
            self._get_state(),
            float(reward),
            terminated,
            truncated,
            info
        )
