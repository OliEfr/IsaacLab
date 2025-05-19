from dataclasses import dataclass


@dataclass
class DefaultEvalConfig:
    eval_metric_subfolder: str = ""  # empty for not using a subfolder
    eval_metric_filename: str = "metrics.yaml"  # this expression will be EVALUATED, ie you can use python code here

    # run checks on env or agent cfg
    def run_checks(self, **kwargs):
        pass

# This is meant as an abstract base class. Others should inherit.
@dataclass
class TargetDistribution:
    eval_metric_filename: str = "f'x_{env_cfg.commands.base_velocity.ranges.lin_vel_x[0]}_y_{env_cfg.commands.base_velocity.ranges.lin_vel_y[0]}_heading_{env_cfg.commands.base_velocity.ranges.heading[0]}.yaml'"
    
    def run_checks(self, **kwargs):
        assert (
            kwargs["env_cfg"].commands.base_velocity.ranges.lin_vel_x[0]
            == kwargs["env_cfg"].commands.base_velocity.ranges.lin_vel_x[1]
        ), "Expected constant target speed."

        assert (
            kwargs["env_cfg"].commands.base_velocity.ranges.lin_vel_y[0]
            == kwargs["env_cfg"].commands.base_velocity.ranges.lin_vel_y[1]
        ), "Expected constant target speed for TargetSpeedDistributionEvaluation."

        assert (
            kwargs["env_cfg"].commands.base_velocity.ranges.heading[0]
            == kwargs["env_cfg"].commands.base_velocity.ranges.heading[1]
        ), "Expected constant target speed for TargetSpeedDistributionEvaluation."

        assert kwargs["args_cli"].x_speed is not None, (
            "You most likely want to set a fixed speed for evaluation."
        )
        assert kwargs["args_cli"].y_speed is not None, (
            "You most likely want to set a fixed speed for evaluation."
        )
        assert kwargs["args_cli"].heading is not None, (
            "You most likely want to set a fixed heading for evaluation."
        )
        
@dataclass
class TargetXYDistribution(TargetDistribution):
    eval_metric_subfolder: str = "TargetXYDistributionEvaluation"

    def run_checks(self, **kwargs):
        super().run_checks(**kwargs)
        
        assert kwargs["args_cli"].heading == 0, (
            "You most likely want to set heading to 0 for evaluation."
        )
        
@dataclass
class TargetXHeadingDistribution(TargetDistribution):
    eval_metric_subfolder: str = "TargetXHeadingDistributionEvaluation"

    def run_checks(self, **kwargs):
        super().run_checks(**kwargs)
        
        assert kwargs["args_cli"].y_speed == 0, (
            "You most likely want to set heading to 0 for evaluation."
        )
            
