"""Shared command-line configuration for V28."""

from dataclasses import fields

from .training import TrainingConfig


def add_config_arguments(parser, defaults=None):
    defaults = defaults or TrainingConfig()
    for field in fields(TrainingConfig):
        name = field.name
        option = "--" + name.replace("_", "-")
        default = getattr(defaults, name)
        if name == "protein_input_width":
            parser.add_argument(option, type=int)
        elif name == "amp_dtype":
            parser.add_argument(option, choices=("bfloat16", "float16"), default=default)
        else:
            parser.add_argument(option, type=type(default), default=default)
    return parser


def config_from_args(args, detected_protein_width):
    values = {field.name: getattr(args, field.name) for field in fields(TrainingConfig)}
    values["protein_input_width"] = int(args.protein_input_width or detected_protein_width)
    return TrainingConfig(**values)


__all__ = ["add_config_arguments", "config_from_args"]
