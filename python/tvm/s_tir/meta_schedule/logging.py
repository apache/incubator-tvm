# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Logging interface in MetaSchedule"""

import logging
import logging.config
import os
import os.path as osp
from collections.abc import Callable
from logging import Logger
from typing import Any


def get_logger(name: str) -> Logger:
    """Create or get a logger by its name. This is essentially a wrapper of python's native logger.

    Parameters
    ----------
    name : str
        The name of the logger.

    Returns
    -------
    logger : Logger
        The logger instance.
    """
    return logging.getLogger(name)


def get_logging_func(logger: Logger) -> Callable[[int, str, int, str], None] | None:
    """Get the logging function.

    Parameters
    ----------
    logger : Logger
        The logger instance.
    Returns
    -------
    result : Optional[Callable]
        The function to do the specified level of logging.
    """
    if logger is None:
        return None

    level2log = {
        logging.DEBUG: logger.debug,
        logging.INFO: logger.info,
        logging.WARNING: logger.warning,
        logging.ERROR: logger.error,
        # logging.FATAL not included
    }

    def logging_func(level: int, filename: str, lineo: int, msg: str):
        if level < 0:  # clear the output in notebook / console
            from IPython.display import (  # type: ignore # pylint: disable=import-outside-toplevel
                clear_output,
            )

            clear_output(wait=True)
        else:
            level2log[level](f"[{os.path.basename(filename)}:{lineo}] " + msg)

    return logging_func


def create_loggers(
    log_dir: str,
    params: list[dict[str, Any]],
    logger_config: dict[str, Any] | None = None,
    disable_existing_loggers: bool = False,
):
    """Create loggers from configuration"""
    if logger_config is None:
        config = {}
    else:
        config = logger_config

    config.setdefault("loggers", {})
    config.setdefault("handlers", {})
    config.setdefault("formatters", {})

    global_logger_name = "tvm.s_tir.meta_schedule"
    global_logger = logging.getLogger(global_logger_name)
    if global_logger.level is logging.NOTSET:
        global_logger.setLevel(logging.DEBUG)
    console_logging_level = logging._levelToName[  # pylint: disable=protected-access
        global_logger.level
    ]

    config["loggers"].setdefault(
        global_logger_name,
        {
            "level": logging.DEBUG,
            "handlers": [
                handler.get_name()
                for handler in global_logger.handlers
                if handler.get_name() is not None
            ]
            + [global_logger_name + ".console", global_logger_name + ".file"],
            "propagate": False,
        },
    )
    config["loggers"].setdefault(
        "{logger_name}",
        {
            "level": "DEBUG",
            "handlers": [
                "{logger_name}.file",
            ],
            "propagate": False,
        },
    )
    config["handlers"].setdefault(
        global_logger_name + ".console",
        {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "tvm.s_tir.meta_schedule.standard_formatter",
            "level": console_logging_level,
        },
    )
    config["handlers"].setdefault(
        global_logger_name + ".file",
        {
            "class": "logging.FileHandler",
            "filename": "{log_dir}/" + __name__ + ".task_scheduler.log",
            "mode": "a",
            "level": "DEBUG",
            "formatter": "tvm.s_tir.meta_schedule.standard_formatter",
        },
    )
    config["handlers"].setdefault(
        "{logger_name}.file",
        {
            "class": "logging.FileHandler",
            "filename": "{log_dir}/{logger_name}.log",
            "mode": "a",
            "level": "DEBUG",
            "formatter": "tvm.s_tir.meta_schedule.standard_formatter",
        },
    )
    config["formatters"].setdefault(
        "tvm.s_tir.meta_schedule.standard_formatter",
        {
            "format": "%(asctime)s [%(levelname)s] %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    )

    # set up dictConfig loggers
    p_config = {"version": 1, "disable_existing_loggers": disable_existing_loggers}
    for k, v in config.items():
        if k in ["formatters", "handlers", "loggers"]:
            p_config[k] = _batch_parameterize_config(v, params)  # type: ignore
        else:
            p_config[k] = v
    logging.config.dictConfig(p_config)

    # check global logger
    if global_logger.level not in [logging.DEBUG, logging.INFO]:
        global_logger.warning(
            "Logging level set to %s, please set to logging.INFO"
            " or logging.DEBUG to view full log.",
            logging._levelToName[global_logger.level],  # pylint: disable=protected-access
        )
    global_logger.info("Logging directory: %s", log_dir)


def _batch_parameterize_config(
    config: dict[str, Any],
    params: list[dict[str, str]],
) -> dict[str, Any]:
    """Parameterize the given configuration with multiple parameters sets.

    Parameters
    ----------
    config : Dict[str, Any]
        The given config dict.
    Params : List[Dict[str, str]]
        List of the given multiple parameters sets.

    Returns
    -------
    result : Dict[str, Any]
        The parameterized configuration.
    """
    results = {}
    for name, cfg in config.items():
        for p in params:
            p_name = name.format(**p)
            if p_name not in results:
                p_cfg = _parameterize_config(cfg, p)
                results[p_name] = p_cfg
    return results


def _parameterize_config(
    config: dict[str, Any],
    params: dict[str, str],
) -> dict[str, Any]:
    """Parameterize the given configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        The given config dict.
    Params : Dict[str, str]
        The given parameters.

    Returns
    -------
    result : Dict[str, Any]
        The parameterized configuration.
    """
    result = {}
    for k, v in config.items():
        if isinstance(k, str):
            k = k.format(**params)
        if isinstance(v, str):
            v = v.format(**params)
        elif isinstance(v, dict):
            v = _parameterize_config(v, params)
        elif isinstance(v, list):
            v = [t.format(**params) for t in v]
        result[k] = v
    return result


def get_loggers_from_work_dir(
    work_dir: str,
    task_names: list[str],
) -> list[Logger]:
    """Create loggers from work directory

    Parameters
    ----------
    work_dir : str
        The work directory.
    task_names : List[str]
        The list of task names.

    Returns
    -------
    loggers : List[Logger]
        The list of loggers.
    """
    log_dir = osp.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    pattern = __name__ + ".task_{i:0" + f"{len(str(len(task_names) - 1))}" + "d}_{name}"
    # Very long names may need be clipped to prevent os errors, we use the first 100 characters.
    loggers = [pattern.format(i=i, name=name[:100]) for i, name in enumerate(task_names)]
    create_loggers(
        log_dir=log_dir,
        params=[{"log_dir": log_dir, "logger_name": logger} for logger in loggers],
    )
    return [get_logger(logger) for logger in loggers]
