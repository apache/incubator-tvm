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

from typing import Any

import tvm_tirx
from tvm.relax.frontend import nn


def test_mutator_naming_basic():
    class Module0(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param0 = nn.Parameter((32, 128), "float64")

    class Module1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mod0 = Module0()
            self.param1 = nn.Parameter((32, 128), "float32")

    class Module2(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mod1 = Module1()
            self.param2 = nn.Parameter((32, 128), "float16")

    class Module3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mod2 = Module2()
            self.param3 = nn.Parameter((32, 128), "float8")

    class Mutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            if node.dtype == "float8":
                assert name == "mod3.param3"
                return node
            elif node.dtype == "float16":
                assert name == "mod3.mod2.param2"
                return node
            elif node.dtype == "float32":
                assert name == "mod3.mod2.mod1.param1"
                return node
            elif node.dtype == "float64":
                assert name == "mod3.mod2.mod1.mod0.param0"
                return node

    mod3 = Module3()
    mutator = Mutator()
    mutator.visit("mod3", mod3)


def test_mutator_naming_moduledict():
    class Module(nn.Module):
        def __init__(self, dtype) -> None:
            super().__init__()
            self.param = nn.Parameter((32, 128), dtype)

    class Mutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            if node.dtype == "float64":
                assert name == "mod_dict.k0.0.param"
                return node
            elif node.dtype == "float32":
                assert name == "mod_dict.k0.1.param"
                return node
            elif node.dtype == "float16":
                assert name == "mod_dict.k1.0.param"
                return node
            elif node.dtype == "float8":
                assert name == "mod_dict.k1.1.param"
                return node

    mod_dict = nn.ModuleDict(
        {
            "k0": nn.ModuleList([Module("float64"), Module("float32")]),
            "k1": nn.ModuleList([Module("float16"), Module("float8")]),
        }
    )
    mutator = Mutator()
    mutator.visit("mod_dict", mod_dict)


def test_mutator_naming_modulelist():
    class Module(nn.Module):
        def __init__(self, dtype) -> None:
            super().__init__()
            self.param = nn.Parameter((32, 128), dtype)

    class Mutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            if node.dtype == "float64":
                assert name == "mod_list.0.0.param"
                return node
            elif node.dtype == "float32":
                assert name == "mod_list.0.1.param"
                return node
            elif node.dtype == "float16":
                assert name == "mod_list.1.0.param"
                return node
            elif node.dtype == "float8":
                assert name == "mod_list.1.1.param"
                return node

    mod_list = nn.ModuleList(
        [
            nn.ModuleList([Module("float64"), Module("float32")]),
            nn.ModuleList([Module("float16"), Module("float8")]),
        ]
    )
    mutator = Mutator()
    mutator.visit("mod_list", mod_list)


def test_mutator_naming_parameter_containers():
    class Module(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param_list = nn.ParameterList(
                [
                    nn.Parameter((32, 128), "float64"),
                    nn.Parameter((32, 128), "float32"),
                ]
            )
            self.param_dict = nn.ParameterDict(
                {
                    "k0": nn.Parameter((32, 128), "float16"),
                    "k1": nn.Parameter((32, 128), "float8"),
                }
            )

    seen = []

    class Mutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            seen.append((name, node.dtype))
            return node

    module = Module()
    mutator = Mutator()
    mutator.visit("", module)

    assert seen == [
        ("param_list.0", "float64"),
        ("param_list.1", "float32"),
        ("param_dict.k0", "float16"),
        ("param_dict.k1", "float8"),
    ]


def test_mutator_module():
    class SubModule1(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class SubModule2(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Module(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mod = SubModule1()

    class Mutator(nn.Mutator):
        def visit_module(self, name: str, node: nn.Module) -> Any:
            if isinstance(node, SubModule1):
                return SubModule2()
            else:
                return node

    mutator = Mutator()
    module = Module()
    assert isinstance(module.mod, SubModule1)
    module = mutator.visit("", module)
    assert isinstance(module.mod, SubModule2)


def test_mutator_moduledict():
    class Module1(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Module2(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Module3(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Mutator(nn.Mutator):
        def visit_module(self, name: str, node: nn.Module) -> Any:
            if isinstance(node, Module3):
                return Module1()
            else:
                return node

    mutator = Mutator()
    module_dict = nn.ModuleDict({"k0": Module1(), "k1": Module2(), "k2": Module3()})
    assert isinstance(module_dict["k0"], Module1)
    assert isinstance(module_dict["k1"], Module2)
    assert isinstance(module_dict["k2"], Module3)
    module_dict = mutator.visit("", module_dict)
    assert isinstance(module_dict["k0"], Module1)
    assert isinstance(module_dict["k1"], Module2)
    assert isinstance(module_dict["k2"], Module1)


def test_mutator_modulelist():
    class Module1(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Module2(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Module3(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Mutator(nn.Mutator):
        def visit_module(self, name: str, node: nn.Module) -> Any:
            if isinstance(node, Module3):
                return Module1()
            else:
                return node

    mutator = Mutator()
    module_list = nn.ModuleList([Module1(), Module2(), Module3()])
    assert isinstance(module_list[0], Module1)
    assert isinstance(module_list[1], Module2)
    assert isinstance(module_list[2], Module3)
    module_list = mutator.visit("", module_list)
    assert isinstance(module_list[0], Module1)
    assert isinstance(module_list[1], Module2)
    assert isinstance(module_list[2], Module1)


def test_mutator_effect():
    class Effect1(nn.Effect):
        pass

    class Effect2(nn.Effect):
        pass

    class Module(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.effect = Effect1()

    class Mutator(nn.Mutator):
        def visit_effect(self, name: str, node: nn.Effect) -> Any:
            if isinstance(node, Effect1):
                return Effect2()

    mutator = Mutator()
    module = Module()
    assert isinstance(module.effect, Effect1)
    module = mutator.visit("", module)
    assert isinstance(module.effect, Effect2)


def test_mutator_param():
    class Module(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter((128, 64), "float16")

    class Mutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            if node.dtype == "float16":
                return nn.Parameter(node.shape, "float32")

    mutator = Mutator()
    module = Module()
    assert module.weight.dtype == "float16"
    module = mutator.visit("", module)
    assert module.weight.dtype == "float32"


def test_mutator_recursively():
    class SubModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter((128, 64), "float16")

    class Module(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mod = SubModule()

    class Mutator(nn.Mutator):
        def visit_param(self, name: str, node: nn.Parameter) -> Any:
            if node.dtype == "float16":
                return nn.Parameter(node.shape, "float32")

    mutator = Mutator()
    module = Module()
    assert module.mod.weight.dtype == "float16"
    module = mutator.visit("", module)
    assert module.mod.weight.dtype == "float32"


if __name__ == "__main__":
    tvm_tirx.testing.main()
