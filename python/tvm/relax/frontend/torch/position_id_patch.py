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

import torch
import torch.nn as nn
from torch.export import export as torch_export
from transformers import AutoModel

from tvm.relax.frontend.torch import from_exported_program


class StateDictWrapper(dict):
    """Wrap exported state_dict and inject extra keys (non-persistent buffers)."""

    def __init__(self, base_dict, extra):
        super().__init__(base_dict)
        self.extra = extra

    def __getitem__(self, key):
        if key in self.extra:
            return self.extra[key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.extra:
            return self.extra[key]
        return super().get(key, default)


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-multilingual-uncased")
        self.cls = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, x, mask=None):
        out = self.bert(x, attention_mask=mask).last_hidden_state[:, 0, :]
        return self.cls(out)


def main():
    torch.manual_seed(0)
    m = M().eval()

    x = torch.randint(0, 30522, (2, 16))
    mask = torch.ones_like(x)

    ep = torch_export(m, (x, mask))
    print("\n torch.export completed successfully\n")

    # --- Build extra buffers dict ---
    extra = {}
    for buf_name in m.bert.embeddings._non_persistent_buffers_set:
        tensor = m.bert.embeddings._buffers.get(buf_name)
        if tensor is not None:
            extra[f"bert.embeddings.{buf_name}"] = tensor
            print(f"Injecting buffer: bert.embeddings.{buf_name} -> shape {tensor.shape}")

    # Wrap exported state_dict
    sd_wrapped = StateDictWrapper(ep.state_dict, extra)

    # EP wrapper to override state_dict access
    class EPWrapper:
        def __init__(self, ep, sd_wrapped):
            self.__dict__["_ep"] = ep
            self.__dict__["_sd"] = sd_wrapped

        def __getattr__(self, name):
            if name == "state_dict":
                return self._sd
            return getattr(self._ep, name)

    ep_wrapped = EPWrapper(ep, sd_wrapped)

    # Import to TVM
    try:
        mod = from_exported_program(ep_wrapped)
        print("\n TVM import succeeded — all non-persistent buffers injected!\n")
    except Exception as e:
        print("\n TVM import failed with exception:")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
