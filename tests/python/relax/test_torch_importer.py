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

import tvm
from tvm import relax
import tvm.testing
from tvm.script.parser import relax as R


from tvm.relax.frontend.pytorch_fx import TorchFXTranslator

import torch
from torch.nn import Module


def verify_model(torch_model, input_info, binding, expected):
    mod = TorchFXTranslator().from_pytorch(torch_model, input_info)
    print(mod.script())
    binding = {k: tvm.nd.array(v) for k, v in binding.items()}
    expected = relax.transform.BindParams("main", binding)(expected)
    tvm.ir.assert_structural_equal(mod, expected)


def test_conv():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    class Conv2D1(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=True)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7), dtype="float32"),
            w2: R.Tensor((1, 6, 1, 1), dtype="float32"),
        ) -> R.Tensor((1, 6, 4, 4), dtype="float32"):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv3: R.Tensor((1, 6, 4, 4), dtype="float32") = R.add(lv1, w2)
                gv: R.Tensor((1, 6, 4, 4), dtype="float32") = lv3
                R.output(gv)
            return gv

    class Conv2D2(Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 6, 7, bias=False)

        def forward(self, input):
            return self.conv(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((6, 3, 7, 7), dtype="float32"),
        ) -> R.Tensor((1, 6, 4, 4), dtype="float32"):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 6, 4, 4), dtype="float32") = R.nn.conv2d(
                    input_1,
                    w1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                gv: R.Tensor((1, 6, 4, 4), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    model = Conv2D1()
    binding = {"w1": model.conv.weight.numpy(), "w2": model.conv.bias.numpy().reshape(1, 6, 1, 1)}
    verify_model(model, input_info, binding, expected1)

    model = Conv2D2()
    binding = {"w1": model.conv.weight.numpy()}
    verify_model(model, input_info, binding, expected2)


def test_linear():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    class Dense1(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=True)

        def forward(self, input):
            return self.linear(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((10, 7), dtype="float32"),
            w2: R.Tensor((1, 7), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 7), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 7), dtype="float32") = R.matmul(
                    input_1, w1, out_dtype="float32"
                )
                lv1: R.Tensor((1, 3, 10, 7), dtype="float32") = R.add(lv, w2)
                gv: R.Tensor((1, 3, 10, 7), dtype="float32") = lv1
                R.output(gv)
            return gv

    class Dense2(Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 7, bias=False)

        def forward(self, input):
            return self.linear(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((10, 7), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 7), dtype="float32"):
            # block 0
            with R.dataflow():
                lv1: R.Tensor((1, 3, 10, 7), dtype="float32") = R.matmul(
                    input_1, w1, out_dtype="float32"
                )
                gv: R.Tensor((1, 3, 10, 7), dtype="float32") = lv1
                R.output(gv)
            return gv

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    model = Dense1()
    binding = {"w1": model.linear.weight.numpy().T, "w2": model.linear.bias.numpy().reshape(1, -1)}
    verify_model(model, input_info, binding, expected1)

    model = Dense2()
    binding = {"w1": model.linear.weight.numpy().T}
    verify_model(model, input_info, binding, expected2)


def test_relu():
    torch.set_grad_enabled(False)

    class ReLU(Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, input):
            return self.relu(input)

    @tvm.script.ir_module
    class expected:
        @R.function
        def main(
            input_1: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((10, 10), dtype="float32") = R.nn.relu(input_1)
                gv: R.Tensor((10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    input_info = {"input_1": ([10, 10], "float32")}
    verify_model(ReLU(), input_info, {}, expected)


def test_maxpool2d():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class MaxPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[1, 1])

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[1, 1],
                    strides=[1, 1],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    class MaxPool2d2(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[2, 2], dilation=[2, 3])

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected2:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 4, 4), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 4, 4), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    dilation=[2, 3],
                    padding=[0, 0, 0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tensor((1, 3, 4, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    class MaxPool2d3(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.MaxPool2d(kernel_size=[4, 4], padding=2, stride=2)

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected3:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 6, 6), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 6, 6), dtype="float32") = R.nn.max_pool2d(
                    input_1,
                    pool_size=[4, 4],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[2, 2, 2, 2],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                gv: R.Tensor((1, 3, 6, 6), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(MaxPool2d(), input_info, {}, expected1)
    verify_model(MaxPool2d2(), input_info, {}, expected2)
    verify_model(MaxPool2d3(), input_info, {}, expected3)


def test_adaptive_avgpool2d():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class AdaptiveAvgPool2d(Module):
        def __init__(self):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool2d([10, 10])

        def forward(self, input):
            return self.pool(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    input_1, output_size=[10, 10], layout="NCHW", out_layout="NCHW"
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(AdaptiveAvgPool2d(), input_info, {}, expected1)


def test_flatten():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class Flatten(Module):
        def __init__(self):
            super().__init__()
            self.f = torch.nn.Flatten(1, -1)

        def forward(self, input):
            return self.f(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((300,), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((300,), dtype="float32") = R.flatten(input_1)
                gv: R.Tensor((300,), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Flatten(), input_info, {}, expected1)


def test_batchnorm2d():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class BatchNorm2d(Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, input):
            return self.bn(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
            w3: R.Tensor((3,), dtype="float32"),
            w4: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((1, 3, 10, 10), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                    R.Tensor((3,), dtype="float32"),
                ) = R.nn.batch_norm(
                    input_1,
                    w1,
                    w2,
                    w3,
                    w4,
                    axis=1,
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                lv1: R.Tensor((1, 3, 10, 10), dtype="float32") = lv[0]
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv1
                R.output(gv)
            return gv

    model = BatchNorm2d()
    binding = {
        "w1": model.bn.weight.numpy(),
        "w2": model.bn.bias.numpy(),
        "w3": model.bn.running_mean.numpy(),
        "w4": model.bn.running_var.numpy(),
    }
    verify_model(BatchNorm2d(), input_info, binding, expected1)


def test_embedding():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([4], "int64")}

    class Embedding(Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(10, 3)

        def forward(self, input):
            return self.embedding(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((4,), dtype="int64"), w1: R.Tensor((10, 3), dtype="float32")
        ) -> R.Tensor((4, 3), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((4,), dtype="int32") = R.astype(input_1, dtype="int32")
                lv1: R.Tensor((4, 3), dtype="float32") = R.take(w1, lv, axis=0)
                gv: R.Tensor((4, 3), dtype="float32") = lv1
                R.output(gv)
            return gv

    model = Embedding()
    binding = {"w1": model.embedding.weight.numpy()}
    verify_model(model, input_info, binding, expected1)


def test_dropout():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class Dropout(Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, input):
            return self.dropout(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = input_1
                R.output(gv)
            return gv

    verify_model(Dropout(), input_info, {}, expected1)


def test_layernorm():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class LayerNorm(Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm((10, 10))

        def forward(self, input):
            return self.ln(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((10, 10), dtype="float32"),
            w2: R.Tensor((10, 10), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.layer_norm(
                    input_1,
                    w1,
                    w2,
                    axes=[-2, -1],
                    epsilon=1e-05,
                    center=True,
                    scale=True,
                )
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    model = LayerNorm()
    binding = {
        "w1": model.ln.weight.numpy(),
        "w2": model.ln.bias.numpy(),
    }
    verify_model(LayerNorm(), input_info, binding, expected1)


def test_silu():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class SiLU(Module):
        def __init__(self):
            super().__init__()
            self.silu = torch.nn.SiLU()

        def forward(self, input):
            return self.silu(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.silu(input_1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(SiLU(), input_info, {}, expected1)


def test_groupnorm():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class GroupNorm(Module):
        def __init__(self):
            super().__init__()
            self.gn = torch.nn.GroupNorm(3, 3)

        def forward(self, input):
            return self.gn(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32"),
            w1: R.Tensor((3,), dtype="float32"),
            w2: R.Tensor((3,), dtype="float32"),
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 1, 10, 10), dtype="float32") = R.reshape(
                    input_1, (1, 3, 1, 10, 10)
                )
                lv1: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.mean(
                    lv, axis=[2, 3, 4], keepdims=True
                )
                lv2: R.Tensor((1, 3, 1, 10, 10), dtype="float32") = R.subtract(lv, lv1)
                lv3: R.Tensor((1, 3, 1, 10, 10), dtype="float32") = R.multiply(lv2, lv2)
                lv4: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.sum(
                    lv3, axis=[2, 3, 4], keepdims=True
                )
                lv5: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.divide(lv4, R.const(100.0))
                lv6: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.add(lv5, R.const(1e-05))
                lv7: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.sqrt(lv6)
                lv8: R.Tensor((1, 3, 1, 10, 10), dtype="float32") = R.divide(lv2, lv7)
                lv9: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.reshape(w1, (1, 3, 1, 1, 1))
                lv10: R.Tensor((1, 3, 1, 1, 1), dtype="float32") = R.reshape(w2, (1, 3, 1, 1, 1))
                lv11: R.Tensor((1, 3, 1, 10, 10), dtype="float32") = R.multiply(lv8, lv9)
                lv12: R.Tensor((1, 3, 1, 10, 10), dtype="float32") = R.add(lv11, lv10)
                lv13: R.Tensor((1, 3, 10, 10), dtype="float32") = R.reshape(lv12, (1, 3, 10, 10))
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv13
                R.output(gv)
            return gv

    model = GroupNorm()
    binding = {
        "w1": model.gn.weight.numpy(),
        "w2": model.gn.bias.numpy(),
    }
    verify_model(model, input_info, binding, expected1)


def test_softmax():
    torch.set_grad_enabled(False)
    torch.random.manual_seed(0)

    input_info = {"input_1": ([1, 3, 10, 10], "float32")}

    class Softmax(Module):
        def __init__(self):
            super().__init__()
            self.sm = torch.nn.Softmax(dim=1)

        def forward(self, input):
            return self.sm(input)

    @tvm.script.ir_module
    class expected1:
        @R.function
        def main(
            input_1: R.Tensor((1, 3, 10, 10), dtype="float32")
        ) -> R.Tensor((1, 3, 10, 10), dtype="float32"):
            # block 0
            with R.dataflow():
                lv: R.Tensor((1, 3, 10, 10), dtype="float32") = R.nn.softmax(input_1, axis=1)
                gv: R.Tensor((1, 3, 10, 10), dtype="float32") = lv
                R.output(gv)
            return gv

    verify_model(Softmax(), input_info, {}, expected1)


if __name__ == "__main__":
    # tvm.testing.main()
    test_conv()
    test_linear()
    test_relu()
    test_maxpool2d()
    test_adaptive_avgpool2d()
    test_flatten()
    test_batchnorm2d()
    test_embedding()
    test_dropout()
    test_layernorm()
    test_silu()
    test_groupnorm()
    test_softmax()
