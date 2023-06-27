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
# pylint: disable=missing-docstring
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
# pylint: disable=missing-docstring
from tvm import dlight as dl
from tvm.ir import assert_structural_equal
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.target import Target


def test_decode_gemv():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def func(W: T.Buffer((4096, 512), "uint32"), S: T.Buffer((4096, 128), "float16"), V: T.Buffer((1, 1, 4096), "float16"), C: T.Buffer((1, 1, 4096), "float16")):
            T.func_attr({"global_symbol": "main", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            B = T.alloc_buffer((4096, 4096), "float16")
            for i, j in T.grid(4096, 4096):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(W[v_i, v_j // 8], S[v_i, v_j // 32])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(W[v_i, v_j // 8], T.Cast("uint32", v_j % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * S[v_i, v_j // 32]
            for i0, i1, i2, k in T.grid(1, 1, 4096, 4096):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                    T.reads(V[v_i0, v_i1, v_k], B[v_i2, v_k])
                    T.writes(C[v_i0, v_i1, v_i2])
                    with T.init():
                        C[v_i0, v_i1, v_i2] = T.float16(0)
                    C[v_i0, v_i1, v_i2] = C[v_i0, v_i1, v_i2] + V[v_i0, v_i1, v_k] * B[v_i2, v_k]


    @I.ir_module
    class After:
        @T.prim_func
        def main(
            A: T.Buffer((1, 32, 1, 128), "float16"),
            C: T.Buffer((1, 1, 4096), "float16"),
        ):
            T.func_attr({"tir.is_scheduled": 1})
            # with T.block("root"):
            for i_j_k_fused_0 in T.thread_binding(4, thread="blockIdx.x"):
                for i_j_k_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
                    with T.block("T_reshape"):
                        vi = T.axis.spatial(1, 0)
                        vj = T.axis.spatial(1, 0)
                        vk = T.axis.spatial(4096, i_j_k_fused_0 * 1024 + i_j_k_fused_1)
                        T.reads(A[0, vk % 4096 // 128, 0, vk % 128])
                        T.writes(C[vi, vj, vk])
                        C[vi, vj, vk] = A[0, vk % 4096 // 128, 0, vk % 128]
    # fmt: on

    target = Target("nvidia/geforce-rtx-3090-ti")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.DecodeGEMV())(Before)  # pylint: disable=not-callable
    mod.show(black_format=False)
    # assert_structural_equal(mod, After)


if __name__ == "__main__":
    test_decode_gemv()
