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
"""A fallback schedule rule for GPU operators."""

from typing import List, Union, Tuple, Optional
from functools import reduce
import tvm
from tvm import tir
from tvm.tir import Schedule, Block
from tvm._ffi import get_global_func
from tvm.target import Target
from tvm.tir.schedule import BlockRV

from ..base import BlockInfo, ScheduleRule, try_inline
from .fallback import _max_threads_per_block


class DecodeGEMV(ScheduleRule):
    def __init__(self) -> None:
        super().__init__()
        self.is_trivial_binding = get_global_func("tir.schedule.IsTrivialBinding")
        self.is_reduction = get_global_func("tir.schedule.IsReductionBlock")
        self.get_block_realize = get_global_func("tir.schedule.GetBlockRealize")

    def detect_gemv(
        self, sch: Schedule, scope_block_rv: BlockRV, block_rv: BlockRV
    ) -> Tuple[List[tir.Var], Optional[tir.Var]]:
        if not self.is_trivial_binding(sch, block_rv):
            return None
        if not self.is_reduction(sch, block_rv, scope_block_rv):
            return None

        block: tir.Block = sch.get(block_rv)
        block_realize: tir.BlockRealize = self.get_block_realize(sch, block_rv)
        block_iters = block.iter_vars

        var_range_map = {iter.var: iter.dom for iter in block_iters}
        var_var_map = dict()
        for iter, binding in zip(block_iters, block_realize.iter_values):
            assert isinstance(binding, tir.Var)
            var_var_map[iter.var] = binding

        print(var_range_map)
        print(var_var_map)

        # C[S0] = C[S0] + f(A_i[S_i, R]), S_i >= S_{i+1}
        # reduce to (appromximately if we ignore smaller buffer accesses)
        # C[S0] = C[S0] + A_0[S_0, R], which is just a reduction
        if not isinstance(block.body, tir.BufferStore) or not isinstance(block.body.value, tir.Add):
            return None

        lhs = block.body.value.a
        rhs = block.body.value.b
        if not isinstance(lhs, tir.BufferLoad):
            lhs, rhs = rhs, lhs
        if not isinstance(lhs, tir.BufferLoad):
            return None

        # TODO: consider visit the body to collect buffer access
        reads = sorted(
            block.reads,
            key=lambda read: reduce(lambda x, y: x * y, read.buffer.shape),
            reverse=True,
        )

        # reads[0] is the buffer that decides the iteration space
        access = list()
        for range in reads[0].region:
            if range.extent != 1:
                return None
            access.append(range.min)
        index = reads[0].buffer.offset_of(access)
        assert len(index) == 1
        index = index[0]
        res = tvm.arith.normalize_to_iter_sum(index, var_range_map)
        assert isinstance(res, tvm.arith.IterSumExpr)
        if res.base != 0:
            return None

        # lhs and rhs use the same set of spatial variables
        lhs_vars = set()
        for value in lhs.indices:
            assert isinstance(value, tir.Var)
            assert value in var_range_map
            if not (var_range_map[value].extent == 1 and var_range_map[value].min == 0):
                lhs_vars.add(value)

        loop_order = list()
        rhs_vars = list()
        for split in res.args:
            assert isinstance(split.source.source, tir.Var)
            if split.source.source in lhs_vars:
                rhs_vars.append(split.source.source)
            loop_order.append(var_var_map[split.source.source])

        if len(lhs_vars) != len(rhs_vars):
            return None

        return loop_order

    def apply(  # pylint: disable=too-many-locals
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Union[None, tir.Schedule, List[tir.Schedule]]:
        if target.kind.name == "cuda":
            len_tx = 256
            unroll_depth = 256
        else:
            len_tx = 64
            unroll_depth = 64

        def _inline_all_spatial():
            blocks = []
            spatial_blocks = []
            for block in sch.get_child_blocks(sch.get_block("root")):
                block = BlockInfo(sch, block)
                if block.is_spatial():
                    spatial_blocks.append(block)
                elif spatial_blocks:
                    blocks.extend(try_inline(sch, spatial_blocks))
                    blocks.append(block)
                    spatial_blocks = []
                else:
                    blocks.append(block)
            if spatial_blocks:
                blocks.extend(try_inline(sch, spatial_blocks))
            return blocks

        sch = tir.Schedule(func)
        blocks = _inline_all_spatial()
        sch.mod.show(black_format=False)
        assert len(blocks) > 0

        pattern = self.detect_gemv(sch, sch.get_block("root"), blocks[0].block)
        if pattern is None:
            print("Mismatch")
            return None

        loops = sch.get_loops(blocks[0].block)
        
        
        print(pattern)
        return sch
