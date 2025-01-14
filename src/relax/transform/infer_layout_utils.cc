/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "infer_layout_utils.h"

namespace tvm {
namespace relax {

using tir::IterVar;
using tir::Layout;

Layout TransposeLike(const Layout& input, const Layout& src, const Layout& dst) {
  ICHECK(src.ndim() == dst.ndim() && input.ndim() == src.ndim())
      << "Layouts must have the same size";
  std::vector<IterVar> axes;
  for (size_t i = 0; i < src.ndim(); ++i) {
    axes.push_back(input->axes[src.IndexOf(dst[i])]);
  }
  return Layout(axes);
}

String TransposeStrLike(const String& input, const Layout& src, const Layout& dst) {
  ICHECK(src.ndim() == dst.ndim() && input.size() == src.ndim())
      << "Layouts must have the same size";
  std::string axes;
  for (size_t i = 0; i < src.ndim(); ++i) {
    axes.push_back(input.at(src.IndexOf(dst[i])));
  }
  return axes;
}

int FindAxis(const Layout& dst, int axis) {
  axis = (axis + dst.ndim()) % dst.ndim();
  return dst.name().find('A' + axis);
}

Layout InitialLayout(int ndim) {
  if (ndim == kUnknownNDim) {
    return Layout("");
  }
  ICHECK(ndim > 0 && ndim <= 26) << "Only support up to 26 dimensions";
  return Layout("ABCDEFGHIJKLMNOPQRSTUVWXYZ").SubLayout(0, ndim);
}

NLayout InitialNLayout(const StructInfo& sinfo) {
  auto fmapleaf = [&](const StructInfo& sinfo) -> NLayout {
    if (const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
      return NLayout(InitialLayout(tensor_sinfo->ndim));
    }
    LOG(FATAL) << "Cannot get layout for " << sinfo;
    return Layout::Undef();
  };
  return MapToNestedMsg<Layout>(sinfo, fmapleaf);
}

NLayout InitialNLayout(const Expr& expr) { return InitialNLayout(GetStructInfo(expr)); }

NLayout GetNLayout(const VarLayoutMap& var_layout_map, const Expr& arg) {
  auto fmapleaf = [&](const Expr& expr) -> NLayout {
    if (const auto* var = expr.as<VarNode>()) {
      auto it = var_layout_map.find(GetRef<Var>(var));
      if (it != var_layout_map.end()) {
        return (*it).second;
      }
    } else if (const auto* constant = expr.as<ConstantNode>()) {
      return InitialLayout(constant->data.Shape().size());
    }
    LOG(FATAL) << "Cannot get layout for " << expr;
    return Layout::Undef();
  };
  return MapToNestedMsg<Layout>(arg, fmapleaf);
}

Layout GetLayout(const VarLayoutMap& var_layout_map, const Expr& arg) {
  NLayout nlayout = GetNLayout(var_layout_map, arg);
  ICHECK(nlayout.IsLeaf()) << "Cannot get layout for " << arg;
  return nlayout.LeafValue();
}

bool NoDesiredLayout(const Call& call, const Map<String, Array<String>>& desired_layouts) {
  const OpNode* op_node = call->op.as<OpNode>();
  if (op_node == nullptr) return false;
  const auto& it = desired_layouts.find(op_node->name);
  return it == desired_layouts.end();
}

}  // namespace relax
}  // namespace tvm
